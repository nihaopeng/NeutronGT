#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cusparse.h>

#include <vector>

namespace {

const char* cusparse_status_name(cusparseStatus_t status) {
  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "CUSPARSE_STATUS_ZERO_PIVOT";
    case CUSPARSE_STATUS_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_NOT_SUPPORTED";
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
      return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    default:
      return "CUSPARSE_STATUS_UNKNOWN";
  }
}

#define CUSPARSE_CHECK(expr) \
  do { \
    cusparseStatus_t _status = (expr); \
    TORCH_CHECK( \
        _status == CUSPARSE_STATUS_SUCCESS, \
        "cuSPARSE error ", \
        cusparse_status_name(_status), \
        " (code ", \
        static_cast<int>(_status), \
        ") at ", \
        #expr); \
  } while (0)

void check_csr_tensor(const torch::Tensor& tensor, const char* name, torch::ScalarType expected_dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == expected_dtype, name, " has unexpected dtype");
}

}  // namespace

std::vector<torch::Tensor> csr_spgemm_cuda(
    torch::Tensor rowptr_a,
    torch::Tensor col_a,
    torch::Tensor val_a,
    torch::Tensor rowptr_b,
    torch::Tensor col_b,
    torch::Tensor val_b) {
  check_csr_tensor(rowptr_a, "rowptr_a", torch::kInt32);
  check_csr_tensor(col_a, "col_a", torch::kInt32);
  check_csr_tensor(val_a, "val_a", torch::kFloat32);
  check_csr_tensor(rowptr_b, "rowptr_b", torch::kInt32);
  check_csr_tensor(col_b, "col_b", torch::kInt32);
  check_csr_tensor(val_b, "val_b", torch::kFloat32);
  TORCH_CHECK(rowptr_a.device() == col_a.device() && rowptr_a.device() == val_a.device(), "A CSR tensors must share a device");
  TORCH_CHECK(rowptr_b.device() == col_b.device() && rowptr_b.device() == val_b.device(), "B CSR tensors must share a device");
  TORCH_CHECK(rowptr_a.device() == rowptr_b.device(), "A and B must be on the same CUDA device");

  const auto device = rowptr_a.device();
  const c10::cuda::CUDAGuard device_guard(device);

  const int64_t m = rowptr_a.numel() - 1;
  const int64_t k = rowptr_b.numel() - 1;
  const int64_t n = k;
  const int64_t nnz_a = col_a.numel();
  const int64_t nnz_b = col_b.numel();

  auto index_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto value_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(device);

  auto rowptr_c = torch::zeros({m + 1}, index_options);

  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t mat_a = nullptr;
  cusparseSpMatDescr_t mat_b = nullptr;
  cusparseSpMatDescr_t mat_c = nullptr;
  cusparseSpGEMMDescr_t spgemm_desc = nullptr;

  CUSPARSE_CHECK(cusparseCreate(&handle));
  CUSPARSE_CHECK(cusparseSetStream(handle, c10::cuda::getCurrentCUDAStream(device.index()).stream()));

  CUSPARSE_CHECK(cusparseCreateCsr(
      &mat_a,
      m,
      k,
      nnz_a,
      rowptr_a.data_ptr<int>(),
      col_a.data_ptr<int>(),
      val_a.data_ptr<float>(),
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateCsr(
      &mat_b,
      k,
      n,
      nnz_b,
      rowptr_b.data_ptr<int>(),
      col_b.data_ptr<int>(),
      val_b.data_ptr<float>(),
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateCsr(
      &mat_c,
      m,
      n,
      0,
      rowptr_c.data_ptr<int>(),
      nullptr,
      nullptr,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F));
  CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemm_desc));

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr cusparseOperation_t op_a = CUSPARSE_OPERATION_NON_TRANSPOSE;
  constexpr cusparseOperation_t op_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
  constexpr cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG3;
  constexpr float chunk_fraction = 0.2f;

  size_t buffer_size1 = 0;
  size_t buffer_size2 = 0;
  size_t buffer_size3 = 0;

  CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
      handle,
      op_a,
      op_b,
      &alpha,
      mat_a,
      mat_b,
      &beta,
      mat_c,
      CUDA_R_32F,
      alg,
      spgemm_desc,
      &buffer_size1,
      nullptr));
  auto buffer1 = torch::empty({static_cast<int64_t>(buffer_size1)}, byte_options);
  CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
      handle,
      op_a,
      op_b,
      &alpha,
      mat_a,
      mat_b,
      &beta,
      mat_c,
      CUDA_R_32F,
      alg,
      spgemm_desc,
      &buffer_size1,
      buffer_size1 > 0 ? buffer1.data_ptr() : nullptr));

  CUSPARSE_CHECK(cusparseSpGEMM_estimateMemory(
      handle,
      op_a,
      op_b,
      &alpha,
      mat_a,
      mat_b,
      &beta,
      mat_c,
      CUDA_R_32F,
      alg,
      spgemm_desc,
      chunk_fraction,
      &buffer_size3,
      nullptr,
      &buffer_size2));
  auto buffer3 = torch::empty({static_cast<int64_t>(buffer_size3)}, byte_options);
  CUSPARSE_CHECK(cusparseSpGEMM_estimateMemory(
      handle,
      op_a,
      op_b,
      &alpha,
      mat_a,
      mat_b,
      &beta,
      mat_c,
      CUDA_R_32F,
      alg,
      spgemm_desc,
      chunk_fraction,
      &buffer_size3,
      buffer_size3 > 0 ? buffer3.data_ptr() : nullptr,
      &buffer_size2));

  auto buffer2 = torch::empty({static_cast<int64_t>(buffer_size2)}, byte_options);
  CUSPARSE_CHECK(cusparseSpGEMM_compute(
      handle,
      op_a,
      op_b,
      &alpha,
      mat_a,
      mat_b,
      &beta,
      mat_c,
      CUDA_R_32F,
      alg,
      spgemm_desc,
      &buffer_size2,
      buffer_size2 > 0 ? buffer2.data_ptr() : nullptr));

  int64_t c_rows = 0;
  int64_t c_cols = 0;
  int64_t c_nnz = 0;
  CUSPARSE_CHECK(cusparseSpMatGetSize(mat_c, &c_rows, &c_cols, &c_nnz));

  auto col_c = torch::empty({c_nnz}, index_options);
  auto val_c = torch::empty({c_nnz}, value_options);
  CUSPARSE_CHECK(cusparseCsrSetPointers(
      mat_c,
      rowptr_c.data_ptr<int>(),
      c_nnz > 0 ? col_c.data_ptr<int>() : nullptr,
      c_nnz > 0 ? val_c.data_ptr<float>() : nullptr));
  CUSPARSE_CHECK(cusparseSpGEMM_copy(
      handle,
      op_a,
      op_b,
      &alpha,
      mat_a,
      mat_b,
      &beta,
      mat_c,
      CUDA_R_32F,
      alg,
      spgemm_desc));

  CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemm_desc));
  CUSPARSE_CHECK(cusparseDestroySpMat(mat_a));
  CUSPARSE_CHECK(cusparseDestroySpMat(mat_b));
  CUSPARSE_CHECK(cusparseDestroySpMat(mat_c));
  CUSPARSE_CHECK(cusparseDestroy(handle));

  return {rowptr_c, col_c, val_c};
}
