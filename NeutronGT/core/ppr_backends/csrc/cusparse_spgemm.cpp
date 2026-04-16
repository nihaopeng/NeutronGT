#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> csr_spgemm_cuda(
    torch::Tensor rowptr_a,
    torch::Tensor col_a,
    torch::Tensor val_a,
    torch::Tensor rowptr_b,
    torch::Tensor col_b,
    torch::Tensor val_b);

std::vector<torch::Tensor> csr_spgemm(
    torch::Tensor rowptr_a,
    torch::Tensor col_a,
    torch::Tensor val_a,
    torch::Tensor rowptr_b,
    torch::Tensor col_b,
    torch::Tensor val_b) {
  return csr_spgemm_cuda(rowptr_a, col_a, val_a, rowptr_b, col_b, val_b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csr_spgemm", &csr_spgemm, "cuSPARSE CSR SpGEMM");
}
