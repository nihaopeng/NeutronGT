import os
import shutil
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils import cpp_extension

_EXTENSION_NAME = "torchgt_cusparse_spgemm"
_INT32_MAX = 2**31 - 1


def _candidate_cuda_homes():
    candidates = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    if cpp_extension.CUDA_HOME:
        candidates.append(Path(cpp_extension.CUDA_HOME))
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parents[1])
    for pattern in ("/usr/local/cuda", "/usr/local/cuda-*"):
        for path in Path("/").glob(pattern.lstrip("/")):
            candidates.append(path)
    unique = []
    seen = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _resolve_cuda_home() -> Path:
    for cuda_home in _candidate_cuda_homes():
        nvcc = cuda_home / "bin" / "nvcc"
        include_dir = cuda_home / "include" / "cusparse.h"
        if nvcc.exists() and include_dir.exists():
            os.environ.setdefault("CUDA_HOME", str(cuda_home))
            os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"
            return cuda_home
    raise RuntimeError(
        "APPNP cuSPARSE backend requires a CUDA toolkit with nvcc and cusparse.h available. "
        "Set CUDA_HOME/CUDA_PATH or install a CUDA toolkit that torch.utils.cpp_extension can find."
    )


@lru_cache(maxsize=1)
def load_cusparse_extension():
    cuda_home = _resolve_cuda_home()
    base_dir = Path(__file__).resolve().parent
    source_dir = base_dir / "csrc"
    sources = [
        str(source_dir / "cusparse_spgemm.cpp"),
        str(source_dir / "cusparse_spgemm.cu"),
    ]
    extra_include_paths = [str(source_dir), str(cuda_home / "include")]
    return cpp_extension.load(
        name=_EXTENSION_NAME,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=extra_include_paths,
        extra_ldflags=["-lcusparse"],
        verbose=False,
    )


def _require_cuda_tensor(tensor: torch.Tensor, name: str):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for cuSPARSE")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be a 1-D tensor")


def _prepare_index_tensor(tensor: torch.Tensor, name: str):
    _require_cuda_tensor(tensor, name)
    if tensor.numel() > 0 and int(tensor.max().item()) > _INT32_MAX:
        raise RuntimeError(f"{name} contains indices that exceed the cuSPARSE int32 limit")
    return tensor.to(dtype=torch.int32, copy=False).contiguous()


def _prepare_value_tensor(tensor: torch.Tensor, name: str):
    _require_cuda_tensor(tensor, name)
    return tensor.to(dtype=torch.float32, copy=False).contiguous()


def csr_spgemm(rowptr_a, col_a, val_a, rowptr_b, col_b, val_b):
    ext = load_cusparse_extension()
    rowptr_a_i32 = _prepare_index_tensor(rowptr_a, "rowptr_a")
    col_a_i32 = _prepare_index_tensor(col_a, "col_a")
    val_a_f32 = _prepare_value_tensor(val_a, "val_a")
    rowptr_b_i32 = _prepare_index_tensor(rowptr_b, "rowptr_b")
    col_b_i32 = _prepare_index_tensor(col_b, "col_b")
    val_b_f32 = _prepare_value_tensor(val_b, "val_b")

    rowptr_c, col_c, val_c = ext.csr_spgemm(
        rowptr_a_i32,
        col_a_i32,
        val_a_f32,
        rowptr_b_i32,
        col_b_i32,
        val_b_f32,
    )
    return rowptr_c.to(dtype=torch.long), col_c.to(dtype=torch.long), val_c
