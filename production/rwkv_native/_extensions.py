"""One registry for the CUDA extensions shipped with the production package."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Extension:
    sources: tuple[str, ...]
    includes: tuple[str, ...] = ()
    libraries: tuple[str, ...] = ()


EXTENSIONS = {
    "core": Extension(
        (
            "binding.cpp",
            "wkv.cu",
            "backward_ops.cu",
            "mix_backward.cu",
            "key_backward.cu",
            "readout_forward.cu",
            "key_forward.cu",
        )
    ),
    "ops": Extension(("ops.cpp", "ops.cu")),
    "frontend": Extension(("frontend.cpp", "frontend.cu")),
    "attention": Extension(("attention.cpp", "attention.cu")),
    "fixed": Extension(("fixed_gemm.cpp", "fixed_gemm.cu"), ("third_party/cutlass/include",)),
    "lt": Extension(("gemm_lt.cpp",), libraries=("cublasLt",)),
    "adjoints": Extension(("adjoints.cpp",), libraries=("cublas",)),
    "gradient_pack": Extension(("gradient_pack.cu",)),
    "ffn_key": Extension(("ffn_key.cpp", "ffn_key.cu")),
    "rank_group": Extension(("rank_group.cpp", "rank_group.cu")),
    "pre_group": Extension(("pre_group.cpp", "pre_group.cu")),
}


@lru_cache(None)
def extension(kind):
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent
    spec = EXTENSIONS[kind]
    return load(
        name="rwkv_native_production_" + kind,
        sources=[str(root / "csrc" / name) for name in spec.sources],
        extra_include_paths=[str(root / path) for path in spec.includes],
        extra_ldflags=["-l" + name for name in spec.libraries],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo"],
        with_cuda=True,
    )
