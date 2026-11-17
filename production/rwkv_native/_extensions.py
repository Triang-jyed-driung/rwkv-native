"""One registry for the CUDA extensions shipped with the production package."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Extension:
    sources: tuple[str, ...]
    includes: tuple[str, ...] = ()
    libraries: tuple[str, ...] = ()
    cuda_flags: tuple[str, ...] = ()


EXTENSIONS = {
    "wkv_cooperative_inference": Extension(("wkv_cooperative_inference.cu",)),
    "head_gradient_pack": Extension(("head_gradient_pack.cu",)),
    "scaled_half": Extension(("scaled_half_pack.cu",), libraries=("cublas",)),
    "wkv_simt_backward": Extension(("wkv_simt_backward.cu",)),
    "wkv_cooperative": Extension(("wkv_cooperative.cu",)),
    "wkv_mma_backward": Extension(("wkv_mma_backward.cu",), cuda_flags=("--use_fast_math", "--maxrregcount=240")),
    "decode_inputs": Extension(("decode_inputs.cu",)),
    "decode_inputs_wide": Extension(("decode_inputs_wide.cu",)),
    "decode_rank": Extension(("decode_rank.cu",)),
    "decode_up": Extension(("decode_up.cu",)),
    "frontend_prefill": Extension(("frontend_prefill.cu",)),
    "ffn_down": Extension(("ffn_down.cu",), ("third_party/cutlass/include",)),
    "wkv_inference": Extension(("wkv_inference.cpp", "wkv_inference_prefetch.cu", "wkv_inference_register.cu")),
    "head_ce": Extension(("head_ce.cu",)),
    "flash_linear": Extension(("flash_linear.cu",)),
    "embedding": Extension(("embedding.cu",)),
    "sparse_ffn": Extension(("sparse_ffn.cu",)),
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
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo", *spec.cuda_flags],
        with_cuda=True,
    )
