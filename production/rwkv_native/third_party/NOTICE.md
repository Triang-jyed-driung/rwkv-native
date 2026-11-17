# Bundled source origins

CUTLASS/CUTE headers come from NVIDIA CUTLASS v4.1.0
(https://github.com/NVIDIA/cutlass/tree/v4.1.0).
The upstream archive SHA256 is
8d4675b11e9e5207e3940eaac0f46db934ada371cbb3627c9fda642d912b6230.
See cutlass/LICENSE.txt and copyright notices in the headers.

The single-row FFN key and grouped input-projection kernels were adapted from
TileLang 0.1.15 static SM80 CUDA exports. They retain the required MMA helpers
and do not require TileLang at runtime.

Grouped W/A/G/(V) expansion was adapted from FlashRWKV2's
flashrwkv2/tmix/wkv_prepare/tilelang.py, source SHA256
87f61eace5d54959e7469315f57194a37e0ea41199aa1997172223c1d7aae1a3.
The CUDA exports use FP16 operands, FP32 outputs and a synchronous SM80 schedule.

Attention mixing and grouped input-projection scheduling follow Albatross's
rkv_lowrank_pre_executor_kernel.

The sparse FFN down projection in csrc/sparse_ffn.cu adapts Albatross
faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu, commit
545dd1870ea6cf424d7f7d4da16b7f87491405db. It retains the ballot/prefix compaction
and coalesced nonzero-feature reads, and changes activation input, operand types,
accumulation, output reduction, and tail handling.

The standalone CUDA projection in csrc/flash_linear.cu translates the SM80
copy pipeline, MMA ordering and FullCol warp placement of FlashRWKV2
flashrwkv2/_internal/linear.py (commit 311d3c0ebf1ebcf5a62853c45c43b8ead3fa1c37).
It supports FP16/BF16 operands and an optional fused ReLU-squared packing
epilogue. It has no TileLang runtime dependency.

The float-input staging kernel in csrc/wkv_prefetch.cuh adapts multi-token
prefetch from FlashRWKV2 flashrwkv2/tmix/wkv7/tilelang.py (same commit above).
It retains rwkv-native's balanced pair-product reduction tree, FP32 decay,
state update order, arbitrary masks, and configurable checkpoint spacing.
