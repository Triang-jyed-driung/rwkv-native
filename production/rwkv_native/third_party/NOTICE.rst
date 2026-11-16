Bundled source origins
======================

CUTLASS/CUTE headers come from NVIDIA CUTLASS v4.1.0
(https://github.com/NVIDIA/cutlass/tree/v4.1.0).
The upstream archive SHA256 is
8d4675b11e9e5207e3940eaac0f46db934ada371cbb3627c9fda642d912b6230.
See cutlass/LICENSE.txt and copyright notices in the headers.

The single-row FFN key and grouped input-projection kernels were adapted from
TileLang 0.1.15 static SM80 CUDA exports. They retain the required MMA helpers
and do not require TileLang at runtime. See LICENSE.TileLang.

Grouped W/A/G/(V) expansion was adapted from FlashRWKV2's
flashrwkv2/tmix/wkv_prepare/tilelang.py, source SHA256
87f61eace5d54959e7469315f57194a37e0ea41199aa1997172223c1d7aae1a3.
The CUDA exports use FP16 operands, FP32 outputs and a synchronous SM80 schedule.
See LICENSE.FlashRWKV2 and LICENSE.TileLang.

Attention mixing and grouped input-projection scheduling follow Albatross's
rkv_lowrank_pre_executor_kernel. See ALBATROSS_LICENSE and
references/Albatross-Apache-2.0.txt.

Original export paths and per-file hashes are preserved with the archived
implementation in research/legacy.zip at the project root.
