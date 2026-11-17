# Performance, optimizations, and validation

RWKV Native brings faster training, efficient streaming decode, and reproducible execution to one PyTorch interface. The recorded BF16 suite delivers a **1.50x six-track geometric mean over FlashRWKV2**, with training leading the gains.

The six tracks combine decode, prefill, and training with `"strict"` and `"none"` consistency. Each runs in FP16 and BF16; Flash comparisons use BF16 only. Choose `"strict"` for stable forward arithmetic across batches and streaming calls, or `"none"` for shape-dependent performance choices. Both retain FP32 model activations and recurrent state.

This guide explains the optimizations behind those gains, highlights representative results recorded on November 17, 2026, and gives you the commands to measure your own setup.

## Measurement scope

The recorded environment was an RTX 4090 with 24 GB VRAM, PyTorch 2.14.1+cu130, and CUDA 13.0, primarily using CUDA Graph replay. Models were 2.9B (32 layers, C=2560, H=40) and 7.2B (32 layers, C=4096, H=64). Each measured model scope includes embedding, blocks, final normalization, and the vocabulary head.

| Workload | Fixed `(B,T)` shapes | Timed work |
| --- | --- | --- |
| Decode | `(1,1)`, `(192,1)` | One complete model forward step with recurrent state. |
| Prefill | `(1,256)`, `(1,4096)` | Complete sequence forward. |
| Rectangular prefill | `(20,16)`, `(60,64)` | Dense rectangular input without a mask; included in the prefill track. |
| Training | `(16,1024)`, `(1,16384)` | Forward, all-token mean cross-entropy, and backward; excludes optimizer.step. |

Latency measures execution after compilation, warmup, and graph capture. Optional profiling runs separately. Defaults are one warmup and one timed sample, labeled `single_sample`. Increase `--repeats` for a sampled P10: the 10th percentile on the faster side of the latency distribution.

`"strict"` inference benchmarks project all tokens; `"none"` inference benchmarks project the last token. The API's `last_only` option lets applications choose either output scope independently. Speedups compare candidate and baseline within each track, using matching vocabulary-head work.

Initial states are zero and remain unmodified between calls. Benchmarks use batch-broadcast zero-state storage to avoid unnecessary resident copies; full output state is still computed and retained. Training uses checkpointing and a default vocabulary chunk of 128 flattened token rows.

## Representative results

These examples use the 2.9B model in BF16. Decode and prefill include all 32 layers. Training measures the first two blocks together with embedding, final normalization, and the vocabulary head. Flash denotes the matching FlashRWKV2 BF16 baseline.

| Case | Native latency | Flash latency | Interpretation |
| --- | --- | --- | --- |
| `"strict"` decode, `B=1,T=1` | 7.50 ms | 7.52 ms | Inherited P10 for an unchanged path; essentially tied for one request. |
| `"none"` decode, `B=1,T=1` | 6.37 ms | 7.51 ms | Inherited P10, about 1.18x; small batches benefit from fusion and sparse FFN. |
| `"strict"` prefill, `B=1,T=256` | 15.75 ms | 17.20 ms | One candidate observation against baseline P10, about 1.09x. |
| `"none"` prefill, `B=1,T=256` | 16.32 ms | 17.05 ms | One candidate observation against baseline P10, about 1.04x. |
| `"strict"` training, two blocks, `B=1,T=16384` | 465.20 ms | 1222.09 ms | One measured candidate sample, about 2.63x, including CE and backward. |
| `"none"` training, two blocks, target `B=1,T=16384` | **Estimated 441.00 ms** | 1221.76 ms | Actually measured 330.75 ms at `T=12288`, then scaled by token count; the 2.77x ratio is an estimate. |

The BF16 six-track equal-weight geometric mean against frozen Flash baselines reached **1.504x**. The FP16/BF16 twelve-track mean against frozen native baselines reached **1.583x**.

| Aggregate composition | Coverage |
| --- | --- |
| New measurements | 24 single-sample timings |
| Reused measurements | 40 P10 timings for unchanged execution paths |
| Training estimates | 6 cases scaled by token count |
| Inference estimates | 8 cases with extrapolated block-body time and measured embedding/head time |
| Aggregation | Geometric mean within each track, then equal-weight geometric mean across tracks; includes the labeled estimates |

BF16 track means against Flash were approximately: `"strict"`/`"none"` decode **1.216x / 1.324x**, prefill **1.081x / 1.075x**, and training **2.359x / 2.622x**. Training accounts for the largest gains; long prefill remains a tighter comparison. Remeasure on the target device when choosing deployment settings.

## What improves each track

### `"strict"` decode: reduce launch overhead while preserving per-row arithmetic

Single-token latency is especially sensitive to kernel launches and weight reads. The implementation combines attention normalization, shift, mixing, and eligible input projections, and groups W/A/G/V low-rank expansion for supported widths and ranks. The FFN frontend and single-row up projection also have specialized paths to reduce temporary tensors and repeated reads.

`"strict"` projections use fixed CUDA reductions, with constrained cuBLASLt configurations on selected devices and shapes. Equivalent tile selection preserves each row's arithmetic order. WKV keeps FP32 state, with vectorized prefetch where the batch shape permits. The result is an optimized path that shares the same forward contract across training, prefill, and decode.

### `"none"` decode: exploit sparse activations at small row counts

`"none"` mode permits shape-dependent projection selection. For no-grad calls with very few token rows, the FFN down projection exploits zeros from ReLU-squared activation. Ballot/prefix compaction identifies nonzero features so the kernel reads only the required weight regions, reducing bandwidth pressure for single-request decode.

Dispatch reserves sparse compression for small row counts, where reduced weight traffic pays off, and selects dense computation for larger workloads. Outputs and recurrent state remain FP32. Training uses the corresponding autograd-capable implementation.

### `"strict"` prefill: increase WKV parallelism and fuse preparation/writeback

With a small batch and a long sequence, the number of heads can provide too few GPU blocks. Eligible cooperative WKV paths assign two CTAs to different value columns of one head, exposing more spatial parallelism while retaining FP32 state updates and reduction order. Register paths use multi-token prefetch to reduce recurring load stalls.

Shape-specific attention paths reduce preparation, recurrence, and readout overhead. Readout conversion can be combined with packing for the next projection. FFN up projection fuses activation/packing, while down projection adds the residual at writeback, avoiding large intermediates. `"strict"` fusion preserves forward arithmetic. Masked inputs and other shapes use their corresponding general paths.

### `"none"` prefill: shape-specific dense projections with accuracy fallbacks

Long inputs benefit from dense matrix throughput. Projection tiles depend on row count, kernels read supported weight layouts directly, and activation, dtype conversion, and residual addition are fused around projections to avoid redundant FP32 intermediates.

Whole-model logits and state accuracy guide dispatch. FP16 uses its established down-projection reduction, while BF16 C=2560 and C=4096 receive tailored choices. Forward WKV uses validated FP32 arithmetic. This combination targets projection throughput while preserving the model's numerical behavior.

### `"strict"` training: FP32 backward with bounded recomputation

WKV backward reconstructs state between periodic checkpoints, keeping recurrent-state storage compact. Long sequences use FP32 SIMT backward, with segmented affine scans when batch-times-head parallelism is small. Separate accuracy requirements for backward allow its scheduling to improve independently of the `"strict"` forward contract.

Block checkpointing reduces activation peaks; large BF16 inputs replay attention and FFN separately. Streamed vocabulary CE recomputes logits and writes input-gradient slices directly. Large BF16 projection backward operations independently range-scale operands before converting them to FP16 for faster GEMMs, while accumulation and activation gradients remain FP32.

### `"none"` training: Tensor Core backward and grouped vocabulary gradients

Eligible long, unmasked sequences use WKV Tensor Core backward with BF16 operands, FP16 checkpoint storage, and FP32 output gradients. Short sequences and explicitly masked calls retain FP32 backward. This increases matrix throughput within the allowed gradient-error budget while keeping forward state FP32.

Both training modes share vocabulary-head improvements. For large BF16 inputs, backward groups the default 128-row forward chunks into 1024/2048-row groups to reduce small GEMMs and repeated packing. Only the consumed FP16 component is produced, and head parameter gradients accumulate in FP32 across chunks before one conversion to the parameter dtype. Gains come from less memory traffic, allocation, and launch overhead as well as faster recurrence.

## Correctness, accuracy, and consistency

Three complementary checks support the implementation:

- **Correctness:** independent recurrence, outputs and state, parameter and initial-state gradients, masking, checkpoint replay, and streamed loss.
- **Accuracy:** low-precision output and gradient RMSE against an independent FP64 reference.
- **Consistency:** exact `"strict"`-forward comparisons across the batch, streaming, and gradient-mode transformations in the API contract.

Optimization acceptance compares candidate RMSE with frozen-native RMSE against the same FP64 reference, separately for the relevant outputs. Relative error-growth limits are 1.2x for `"strict"` and 2.0x for `"none"`. The report exposes per-parameter and state errors alongside aggregate gradient RMSE. A zero-error baseline is checked by requiring zero candidate error for the corresponding metric.

A representative long-sequence check used the 7.2B model's first two blocks at T=16384: aggregate gradient RMSE over 69 parameters was **0.996153x** the frozen native error for `"strict"` and **0.996469x** for `"none"`. Logits and all three state error ratios were **1.0**. Both configurations maintained the reference accuracy level in this long-context case.

### Self-contained core regression suite

From `production/`:

```sh
python -m pytest -q
```

`pyproject.toml` explicitly selects the release's core regression suite.

| Validation code | Purpose |
| --- | --- |
| [test_strict_forward_contract.py](tests/test_strict_forward_contract.py) | Production hidden widths and low-rank shapes; training/inference, batch positions/permutations, single-token/full-sequence calls, and padding; checks logits and every layer's three state groups. |
| [test_masked_backward.py](tests/test_masked_backward.py) | Compares all parameter and initial-state gradients with physically compacted inputs, including internal gaps and fully masked recurrence. |
| [test_wkv.py](tests/test_wkv.py), [wkv_reference.py](tests/wkv_reference.py) | Core recurrence against independent FP64 forward/backward, including tails, streaming, and checkpoint intervals. |
| [test_checkpoint.py](tests/test_checkpoint.py), [test_large_checkpoint.py](tests/test_large_checkpoint.py) | Outputs, state, and parameter gradients with and without ordinary block or large BF16 sublayer recomputation. |
| [test_head_loss.py](tests/test_head_loss.py), [test_training_benchmark_loss.py](tests/test_training_benchmark_loss.py) | Streamed CE, ignored targets, gradients, graph replay, and the benchmark loss objective. |
| [test_runtime_config.py](tests/test_runtime_config.py) | End-to-end behavior across precision/consistency settings, functional state, and gradient mode. |
| [test_inference_fp64_reference.py](tests/test_inference_fp64_reference.py), [test_training_fp64_reference.py](tests/test_training_fp64_reference.py) | Validates the reference itself through batch independence, an analytically simplified case, and finite-difference gradients. |

The suite generates its own weights, including production-width models, for repeatable validation straight from a fresh checkout.

### End-to-end accuracy with real weights

Run the following from `production/`. Set `MODEL` to your original-format RWKV7 checkpoint. The commands generate their references and reports in the local `OUT` directory.

```sh
export MODEL=/absolute/path/to/model.pth
export OUT=../artifacts/validation
mkdir -p "$OUT"

# Build an independent CPU FP64 inference reference, then validate the GPU model.
python tools/inference_fp64_check.py --checkpoint "$MODEL" \
  --reference "$OUT/inference.pt" --build-reference --batch 1 --tokens 257 --layers 2
python tools/inference_fp64_check.py --checkpoint "$MODEL" \
  --reference "$OUT/inference.pt" --output "$OUT/inference-bf16-strict.json" \
  --batch 1 --tokens 257 --layers 2 --precision bf16 --consistency "strict"

# Training reference: independent CPU FP64 body, CUDA FP64 head, and autograd.
python tools/training_fp64_check.py --checkpoint "$MODEL" \
  --reference "$OUT/training.pt" --build-reference --batch 1 --tokens 257 --layers 2
python tools/training_fp64_check.py --checkpoint "$MODEL" \
  --reference "$OUT/training.pt" --output "$OUT/training-bf16-strict.json" \
  --batch 1 --tokens 257 --layers 2 --precision bf16 --consistency "strict"
```

Start with two layers to check the environment; increase layers, batch, and length as memory and time permit. Reference and validation must use the same checkpoint, B/T, and layer count. Training also requires matching `--head-rows`. Change precision to fp16 and consistency to `"none"`, using distinct output names, to cover all four configurations. The same reference can be reused.

Inference reports logits and per-state-group RMSE. Training additionally reports loss, aggregate and per-parameter gradient RMSE, and checks the gradient set and finiteness. References promote original weights to FP64, so errors include runtime weight conversion. Inference `--stage-weights` loads parameters layer by layer to reduce resident memory; it is only for untimed accuracy validation.

The accuracy CLIs produce the detailed measurements for acceptance: logits, state, and gradient errors. For before/after comparisons, generate both reports against identical references and scopes, then apply the relative-error thresholds above to each output. Successful execution confirms the computation and built-in gradient-set and finiteness assertions.

### `"strict"` forward consistency with real weights

After generating the references above:

```sh
python tools/inference_forward_invariance.py --checkpoint "$MODEL" \
  --reference "$OUT/inference.pt" --output "$OUT/inference-consistency.json" \
  --precision bf16
python tools/training_forward_invariance.py --checkpoint "$MODEL" \
  --reference "$OUT/training.pt" --output "$OUT/training-consistency.json" \
  --precision bf16
```

The tools read B/T, layer count, and token IDs from the reference, then assert training/inference, batch separation, and streaming forward equality. Repeat with fp16 for the other precision. Nonzero initial states, batch permutations, and masks receive systematic coverage in the synthetic-model regressions above.

## Remeasuring end-to-end speed

After correctness checks pass, from `production/`, with `MODEL` and `OUT` set as above:

```sh
python tools/benchmark_model.py --checkpoint "$MODEL" \
  --output "$OUT/speed-inference.json" --workload inference \
  --precisions bf16 --consistencies "strict" "none" --execution graph \
  --cases decode:1:1 prefill:1:256 --warmup 3 --repeats 20

# Use the same two-layer training scope for each comparison.
python tools/benchmark_model.py --checkpoint "$MODEL" \
  --output "$OUT/speed-training.json" --workload train --layers 2 \
  --precisions bf16 --consistencies "strict" "none" --execution graph \
  --cases train:1:16384 --loss-chunk 128 --warmup 3 --repeats 20
```

Omit `--cases` for all fixed shapes in the selected workload, and omit `--precisions` for both precisions. `--execution eager` reports forward/backward separately; graph mode reports complete replay. JSON results include raw samples, the latency statistic, memory peaks, actual layer count, head scope, and source fingerprints. The compatibility key `p10` contains only the single observation when `latency_statistic` is `single_sample`.

Automatic memory handling can select the first two blocks for training on the largest model and records the selected depth and reason. If model loading reaches the memory limit, choose a smaller scope with `--layers`. For directly comparable results, match layers, B/T, vocabulary scope, loss chunk size, and latency statistic. Report partial-depth measurements, reduced-length measurements, and extrapolated estimates with their respective scopes.

For Flash comparisons, `--implementation flash --precisions bf16` uses the bundled [flash_model.py](tools/flash_model.py) adapter with a separate compatible FlashRWKV2 installation (the recorded source revision was `311d3c0ebf1ebcf5a62853c45c43b8ead3fa1c37`). The native benchmark uses the package's own runtime. Record Flash's tuning status and validate the adapter numerically with matching FP32 state and functional state-copy semantics inside timing. Its license is [FlashRWKV2.LICENSE](tools/FlashRWKV2.LICENSE).
