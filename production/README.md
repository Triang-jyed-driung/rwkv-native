# RWKV7 CUDA implementation

## Installation and usage

Requires Python >= 3.10, PyTorch >= 2.12, Ninja, a compatible CUDA toolkit, and an NVIDIA GPU with SM80 or newer. CUDA extensions compile on first use and are cached afterward.

Install from the repository root:

```sh
python -m pip install --no-build-isolation ./production
```

Basic usage:

```python
import torch
from rwkv_native import FastRWKV7

model = FastRWKV7.from_checkpoint("model.pth")
tokens = torch.tensor([[1, 2, 3]], device="cuda")
with torch.no_grad():
    logits, state = model(tokens, last_only=True)
    logits, state = model(tokens[:, -1:], state, last_only=True)
```

Training uses standard autograd. For long sequences, enable block gradient checkpointing explicitly, then compute vocabulary projections and losses in chunks to avoid retaining vocabulary logits for every token:

```python
hidden, state = model.forward_features(tokens, checkpoint_blocks=True)
logits = model.project(hidden[:, :256], model.weight("head.weight").T)
```

## API conventions

- `tokens` is a CUDA integer tensor with shape `[B, T]`. Model weights default to FP16; BF16 is also supported.
- `attention_mask` is a boolean tensor with shape `[B, T]`. Positions marked `False` skip state updates and produce zero logits. Left padding, right padding, and internal gaps are supported.
- `last_only=True` returns logits for the last physical position with shape `[B, 1, V]`. If that position is masked, its logits are zero.
- State is passed explicitly through return values. Input state is not modified. Use `state.detach()` to stop gradients between segments.
- Activations, residuals, and WKV state use FP32. WKV state has layout `[B, H, K, V]`, with a contiguous last axis and head size 64.
- `forward_features` returns features after `ln_out`, including masked positions. Select valid positions when computing loss.
- Only first-order gradients are supported. Operators call CUDA extensions directly through `autograd.Function`; they have no `torch.compile` registration.

## Code entry points

- `rwkv_native/model.py`: model, attention/channel blocks, and explicit checkpointing.
- `rwkv_native/state.py`: recurrent state and token shift that skips invalid tokens.
- `rwkv_native/ops/policy.py`: projection backend selection by weight shape.
- `rwkv_native/ops/attention.py` and `channel.py`: attention input projections and fused channel operations.
- `rwkv_native/ops/normalization.py`: normalization, activations, and shared normalization/shift backward operations.
- `rwkv_native/ops/gemm.py`: autograd-enabled matrix multiplication and forward-only `project_rows`.
- `rwkv_native/_extensions.py`: CUDA extension source registry and loader.
- `rwkv_native/csrc/`: CUDA/C++ operators.
- `rwkv_native/third_party/`: CUTLASS/CUTE headers, licenses, and source notices.

Fused paths are enabled by default. Some SM120/FP16 projections use fixed cuBLASLt configurations; other cases use CUDA fallbacks. Set `backend="fixed"` to force fixed CUDA projections. Fusion switches are intended for diagnostics; avoid changing summation order based on sequence length.

Runtime dependencies are PyTorch, Ninja, CUDA/cuBLAS, and the bundled headers. FlashRWKV2, Albatross, TileLang, and archived research code are not imported or executed at runtime.

## Validation

Run `python -m pytest -q` from this directory on a CUDA GPU. Tests cover forward and backward operations, prefill/decode, masked shifts, batch reordering, and explicit checkpointing.

Use `tools/capture_model.py --checkpoint model.pth --output result.json` to capture real-model fingerprints. It covers five input groups for the 1.5B model, per-layer state, and one full set of parameter gradients.

Historical benchmarks and accuracy results are stored locally in `../artifacts/`; archived implementation code is in `../research/`. These materials are ignored by default and may be absent from a fresh checkout.
