# RWKV7 usage and API reference

Build RWKV7 applications with a compact PyTorch interface and optimized CUDA execution. Bring your model weights and tokenizer, then use the same model for prompt processing, streaming generation, and training. The public exports are `FastRWKV7`, `RuntimeConfig`, `State`, and `wkv7`.

See [Performance and validation](PERFORMANCE.md) for optimization details and reproducible checks.

## Installation

Requirements: Python >= 3.10, PyTorch >= 2.8, setuptools >= 68, Ninja, a compatible CUDA Toolkit, a C++ compiler, and an NVIDIA SM80+ GPU. Install the appropriate PyTorch build first and check that `nvcc --version` works. From the repository root:

```sh
python -m pip install --no-build-isolation ./production
```

For tests, install `./production[dev]` instead. CUDA extensions compile on demand and are cached for subsequent calls. Warm up the model before measuring inference latency. GPU memory holds weights, recurrent state, activations, and training gradients.

The PyTorch 2.8 baseline follows a static review of the Python and C++ extension APIs. It includes [CUDA matrix multiplication with FP32 output](https://docs.pytorch.org/docs/2.8/generated/torch.mm.html) for FP16/BF16 operands. The recorded 192-test regression run used PyTorch 2.14.1+cu130 on an RTX 4090.

## Prefill and token-by-token inference

```python
import torch
from rwkv_native import FastRWKV7

model = FastRWKV7.from_checkpoint(
    "model.pth", device="cuda", precision="bf16", consistency="strict"
)
# Replace these example IDs with output from the model's tokenizer.
tokens = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int64)
with torch.inference_mode():
    logits, state = model(tokens, last_only=True)  # Process the entire prompt.
    generated = []
    for _ in range(8):
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_token)
        logits, state = model(next_token, state, last_only=True)
    generated_ids = torch.cat(generated, dim=1)
```

`logits` contains unnormalized vocabulary scores. This example uses greedy selection; the caller supplies sampling and stopping rules. Continue with the new token and returned state to carry the conversation forward efficiently.

Use `torch.no_grad()` or `torch.inference_mode()` to select efficient inference execution. `model.eval()` controls the standard PyTorch module mode.

## Construction and configuration

```python
FastRWKV7.from_checkpoint(path, **kwargs)
FastRWKV7(weights, *, device="cuda", dtype=None, precision=None,
          consistency="strict", backend="auto", adjoint_mode="store",
          fused_frontend=True, equivalent_tiles=True, fused_attention=True,
          fused_rank_group=True, fused_pre_group=True, sparse_ffn=True)
```

`path` is a local PyTorch weight file loaded with `torch.load(weights_only=True, mmap=True)`. It contains the original RWKV7 parameter dictionary, including `emb.weight`, `head.weight`, consecutively numbered blocks starting at `blocks.0`, and normalization parameters. For a training snapshot with an outer `state_dict` key, extract the original-format parameter dictionary and pass it to `FastRWKV7(weights, ...)`.

`weights` accepts the same name-to-tensor mapping with the original matrix orientations. The hidden width is a multiple of 64, and the head size is 64.

| Argument | Default | Meaning |
| --- | --- | --- |
| `device` | `"cuda"` | Parameter device; `"cuda:1"` also works. Inputs and state must use the same device. |
| `precision` | `None`, resolved to `"fp16"` | `"fp16"` or `"bf16"`; controls parameters and projection operands. |
| `dtype` | `None` | Compatibility alias using `torch.float16` or `torch.bfloat16`; conflicting precision settings raise an error. |
| `consistency` | `"strict"` | `"strict"` enforces the forward contract below; `"none"` permits shape-dependent arithmetic. |
| `backend` | `"auto"` | Automatic projection selection; `"fixed"` forces fixed CUDA projections for diagnostics. |
| `adjoint_mode` | `"store"` | Optimized projection backward; `"fp32"` selects FP32 projection backward for diagnostics. |
| `fused_frontend` | `True` | Allows fused normalization, token shift, and channel frontend operations. |
| `fused_attention` | `True` | Allows fused attention input preparation and projection. |
| `fused_rank_group` | `True` | Allows grouped low-rank expansion for supported shapes. |
| `fused_pre_group` | `True` | Allows grouped input projections for supported shapes. |
| `equivalent_tiles` | `True` | Allows projection tile schedules that preserve the required arithmetic. |
| `sparse_ffn` | `True` | Allows sparse FFN inference for a few token rows in `"none"` mode with a non-fixed backend. |

A fusion switch permits an optimization; actual dispatch also depends on shape, dtype, backend, and gradient mode. Leave these switches at their defaults for normal use.

Both precisions support training and inference. Model activations, residuals, logits, and recurrent state are FP32. PyTorch's gradient switch selects training or inference execution, independently of the `train()` / `eval()` flag.

`RuntimeConfig(precision="fp16", consistency="strict")` is immutable; its `.dtype` property returns the corresponding PyTorch dtype. Configure the model through the individual constructor arguments above. Inspect `model.config` for resolved settings and `model.backend_report` for projection choices.

`model.to(dtype=torch.bfloat16)` updates the configuration and rebuilds the projection policy. The model runs on CUDA only, with FP16 or BF16 parameters. `model.width`, `model.heads`, and `model.layers` expose the hidden width, head count, and layer count.

## Forward calls, masks, and state

B denotes batch size, T token count, C hidden width, V vocabulary size, and H=C/64. B and T must be positive.

```python
logits, next_state = model(tokens, state=None, attention_mask=None, last_only=False)
hidden, next_state = model.forward_features(
    tokens, state=None, attention_mask=None, checkpoint_blocks=False
)
state = model.initial_state(batch)
```

`last_only` and `checkpoint_blocks` are keyword-only arguments.

| Value | Type and shape | Contract |
| --- | --- | --- |
| `tokens` | CUDA int64 `[B,T]` | Valid token IDs are in `[0,V)`. |
| `state` | `State` or None | None starts each sequence from zero state. |
| `attention_mask` | CUDA bool `[B,T]` or None | True is valid; False skips state updates. Left/right padding and internal gaps are supported. |
| `logits` | FP32 `[B,T,V]` | Explicitly masked positions return zero logits. |
| Logits with `last_only=True` | FP32 `[B,1,V]` | Projects only the last physical position; all input tokens still participate in recurrence. |
| `hidden` | FP32 `[B,T,C]` | Output after `ln_out`; use the mask to select valid rows. |
| `next_state` | `State` | State after the valid tokens at every layer; preserves the input state. |

`last_only=True` selects the final physical position. A masked final column returns zero logits. With right padding, retrieve the last valid logits explicitly:

```python
mask = torch.tensor([[True, True, False]], device="cuda")
with torch.no_grad():
    logits, state = model(tokens, attention_mask=mask)
    positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
    last = positions.masked_fill(~mask, -1).amax(dim=1)
    # Each row in this example has a valid token. Handle fully masked rows separately.
    last_logits = logits[torch.arange(tokens.shape[0], device=tokens.device), last]
```

`State(attention, channel, wkv)` contains three lists, each with one tensor per layer:

| Field | Per-layer tensor | Contents |
| --- | --- | --- |
| `attention` | FP32 `[B,C]` | Normalized attention input of the most recent valid token. |
| `channel` | FP32 `[B,C]` | Normalized FFN input of the most recent valid token. |
| `wkv` | FP32 `[B,H,64,64]` | Recurrent matrix; the last two axes are key and value. |

State matches the model, batch, and device. When reordering a batch, apply the same indices to every tensor in all three lists. Start each independent request with None or a fresh zero state.

`state.detach()` returns state with its gradient history removed and shares tensor storage with the original. Clone individual tensors when you need independently editable copies.

For the same runtime environment, parameters, precision, and per-sequence initial state, `"strict"` mode guarantees bitwise-identical valid-token logits and final attention/channel/WKV state across:

- Gradients enabled versus disabled.
- Individual samples, different batch positions, and batch permutations with matching state permutations.
- Repeated single-token calls carrying state versus one complete-sequence call.
- An unmasked sequence versus the same sequence with masked prefix and suffix padding.

Backward gradients are validated by numerical accuracy. Choose `"strict"` for these forward invariances within a fixed GPU and software environment, or `"none"` for shape-dependent performance choices.

## Training and truncated backpropagation

```python
import torch
from rwkv_native import FastRWKV7

model = FastRWKV7.from_checkpoint("model.pth", precision="bf16")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# Replace with a dataset batch. Adjacent tokens form next-token prediction targets.
sequence = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.int64)
inputs, targets = sequence[:, :-1], sequence[:, 1:]
optimizer.zero_grad(set_to_none=True)
hidden, state = model.forward_features(inputs, checkpoint_blocks=True)
loss = model.training_loss(hidden, targets, chunk_rows=128)
loss.backward()
optimizer.step()
state = state.detach()  # Carry this only if the next segment continues the same sequence.
```

`checkpoint_blocks=True` recomputes block operations during backward to reduce saved activations. Large BF16 inputs replay attention and FFN separately to reduce their combined memory peak. Use `forward_features()` for checkpointed training; gradient-disabled calls take the inference path.

```python
loss = model.training_loss(hidden, targets, chunk_rows=128)
```

`hidden` is CUDA FP32 `[B,T,C]`, or flattened `[N,C]`. Targets are int64 on the same device with one element per hidden row, normally `[B,T]`. Values are in `[0,V)` or equal to `-100`. Prepare next-token targets by shifting the dataset sequence as shown above.

The result is scalar mean cross-entropy over targets other than `-100`. `chunk_rows` is a positive integer limiting flattened token rows per vocabulary chunk, independently of batch size. Logits are generated in chunks and recomputed during backward instead of retaining the complete `[B,T,V]` tensor.

Apply the loss mask with `targets = targets.masked_fill(~mask, -100)`, including any invalid prediction targets. Train on batches with at least one valid target; an all-ignored batch has an undefined mean (NaN) and should be skipped. The loss supports standard optimizers and first-order autograd.

Detaching state truncates backpropagation between segments. To propagate gradients across segments, keep the graph and finish the corresponding backward passes before updating parameters.

## Projection and parameter access

```python
weight = model.weight("head.weight")       # [V,C] parameter
logits = model.project(hidden, weight.T)    # [B,T,V]
backend = model.projection_backend(weight.T)
```

`weight(name)` accepts the original parameter name and returns an `nn.Parameter`. The internal `ParameterDict` encodes dots as `__`; therefore, `model.state_dict()` uses different keys from the original checkpoint format. To save weights that `from_checkpoint` can load, restore the original names:

```python
weights = {name.replace("__", "."): value.detach().cpu()
           for name, value in model.weights.items()}
torch.save(weights, "trained.pth")
```

`project(value, weight, *, relu_squared=False, activation=0)` accepts activations with last dimension K and a `[K,N]` weight, returning FP32 output with the same leading dimensions and last dimension N. Use the defaults for the vocabulary head. `relu_squared` and nonzero `activation` select internal fusion operations. `projection_backend(weight)` reports the backend selected for that shape.

## Standalone WKV

```python
from rwkv_native import wkv7

y, final = wkv7(r, w, k, v, a, b, state=None, attention_mask=None,
                checkpoint_size=16, approximate_backward=False)
```

The six inputs share shape `[B,T,H,64]`, dtype, and CUDA device. FP16, BF16, and FP32 are supported. Supply `w` before the decay transformation shown below.

At each valid token, `decay = exp(-exp(-0.5) * sigmoid(w))`. With S indexed by key K and value D, the recurrence is:

```text
q[D]    = sum_K S[K,D] * a[K]
S'[K,D] = S[K,D] * decay[K] + b[K] * q[D] + k[K] * v[D]
y[D]    = sum_K S'[K,D] * r[K]
```

State is FP32 `[B,H,64,64]`; None creates zero state. `final` has the same state shape and dtype. Output y has the input shape and dtype. The optional mask is bool `[B,T]` on the same device; False produces zero output and retains the previous state. Input state is not modified and supports first-order gradients.

`checkpoint_size` is an integer from 1 to 256, default 16, controlling state reconstruction during backward. `approximate_backward=True` permits a Tensor Core approximate backward for eligible long FP32-input sequences with the default checkpoint interval. It changes backward numerical behavior; leave it False for the standard standalone interface.

The custom operators integrate directly with PyTorch and support first-order gradients only. Execution options are eager mode and CUDA Graph replay.

## Implementation entry points

- [model.py](rwkv_native/model.py): public model and dispatch.
- [state.py](rwkv_native/state.py): explicit state and valid-token shift.
- [policy.py](rwkv_native/ops/policy.py): projection policy.
- [_extensions.py](rwkv_native/_extensions.py): bundled CUDA source registry and on-demand compilation.
- [NOTICE.md](rwkv_native/third_party/NOTICE.md): third-party origins and accompanying licenses.

The native runtime uses PyTorch, Ninja, CUDA/cuBLAS, and bundled headers. All required CUDA implementations ship with the package; FlashRWKV2 and TileLang Python packages are not required.
