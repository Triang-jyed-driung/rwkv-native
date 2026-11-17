# RWKV Native

Fast RWKV7 training and inference, with the control of PyTorch and the efficiency of native CUDA.

Stream tokens through explicit recurrent state, train longer sequences with gradient checkpointing, and keep vocabulary memory under control with streamed loss. FP16/BF16 weights pair with FP32 activations and state, while fused kernels reduce memory traffic and launch overhead.

## Performance

**1.50x FlashRWKV2 across six BF16 tracks**, with training reaching a **2.62x track speedup**. Fused projections, cooperative WKV kernels, and streamlined backward passes put more GPU time into useful computation.

| Workload | Consistency | Speedup vs. FlashRWKV2 | Main optimizations |
| --- | --- | --- | --- |
| BF16 decode | `"strict"` | **1.22x** | Fused attention/FFN frontends and grouped low-rank projections reduce kernel launches and intermediate tensors. |
| BF16 decode | `"none"` | **1.32x** | Shape-tuned projections and sparse FFN down projection reduce weight traffic for small batches. |
| BF16 prefill | `"strict"` | **1.08x** | Cooperative WKV uses two blocks per head for more parallelism; multi-token prefetch and fused FFN operations reduce memory traffic. |
| BF16 prefill | `"none"` | **1.07x** | Cooperative FP32 WKV and shape-tuned dense projections, with fused activation packing and residual writeback. |
| BF16 training | `"strict"` | **2.36x** | FP32 WKV backward with checkpoint reconstruction and segmented scans; streamed vocabulary loss and grouped head gradients. |
| BF16 training | `"none"` | **2.62x** | Tensor Core WKV backward for long unmasked sequences; grouped vocabulary gradients and separate attention/FFN replay reduce memory pressure. |

RTX 4090, CUDA Graph, 2.9B/7.2B models. Speedups are per-track geometric means over the recorded full-model and partial-model scopes, including labeled estimates. See the [performance guide](production/PERFORMANCE.md) for representative timings, sampling methods, and workload definitions.

## Quick start

Requires Python >= 3.10, PyTorch >= 2.8, Ninja, a compatible CUDA Toolkit, and an NVIDIA GPU with compute capability SM80 or newer. CUDA extensions compile on first use, so a C++ compiler and `nvcc` must be available.

Install a suitable PyTorch build and setuptools >= 68 first, then install from the repository root:

```sh
python -m pip install --no-build-isolation './production[dev]'
```

Omit `[dev]` if you only need the runtime.

```python
import torch
from rwkv_native import FastRWKV7

# Supply your own original-format RWKV7 weight dictionary.
model = FastRWKV7.from_checkpoint("model.pth", precision="bf16")
tokens = torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int64)
with torch.inference_mode():
    logits, state = model(tokens, last_only=True)
    next_token = logits[:, -1].argmax(-1, keepdim=True)
    logits, state = model(next_token, state, last_only=True)
```

The numbers are token IDs. Tokenization, sampling, and stopping conditions belong to the calling application.

- [Usage and API reference](production/README.md): inference, training, masks, state, arguments, and return values.
- [Performance, optimizations, and validation](production/PERFORMANCE.md): the six benchmark tracks, representative results, and reproducible speed, correctness, accuracy, and consistency checks.

The `.gitignore` allowlist defines the release contents: the complete runtime and CUDA sources, required headers and licenses, essential validation code, and these guides. Experiment outputs, raw CSV tables, working notes, weights, and nonessential tests remain local.

Run the self-contained core regression suite on a CUDA GPU:

```sh
cd production
python -m pytest -q
```

Some tests use synthetic models with production hidden widths and require several GB of free GPU memory. The performance guide also explains validation with real weights.

Third-party origins and licenses are documented in [NOTICE.md](production/rwkv_native/third_party/NOTICE.md). A license for this project's own code has not yet been specified.
