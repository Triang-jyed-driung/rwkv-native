# RWKV native

Native PyTorch/CUDA training and inference for RWKV7, with explicit recurrent state, fused operators, and block gradient checkpointing.

## Requirements and installation

Requires Python >= 3.10, PyTorch >= 2.12, Ninja, a compatible CUDA toolkit, and an NVIDIA GPU with SM80 or newer. CUDA extensions compile on first use and are cached afterward.

Install from the repository root:

```sh
python -m pip install --no-build-isolation ./production
```

See [production/README.md](production/README.md) for usage, API conventions, and implementation details.

## Development

Install test dependencies:

```sh
python -m pip install --no-build-isolation './production[dev]'
```

Run tests on a CUDA GPU:

```sh
cd production
python -m pytest -q
```

To collect tests without executing them, use `python -m pytest --collect-only -q` from `production/`. Collection alone does not validate GPU results.

For real-model output, state, and gradient capture, see `production/tools/capture_model.py --help`.

## Repository layout

- `production/`: runtime package, CUDA/C++ sources, bundled headers, tests, and tools.
- `research/`: historical research archive documentation.
- `artifacts/`: local validation results, benchmarks, and build output; ignored by default.

The local `research/legacy.zip` archive is ignored by default and can be distributed separately. Historical materials are not runtime dependencies. Caches, virtual environments, model weights, and build output are also ignored.

## Third-party notices

See [NOTICE.rst](production/rwkv_native/third_party/NOTICE.rst) and the accompanying license files for bundled source origins and licenses. Preserve these notices when distributing the bundled headers. A license for this project's own code has not yet been specified.
