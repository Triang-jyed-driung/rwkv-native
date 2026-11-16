from pathlib import Path

import torch

import rwkv_native

assert Path(rwkv_native.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


def tiny_weights(width=64):
    torch.manual_seed(17)
    C = width
    weights = {"emb.weight": torch.randn(128, C) * 0.1, "head.weight": torch.randn(128, C) * 0.1}
    for name in [
        "ln_out",
        "blocks.0.ln0",
        *[f"blocks.{i}.ln{j}" for i in range(2) for j in (1, 2)],
    ]:
        weights[name + ".weight"] = torch.ones(C)
        weights[name + ".bias"] = torch.zeros(C)
    for i in range(2):
        p = f"blocks.{i}.att."
        for name in "rwkvag":
            weights[p + "x_" + name] = torch.rand(1, 1, C)
        for name in ("w", "a", "v", "g"):
            weights[p + name + "1"] = torch.randn(C, 8) * 0.1
            weights[p + name + "2"] = torch.randn(8, C) * 0.1
            if name != "g":
                weights[p + name + "0"] = torch.randn(1, 1, C) * 0.1
        for name in ("receptance", "key", "value", "output"):
            weights[p + name + ".weight"] = torch.randn(C, C) * 0.1
        for name in ("k_k", "k_a"):
            weights[p + name] = torch.ones(1, 1, C)
        weights[p + "r_k"] = torch.randn(1, C) * 0.1
        weights[p + "ln_x.weight"] = torch.ones(C)
        weights[p + "ln_x.bias"] = torch.zeros(C)
        f = f"blocks.{i}.ffn."
        weights[f + "x_k"] = torch.rand(1, 1, C)
        weights[f + "key.weight"] = torch.randn(4 * C, C) * 0.1
        weights[f + "value.weight"] = torch.randn(C, 4 * C) * 0.1
    return weights
