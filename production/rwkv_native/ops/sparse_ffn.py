"""Albatross-style exact-zero sparse FFN down projection for tiny inference batches."""

import torch

from .._extensions import extension


def sparse_ffn(value, weight):
    if torch.is_grad_enabled():
        raise RuntimeError("sparse FFN is an inference-only kernel; disable gradients")
    output = extension("sparse_ffn").run(
        value.reshape(-1, value.shape[-1]).contiguous(), weight)
    return output.view(*value.shape[:-1], weight.shape[1])
