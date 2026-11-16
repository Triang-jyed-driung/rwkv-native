"""CUDA packing and reconstruction around low-precision adjoint GEMMs."""

import torch

from .._extensions import extension


def pack(
    x: torch.Tensor, g: torch.Tensor, need_dw: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(extension("gradient_pack").pack(x, g, need_dw))


def combine(products: torch.Tensor, scale: torch.Tensor, parts: int) -> torch.Tensor:
    return extension("gradient_pack").combine(products, scale, parts)
