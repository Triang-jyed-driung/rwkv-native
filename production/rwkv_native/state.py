"""Explicit, immutable-by-convention recurrent state and valid-token shifting."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class State:
    attention: list[torch.Tensor]
    channel: list[torch.Tensor]
    wkv: list[torch.Tensor]  # FP32 [B,H,K,V]; the value axis is contiguous.

    def detach(self):
        return State(
            *[[x.detach() for x in group] for group in (self.attention, self.channel, self.wkv)]
        )


def token_shift(x, previous, mask):
    """Read the previous valid token; masked positions do not update state."""
    batch, tokens, channels = x.shape
    if tokens == 1:
        return previous[:, None], torch.where(mask[:, 0, None], x[:, 0], previous)
    positions = torch.arange(1, tokens + 1, device=x.device).expand(batch, tokens)
    last_valid = torch.where(mask, positions, 0).cummax(1).values
    previous_indices = F.pad(last_valid[:, :-1], (1, 0))
    history = torch.cat((previous[:, None], x), 1)
    shifted = history.gather(1, previous_indices[..., None].expand(batch, tokens, channels))
    final = history.gather(1, last_valid[:, -1:, None].expand(batch, 1, channels)).squeeze(1)
    return shifted, final
