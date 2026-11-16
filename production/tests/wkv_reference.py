"""FP64 test oracle; never imported by the runtime."""

import math

import torch


def wkv7_reference(r, w, k, v, a, b, state, attention_mask=None):
    dtype = state.dtype
    r, w, k, v, a, b = (x.to(dtype) for x in (r, w, k, v, a, b))
    outputs = []
    for t in range(r.shape[1]):
        decay = torch.exp(-math.exp(-0.5) * torch.sigmoid(w[:, t]))
        sa = (state * a[:, t, :, :, None]).sum(-2)
        candidate = (
            state * decay[:, :, :, None]
            + sa[..., None, :] * b[:, t, :, :, None]
            + k[:, t, :, :, None] * v[:, t, :, None, :]
        )
        active = None if attention_mask is None else attention_mask[:, t, None, None]
        state = candidate if active is None else torch.where(active[..., None], candidate, state)
        output = (state * r[:, t, :, :, None]).sum(-2)
        if active is not None:
            output = torch.where(active, output, 0.0)
        outputs.append(output)
    return torch.stack(outputs, dim=1), state
