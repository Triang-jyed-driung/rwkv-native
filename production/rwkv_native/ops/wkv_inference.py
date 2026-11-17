"""Inference-only WKV neighborhoods with the same FP32 arithmetic as core."""

import torch

from .._extensions import extension


def inference_schedule(batch, tokens, heads):
    if batch == tokens == 1 and heads in (40,64):
        return "prefetched"
    if batch * heads >= 128 and tokens >= 8:
        return "prefetched"
    if 32 <= batch * heads < 128 and tokens >= 256:
        return "register"
    return None


def attention_inference(schedule, r, k, v, expanded, first_value, matrix, mask, parameters, heads):
    """Fuse preparation; retain the small-head register recurrence schedule."""
    if torch.is_grad_enabled():
        raise RuntimeError("fused WKV inference requires gradients disabled")
    batch, tokens = r.shape[:2]
    shape = (batch, tokens, heads, 64)
    blend = first_value is not None
    first_value = first_value if blend else v
    vectors = [value.view(shape).contiguous() for value in
               (r, expanded[0], k, v, expanded[1], expanded[3] if blend else expanded[1])]
    parameters = [p.flatten().contiguous() for p in parameters]
    if schedule == "prefetched":
        if matrix.stride(0) < 0 or matrix.stride()[1:] != (4096, 64, 1):
            matrix = matrix.contiguous()
        packed, final = extension("wkv_inference").prefetched_readout(
            vectors, matrix, mask.contiguous(), expanded[2].view(shape).contiguous(),
            parameters, first_value.view(shape).contiguous(), blend)
        return packed.view(batch, tokens, heads * 64), final, first_value
    if schedule != "register":
        raise ValueError("unknown WKV inference schedule")
    y, final, key, value = extension("wkv_cooperative_inference").run(
        vectors, matrix, mask.contiguous(), parameters[3:], first_value.view(shape).contiguous(), blend, 4)
    flat = lambda value: value.view(batch, tokens, heads * 64)
    packed = extension("core").readout_packed_forward(
        [flat(y), r, flat(key), flat(value), *parameters[:3], expanded[2]], heads)
    return packed, final, first_value
