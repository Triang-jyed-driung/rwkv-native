"""Fixed K arithmetic: validated cuBLASLt plans or portable SM80 CUDA GEMMs."""

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .._extensions import extension
from .adjoints import projection_backward
from .normalization import RELU_SQUARED, activation_backward


def _lt_config(index, *, tile, split_k=1, reduction=0, stages=35, workspace_bytes=0):
    """Metadata of a validated SM120 algorithm, checked before any retargeting."""
    return {
        "index": index,
        "algorithm": {
            "id": 67,
            "reduction": reduction,
            "split_k": split_k,
            "stages": stages,
            "tile": tile,
            "workspace_bytes": workspace_bytes,
        },
    }


# Logical [K,N] shapes. Keep these plans fixed across row counts: their K
# reduction order is part of the prefill/decode consistency contract.
_LT_PLANS = {
    "8192,2048": _lt_config(1, tile=33, split_k=8, reduction=2, workspace_bytes=6291456),
    "2048,8192": _lt_config(0, tile=320),
    "2048,2048": _lt_config(0, tile=33, split_k=8, reduction=2, workspace_bytes=6291456),
    "2048,96": _lt_config(4, tile=47, split_k=16, reduction=2, stages=36, workspace_bytes=589824),
    "2048,65536": _lt_config(3, tile=406),
    "2048,256": _lt_config(2, tile=15, split_k=16, reduction=2, workspace_bytes=1572864),
    "2048,64": _lt_config(1, tile=11, split_k=8, reduction=2, stages=36, workspace_bytes=196608),
    "96,2048": _lt_config(6, tile=316, stages=36),
    "256,2048": _lt_config(5, tile=314, stages=36),
    "64,2048": _lt_config(6, tile=314, stages=36),
}
# Equivalent small-M tile; metadata validation still applies to both plans.
_LT_EQUIVALENT = {
    "2048,2048": {
        **_lt_config(6, tile=400, split_k=8, reduction=2, workspace_bytes=6291456),
        "max_rows": 64,
    },
}


def selections(equivalent=False):
    return _LT_EQUIVALENT if equivalent else _LT_PLANS


_validated_lt_plans = {}
_retargeted_lt_plans = {}


def _weight_signature(w):
    return (w.device.index, w.dtype, tuple(w.shape), tuple(w.stride()))


def _is_capturing(device):
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def prepare_lt(w, equivalent=True):
    """Validate the whole weight shape before deciding its model-wide backend.

    Returns (supported, reason). A mismatch never changes only one row count
    to another implementation. Configure the model before graph capture.
    """
    if w.data_ptr() % 16:
        return False, "unaligned weight uses the fixed CUDA fallback"
    key = (_weight_signature(w), equivalent)
    if key in _validated_lt_plans:
        return True, "validated cuBLASLt configuration"
    capability = torch.cuda.get_device_capability(w.device)
    if capability == (8, 9) and w.dtype in (torch.float16, torch.bfloat16):
        return _prepare_sm89_lt(w, equivalent)
    if w.dtype != torch.float16 or capability != (12, 0):
        return False, "no validated cuBLASLt configuration for this architecture/dtype"
    shape = f"{w.shape[0]},{w.shape[1]}"
    selected = selections().get(shape)
    if selected is None:
        return False, "shape uses the fixed CUDA fallback"
    if _is_capturing(w.device):
        raise RuntimeError("Prepare the model before CUDA graph capture")
    canonical = torch.empty((96, w.shape[0]), device=w.device, dtype=w.dtype)
    candidates = extension("lt").candidates(canonical, w, 32 * 1024**2, 32)
    alternate = selections(True).get(shape) if equivalent else None
    chosen = []
    for selection in (selected, alternate):
        if selection is None:
            chosen.append(None)
            continue
        index = selection["index"]
        if index >= len(candidates) or candidates[index].info() != selection["algorithm"]:
            return False, "cuBLASLt heuristic metadata differs from the validated configuration"
        plan = candidates[index]
        # Validate workspace/shape support without running an uninitialized input.
        for rows in (1, 48, 64, 65, 96, 768):
            try:
                plan.retarget(canonical.new_empty(rows, w.shape[0]), w)
            except RuntimeError as exc:
                return False, f"cuBLASLt retarget validation failed: {exc}"
        chosen.append(plan)
    _validated_lt_plans[key] = (chosen[0], chosen[1], alternate["max_rows"] if alternate else 0)
    return True, "validated cuBLASLt configuration"


def _prepare_sm89_lt(w, equivalent):
    # Fixed algorithm across every row count, not shape-dependent heuristic
    # selection at execution time. Ordered split reduction is used for low rank.
    shape = tuple(w.shape)
    if shape not in ((2560,2560),(2560,10240),(2560,96)) or tuple(w.stride()) != (1,2560):
        return False, "shape/layout uses the fixed CUDA fallback"
    if _is_capturing(w.device):
        raise RuntimeError("Prepare the model before CUDA graph capture")
    splits, reduction = (8,2) if shape[1] == 96 else (1,0)
    canonical = torch.empty((256,shape[0]),device=w.device,dtype=w.dtype)
    candidates = extension("lt").candidates(canonical,w,32*1024**2,64)
    for seed in candidates:
        if seed.info()["id"] != 21:
            continue
        try:
            plan = seed.configure(splits,reduction,15,12)
            for rows in (1,17,64,65,192,320,768,4096):
                plan.retarget(canonical.new_empty(rows,shape[0]),w)
        except RuntimeError:
            continue
        _validated_lt_plans[(_weight_signature(w),equivalent)] = (plan,None,0)
        return True, "validated fixed SM89 cuBLASLt algorithm"
    return False, "fixed SM89 cuBLASLt algorithm unavailable; using fixed CUDA"


def _lt_plan(x, w, equivalent):
    key = (_weight_signature(w), equivalent)
    if key not in _validated_lt_plans:
        supported, reason = prepare_lt(w, equivalent)
        if not supported:
            raise RuntimeError(reason)
    base, alternate, max_rows = _validated_lt_plans[key]
    use_alternate = alternate is not None and x.shape[0] <= max_rows
    plan_key = (key, x.shape[0], use_alternate)
    if plan_key not in _retargeted_lt_plans:
        if _is_capturing(x.device):
            raise RuntimeError("Warm up this input shape before CUDA graph capture")
        _retargeted_lt_plans[plan_key] = (alternate if use_alternate else base).retarget(x, w)
    return _retargeted_lt_plans[plan_key]


def _fixed_schedule(k, n, rows):
    if k >= 4096:
        splits = 8
    elif n >= 32768:
        splits = 1
    elif n <= 512 and k >= 1024:
        splits = 8
    elif k >= 1024:
        splits = 4 if k == n else 2
    else:
        splits = 1
    while k % splits or (k // splits) % 32:
        splits //= 2
    tile = 32 if k <= 512 or n <= 512 else 64 if k == n or rows <= 64 else 128
    # Same MMA K order and split reduction, with fewer inactive M rows for
    # decode. K64 / three-stage cp.async also improves uncached weight reads.
    if rows <= 16 and k >= 1024 and n >= 1024 and (k // splits) % 64 == 0:
        tile = 16
    return splits, tile


_lt_chunk_rows = {}


def _lt_row_limit(x, w, equivalent):
    """Keep K arithmetic fixed while allowing larger M tiles and bounded scratch.

    cuBLASLt retargeting changes only M. Zero-workspace algorithms use all rows;
    split-K algorithms grow to 3072 rows (up to 256 MiB scratch). Unsupported
    row counts retain the verified 768-row path.
    """
    if x.shape[0] <= 768:
        return 768
    key = (_weight_signature(w), equivalent, x.shape[0])
    if key in _lt_chunk_rows:
        return _lt_chunk_rows[key]
    if _is_capturing(w.device):
        raise RuntimeError("Warm up GEMM row blocking before graph capture")
    canonical = _validated_lt_plans[(_weight_signature(w), equivalent)][0]
    desired = x.shape[0] if canonical.info()["workspace_bytes"] == 0 else min(3072, x.shape[0])
    while desired > 768:
        try:
            _lt_plan(x[:desired], w, equivalent)
            remainder = x.shape[0] % desired
            if remainder:
                _lt_plan(x[:remainder], w, equivalent)
            _lt_chunk_rows[key] = desired
            return desired
        except RuntimeError:
            desired = (desired // 1536) * 768
    _lt_chunk_rows[key] = 768
    return 768


def project_rows(x: torch.Tensor, w: torch.Tensor, backend: str) -> torch.Tensor:
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0] or min(w.shape) < 1:
        raise ValueError("expected [M,K] activation and nonempty [K,N] weight")
    if (
        x.dtype not in (torch.float16, torch.bfloat16)
        or w.dtype != x.dtype
        or w.device != x.device
        or not x.is_cuda
    ):
        raise ValueError("expected matching CUDA FP16/BF16 operands")
    if backend not in ("fixed", "lt", "lt_equivalent", "pre_group", "torch"):
        raise ValueError("invalid projection backend")
    if not x.shape[0]:
        return x.new_empty((0, w.shape[1]), dtype=torch.float32)
    if backend == "torch":
        return torch.mm(x, w, out_dtype=torch.float32)
    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    if backend == "pre_group":
        return extension("pre_group").run([x], [w.T.contiguous()], -1).view(x.shape[0], w.shape[1])
    if backend == "fixed":
        k, n = w.shape
        kp = (k + 31) // 32 * 32
        np = (n + 3) // 4 * 4
        # Padding is confined to real GEMM operands, not FP32 activation edges.
        xp = F.pad(x, (0, kp - k)) if kp != k else x
        wp = w.T.contiguous()
        if kp != k or np != n:
            wp = F.pad(wp, (0, kp - k, 0, np - n))
        if wp.data_ptr() % 16:
            wp = wp.clone()
        output = x.new_empty((x.shape[0],np),dtype=torch.float32)
        run = extension("fixed").run_out
        # Narrow low-rank projections benefit from fewer launches and concat
        # copies. Keep the wider split-K intermediates bounded: larger blocks
        # regress square projections and the 4096 -> 480 gate on SM89.
        block = 3072 if kp <= 512 or np <= 384 else 768
        # These shapes already retain split-K sums in registers. A single
        # prefill grid reuses weights across M tiles without split scratch.
        if 768 <= xp.shape[0] <= 4096 and (kp, np) in ((4096,4096),(4096,16384),(16384,4096)):
            block = xp.shape[0]
        offset = 0
        for part in xp.split(block):
            splits, tile = _fixed_schedule(kp, np, part.shape[0])
            run(part, wp, splits, tile, output[offset:offset+part.shape[0]])
            offset += part.shape[0]
        return output[:, :n].contiguous()
    if w.data_ptr() % 16:
        raise ValueError("unaligned weights require the fixed CUDA backend")
    equivalent = backend == "lt_equivalent"
    if (_weight_signature(w), equivalent) not in _validated_lt_plans:
        supported, reason = prepare_lt(w, equivalent)
        if not supported:
            raise RuntimeError(reason)
    if (x.shape[0] >= 1024 and tuple(w.shape) in ((2560,2560),(2560,10240))
            and w.T.is_contiguous() and torch.cuda.get_device_capability(w.device) == (8,9)):
        return extension("flash_linear").run(x,w.T,128,False)
    # Same sequential SM80 MMA arithmetic as the validated unsplit FFN key
    # plan. Its single-row schedule avoids the canonical large-row CTA tile.
    if x.shape[0] == 1 and tuple(w.shape) == (2048, 8192):
        return extension("ffn_key").run(x, w.T.contiguous())
    block = _lt_row_limit(x, w, equivalent)
    outputs = [_lt_plan(part, w, equivalent).run(part, w) for part in x.split(block)]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, 0)


class _LinearProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, activation, adjoint_mode, backend):
        # Preserve FP32 ReLU² arithmetic, but never widen an already-packed
        # projection operand only to narrow it back to the same dtype.
        value = x.float() if activation or x.dtype not in (torch.float32,torch.float16,torch.bfloat16) else x
        flat = value.reshape(-1, x.shape[-1]).contiguous()
        operand = (
            extension("ops").activation_pack(flat, w.dtype == torch.bfloat16, activation)
            if activation
            else flat.to(w.dtype)
        )
        ctx.save_for_backward(operand, w, flat if activation else flat.new_empty(0))
        ctx.shape = x.shape
        ctx.dtype = x.dtype
        ctx.activation = activation
        ctx.adjoint_mode = adjoint_mode
        return project_rows(operand, w, backend).view(*x.shape[:-1], w.shape[1])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        operand, w, preactivation = ctx.saved_tensors
        g = grad.reshape(-1, w.shape[1]).float().contiguous()
        dx, dw = projection_backward(
            operand,
            w,
            g,
            ctx.adjoint_mode,
            need_dx=ctx.needs_input_grad[0],
            need_dw=ctx.needs_input_grad[1],
        )
        if dx is not None:
            if ctx.activation:
                dx = activation_backward(preactivation, dx, ctx.activation)
            dx = dx.view(ctx.shape).to(ctx.dtype)
        return dx, dw.to(w.dtype) if dw is not None else None, None, None, None


def matmul(x, w, *, relu_squared=False, activation=0, adjoint_mode="store", backend="fixed"):
    if activation not in (0,2,3) or (relu_squared and activation):
        raise ValueError("choose at most one of ReLU², tanh or sigmoid")
    return _LinearProjection.apply(x, w, RELU_SQUARED if relu_squared else activation, adjoint_mode, backend)


def can_pack_ffn_key(rows, weight, backend):
    """Inference FFN epilogue; strict retains the validated training arithmetic."""
    if (rows == 1 and backend in ('fixed','lt','lt_equivalent')
            and tuple(weight.shape) in ((2560,10240),(4096,16384))
            and weight.T.is_contiguous() and weight.data_ptr()%16 == 0):
        return True
    if backend == "torch":
        return (rows >= 2048 and tuple(weight.shape) in ((2560,10240),(4096,16384))
                and not (weight.dtype == torch.bfloat16 and weight.shape[0] == 4096)
                and weight.T.is_contiguous() and weight.data_ptr() % 16 == 0
                and torch.cuda.get_device_capability(weight.device)[0] >= 8)
    return (rows >= 1024 and backend in ("lt","lt_equivalent")
            and tuple(weight.shape) == (2560,10240) and weight.T.is_contiguous()
            and torch.cuda.get_device_capability(weight.device) == (8,9))
