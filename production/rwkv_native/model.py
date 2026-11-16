"""Production RWKV7: explicit blocks, projection policy and recurrent state."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .ops.attention import PROJECTION_BRANCHES, attention_frontend
from .ops.channel import channel_frontend
from .ops.gemm import matmul
from .ops.key_gate import key_gate
from .ops.normalization import SIGMOID, TANH, activate, layer_norm, value_blend
from .ops.policy import EXPANSION_SHAPES, LOW_RANK_SUFFIXES, ProjectionPolicy
from .ops.rank_group import expand_low_rank
from .ops.readout import readout
from .ops.wkv import wkv7
from .state import State, token_shift


class _Embedding(torch.autograd.Function):
    """Sum repeated-token gradients in FP32 before parameter storage."""

    @staticmethod
    def forward(ctx, tokens, weight):
        ctx.save_for_backward(tokens)
        ctx.rows = weight.shape[0]
        ctx.dtype = weight.dtype
        return F.embedding(tokens, weight).float()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (tokens,) = ctx.saved_tensors
        dw = torch.ops.aten.embedding_dense_backward.default(
            grad.contiguous(), tokens, ctx.rows, -1, False
        )
        return None, dw.to(ctx.dtype)


@dataclass
class _ProjectionContext:
    """Per-call valid-row metadata, shared by every block and the output head."""

    model: "FastRWKV7"
    mask: torch.Tensor
    valid_indices: torch.Tensor | None
    can_fuse: bool

    def project(self, value, weight, *, relu_squared=False):
        if self.valid_indices is None:
            return self.model.project(value, weight, relu_squared=relu_squared)
        compact = value.reshape(-1, value.shape[-1]).index_select(0, self.valid_indices)
        projected = self.model.project(compact, weight, relu_squared=relu_squared)
        output = projected.new_zeros(value.shape[0] * value.shape[1], weight.shape[1])
        return output.index_copy(0, self.valid_indices, projected).view(
            *value.shape[:-1], weight.shape[1]
        )

    def project_many(self, values, weights):
        return tuple(self.project(value, weight) for value, weight in zip(values, weights))


class FastRWKV7(nn.Module):
    """RWKV7 with FP16/BF16 weights and FP32 activations, adjoints and KV state.

    ``backend='auto'`` selects a model-wide arithmetic policy for each weight
    shape. ``backend='fixed'`` forces the portable CUDA fallback. Fusion flags
    are diagnostic switches; defaults use the validated production paths.
    """

    def __init__(
        self,
        weights,
        *,
        device="cuda",
        dtype=torch.float16,
        backend="auto",
        adjoint_mode="store",
        fused_frontend=True,
        equivalent_tiles=True,
        fused_attention=True,
        fused_rank_group=True,
        fused_pre_group=True,
    ):
        super().__init__()
        if backend not in ("auto", "fixed"):
            raise ValueError("backend must be auto or fixed")
        if adjoint_mode not in ("store", "fp32"):
            raise ValueError("adjoint_mode must be store or fp32")
        if dtype not in (torch.float16, torch.bfloat16) or torch.device(device).type != "cuda":
            raise ValueError("FastRWKV7 requires CUDA FP16/BF16 weights")
        self.width = weights["emb.weight"].shape[1]
        if self.width % 64:
            raise ValueError("requires head size 64")
        self.heads = self.width // 64
        self.layers = (
            max(int(name.split(".")[1]) for name in weights if name.startswith("blocks.")) + 1
        )
        self.weights = nn.ParameterDict()
        for name, tensor in weights.items():
            value = tensor.to(device=device, dtype=dtype).contiguous()
            if name.endswith(LOW_RANK_SUFFIXES):
                value = value.T.contiguous().T
            self.weights[name.replace(".", "__")] = nn.Parameter(value)
        self.adjoint_mode = adjoint_mode
        self.fused_frontend = fused_frontend
        self.fused_attention = fused_attention
        self.fused_rank_group = fused_rank_group
        self.fused_pre_group = fused_pre_group
        self.equivalent_tiles = equivalent_tiles
        self._backend_request = backend
        self._configure_backend()

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        weights = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return cls(weights, **kwargs)

    def weight(self, name):
        return self.weights[name.replace(".", "__")]

    def _configure_backend(self):
        example = self.weight("emb.weight")
        if example.device.type != "cuda" or example.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FastRWKV7 parameters must remain CUDA FP16/BF16")
        if torch.cuda.get_device_capability(example.device)[0] < 8:
            raise ValueError("FastRWKV7 requires SM80 or newer")
        self._projection_policy = ProjectionPolicy.build(
            self.weights,
            backend=self._backend_request,
            equivalent_tiles=self.equivalent_tiles,
            grouped_inputs=self.fused_pre_group,
        )
        self.backend_report = self._projection_policy.report
        self.lt_shapes = self._projection_policy.lt_shapes

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_backend_request"):
            self._configure_backend()
        return result

    def initial_state(self, batch):
        options = {"device": self.weight("emb.weight").device, "dtype": torch.float32}
        return State(
            [torch.zeros(batch, self.width, **options) for _ in range(self.layers)],
            [torch.zeros(batch, self.width, **options) for _ in range(self.layers)],
            [torch.zeros(batch, self.heads, 64, 64, **options) for _ in range(self.layers)],
        )

    def projection_backend(self, weight):
        return self._projection_policy.backends.get(tuple(weight.shape), "fixed")

    def project(self, value, weight, *, relu_squared=False):
        return matmul(
            value,
            weight,
            relu_squared=relu_squared,
            adjoint_mode=self.adjoint_mode,
            backend=self.projection_backend(weight),
        )

    def _normalize(self, value, name):
        return layer_norm(value, self.weight(name + ".weight"), self.weight(name + ".bias"))

    def _attention_inputs(self, x, previous, context, root, rank_branches):
        prefix = root + "att."
        weights = [
            self.weight(prefix + name + ".weight").T for name in ("receptance", "key", "value")
        ]
        weights += [self.weight(prefix + name + "1") for name in rank_branches]
        coefficients = [self.weight(prefix + "x_" + name) for name in "rwkvag"]
        if self.fused_attention and context.can_fuse:
            return attention_frontend(
                x,
                previous,
                self.weight(root + "ln1.weight"),
                self.weight(root + "ln1.bias"),
                coefficients,
                weights,
                context.mask,
                self.adjoint_mode,
                tuple(self.projection_backend(weight) for weight in weights),
            )
        normalized = self._normalize(x, root + "ln1")
        past, shift = token_shift(normalized, previous, context.mask)
        delta = past - normalized
        mixed = [normalized + delta * coefficient for coefficient in coefficients]
        inputs = [mixed[branch] for branch in PROJECTION_BRANCHES[: len(weights)]]
        receptance, key, value, *reduced = context.project_many(inputs, weights)
        return receptance, key, value, reduced, shift

    def _attention(self, x, previous, matrix, first_value, context, layer):
        batch, tokens = x.shape[:2]
        root = f"blocks.{layer}."
        prefix = root + "att."
        rank_branches = "wag" if layer == 0 else "wagv"
        r, k, v, reduced, shift = self._attention_inputs(x, previous, context, root, rank_branches)
        weights = tuple(self.weight(prefix + name + "2") for name in rank_branches)
        grouped = (
            self.fused_rank_group
            and tuple(tuple(w.shape) for w in weights) == EXPANSION_SHAPES[: len(weights)]
            and all(w.dtype == torch.float16 for w in weights)
        )
        if grouped:
            expanded = expand_low_rank(reduced, weights, self.adjoint_mode)
        else:
            activated = (
                activate(reduced[0], TANH),
                reduced[1],
                activate(reduced[2], SIGMOID),
            ) + tuple(reduced[3:])
            expanded = context.project_many(activated, weights)
        decay = expanded[0] + self.weight(prefix + "w0")
        gate = expanded[2]
        if layer == 0:
            first_value = v
        else:
            v = value_blend(v, first_value, self.weight(prefix + "v0"), expanded[3])
        shape = (batch, tokens, self.heads, 64)
        k, negative_key, adapted_key = key_gate(
            k,
            self.weight(prefix + "k_k"),
            self.weight(prefix + "a0"),
            expanded[1],
            self.weight(prefix + "k_a"),
            self.heads,
        )
        y, next_matrix = wkv7(
            r.view(shape),
            decay.view(shape),
            k.view(shape),
            v.view(shape),
            negative_key.view(shape),
            adapted_key.view(shape),
            matrix,
            context.mask,
        )
        mixed = readout(
            y.view(batch, tokens, self.width),
            r,
            k,
            v,
            self.weight(prefix + "r_k"),
            self.weight(prefix + "ln_x.weight"),
            self.weight(prefix + "ln_x.bias"),
            gate,
            self.heads,
        )
        return mixed, first_value, shift, next_matrix

    def _channel(self, x, mixed, previous, context, layer):
        root = f"blocks.{layer}."
        output_weight = self.weight(root + "att.output.weight").T
        key_weight = self.weight(root + "ffn.key.weight").T
        backends = (self.projection_backend(output_weight), self.projection_backend(key_weight))
        if self.fused_frontend and context.can_fuse and "fixed" not in backends:
            key, shift, residual = channel_frontend(
                x,
                mixed,
                output_weight,
                previous,
                self.weight(root + "ln2.weight"),
                self.weight(root + "ln2.bias"),
                self.weight(root + "ffn.x_k"),
                key_weight,
                context.mask,
                self.adjoint_mode,
                backends,
            )
        else:
            residual = x + context.project(mixed, output_weight)
            normalized = self._normalize(residual, root + "ln2")
            past, shift = token_shift(normalized, previous, context.mask)
            key = normalized + (past - normalized) * self.weight(root + "ffn.x_k")
            key = context.project(key, key_weight)
        output = residual + context.project(
            key, self.weight(root + "ffn.value.weight").T, relu_squared=True
        )
        return output, shift

    def _forward_block(self, x, first_value, attention, channel, matrix, *, layer, context):
        mixed, first_value, attention, matrix = self._attention(
            x, attention, matrix, first_value, context, layer
        )
        x, channel = self._channel(x, mixed, channel, context, layer)
        return x, first_value, attention, channel, matrix

    def _features(self, tokens, state, attention_mask, checkpoint_blocks):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must be nonempty [B,T]")
        mask = (
            torch.ones_like(tokens, dtype=torch.bool) if attention_mask is None else attention_mask
        )
        if mask.shape != tokens.shape or mask.dtype != torch.bool or mask.device != tokens.device:
            raise ValueError("attention_mask must be bool [B,T] on the token device")
        indices = (
            mask.flatten().nonzero().flatten()
            if attention_mask is not None and tokens.shape[1] > 1
            else None
        )
        context = _ProjectionContext(
            self, mask, indices, tokens.shape[1] == 1 or attention_mask is None
        )
        state = self.initial_state(tokens.shape[0]) if state is None else state
        x = _Embedding.apply(tokens.masked_fill(~mask, 0), self.weight("emb.weight"))
        x = self._normalize(x, "blocks.0.ln0")
        next_state = State([], [], [])
        first_value = None
        for layer in range(self.layers):
            args = (x, first_value, state.attention[layer], state.channel[layer], state.wkv[layer])
            if checkpoint_blocks and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                result = checkpoint(
                    self._forward_block,
                    *args,
                    layer=layer,
                    context=context,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                result = self._forward_block(*args, layer=layer, context=context)
            x, first_value, attention, channel, matrix = result
            next_state.attention.append(attention)
            next_state.channel.append(channel)
            next_state.wkv.append(matrix)
        return self._normalize(x, "ln_out"), next_state, context

    def forward_features(self, tokens, state=None, attention_mask=None, *, checkpoint_blocks=False):
        """Return normalized hidden rows and state, optionally checkpointing blocks."""
        features, state, _ = self._features(tokens, state, attention_mask, checkpoint_blocks)
        return features, state

    def forward(self, tokens, state=None, attention_mask=None, *, last_only=False):
        features, state, context = self._features(tokens, state, attention_mask, False)
        if last_only:
            logits = self.project(features[:, -1:], self.weight("head.weight").T)
            mask = context.mask[:, -1:]
        else:
            logits = context.project(features, self.weight("head.weight").T)
            mask = context.mask
        return logits.masked_fill(~mask[..., None], 0), state
