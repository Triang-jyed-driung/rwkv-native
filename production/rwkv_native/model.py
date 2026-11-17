"""Production RWKV7: explicit blocks, projection policy and recurrent state."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .config import RuntimeConfig
from ._extensions import extension
from .ops.attention import PROJECTION_BRANCHES, attention_frontend
from .ops.channel import channel_frontend
from .ops.gemm import matmul, can_pack_ffn_key
from .ops.key_gate import key_gate
from .ops.normalization import SIGMOID, TANH, layer_norm, value_blend
from .ops.policy import EXPANSION_SHAPES, LOW_RANK_SUFFIXES, ProjectionPolicy
from .ops.rank_group import expand_low_rank
from .ops.decode import expand_rank as expand_decode_rank
from .ops.readout import readout
from .ops.sparse_ffn import sparse_ffn
from .ops.wkv import wkv7
from .ops.wkv_inference import inference_schedule, attention_inference
from .state import State, token_shift


class _Embedding(torch.autograd.Function):
    """Sum repeated-token gradients in FP32 before parameter storage."""

    @staticmethod
    def forward(ctx, tokens, weight):
        ctx.save_for_backward(tokens)
        ctx.rows = weight.shape[0]
        ctx.dtype = weight.dtype
        return extension("embedding").forward(tokens.contiguous(),weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (tokens,) = ctx.saved_tensors
        dw = extension("embedding").backward(grad.contiguous(),tokens.contiguous(),ctx.rows,
                                             ctx.dtype==torch.bfloat16)
        return None, dw


@dataclass
class _ProjectionContext:
    """Per-call valid-row metadata, shared by every block and the output head."""

    model: "FastRWKV7"
    mask: torch.Tensor
    valid_indices: torch.Tensor | None
    can_fuse: bool

    def project(self, value, weight, *, relu_squared=False, activation=0):
        if self.valid_indices is None:
            return self.model.project(value, weight, relu_squared=relu_squared, activation=activation)
        compact = value.reshape(-1, value.shape[-1]).index_select(0, self.valid_indices)
        projected = self.model.project(compact, weight, relu_squared=relu_squared, activation=activation)
        output = projected.new_zeros(value.shape[0] * value.shape[1], weight.shape[1])
        return output.index_copy(0, self.valid_indices, projected).view(
            *value.shape[:-1], weight.shape[1]
        )

    def project_many(self, values, weights):
        return tuple(self.project(value, weight) for value, weight in zip(values, weights))


class FastRWKV7(nn.Module):
    """RWKV7 with FP16/BF16 weights and FP32 activations, adjoints and KV state.

    PyTorch's gradient switch selects training or inference. ``precision`` selects
    FP16/BF16 weights and projection operands, with FP32 activations and state.
    ``consistency='strict'`` retains the shared, batch-invariant forward;
    ``'none'`` permits shape-dependent GEMMs. Neither constrains backward order.
    ``backend='fixed'`` remains a diagnostic override for projections.
    """

    def __init__(
        self,
        weights,
        *,
        device="cuda",
        dtype=None,
        precision=None,
        consistency="strict",
        backend="auto",
        adjoint_mode="store",
        fused_frontend=True,
        equivalent_tiles=True,
        fused_attention=True,
        fused_rank_group=True,
        fused_pre_group=True,
        sparse_ffn=True,
    ):
        super().__init__()
        self.config = RuntimeConfig.resolve(
            precision=precision, consistency=consistency, dtype=dtype
        )
        dtype = self.config.dtype
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
        self.sparse_ffn = sparse_ffn and self.config.consistency == "none" and backend != "fixed"
        for name, tensor in weights.items():
            value = tensor.to(device=device, dtype=dtype).contiguous()
            if name.endswith(LOW_RANK_SUFFIXES) or (self.sparse_ffn and name.endswith("ffn.value.weight")):
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
            consistency=self.config.consistency,
        )
        self.backend_report = self._projection_policy.report
        self.lt_shapes = self._projection_policy.lt_shapes

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_backend_request"):
            self._configure_backend()
            self.config = replace(
                self.config,
                precision="fp16" if self.weight("emb.weight").dtype == torch.float16 else "bf16",
            )
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

    def training_loss(self, hidden, targets, *, chunk_rows=128):
        """All-token mean CE, streaming the vocabulary head and its gradients.

        ``hidden`` comes from ``forward_features``; targets equal to -100 are
        ignored. The projection follows this model's forward precision policy.
        """
        from .ops.loss import head_cross_entropy
        weight = self.weight("head.weight")
        return head_cross_entropy(hidden,weight,targets,chunk_rows=chunk_rows,
                                  backend=self.projection_backend(weight.T),adjoint_mode=self.adjoint_mode)

    def project(self, value, weight, *, relu_squared=False, activation=0):
        return matmul(
            value,
            weight,
            relu_squared=relu_squared,
            activation=activation,
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
            and self.config.consistency == "strict"
            and tuple(tuple(w.shape) for w in weights) == EXPANSION_SHAPES[: len(weights)]
            and all(w.dtype == torch.float16 for w in weights)
        )
        if (self.fused_rank_group and self.config.consistency == 'strict'
                and context.can_fuse and batch == tokens == 1 and self.width in (2560,4096)
                and all(0 < w.shape[0] <= 512 and w.shape[0] % 32 == 0 for w in weights)
                and not torch.is_grad_enabled()):
            expanded=expand_decode_rank(reduced,weights)
        elif grouped:
            expanded = expand_low_rank(reduced, weights, self.adjoint_mode)
        else:
            expanded = tuple(context.project(value,weight,activation=kind)
                             for value,weight,kind in zip(reduced,weights,(TANH,0,SIGMOID,0)))
        schedule = (inference_schedule(batch, tokens, self.heads)
                    if self.fused_attention and context.can_fuse and not torch.is_grad_enabled()
                    else None)
        if batch == tokens == 1 and self.config.consistency != 'strict':
            schedule = None
        if schedule is not None:
            names = ("r_k", "ln_x.weight", "ln_x.bias", "w0", "k_k", "a0", "k_a",
                     "v0" if layer else "w0")
            mixed, next_matrix, first_value = attention_inference(
                schedule, r, k, v, expanded, first_value, matrix, context.mask,
                [self.weight(prefix + name) for name in names], self.heads)
            return mixed, first_value, shift, next_matrix
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
            approximate_backward=(self.config.consistency == "none" and tokens >= 256
                                  and context.valid_indices is None),
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
            packed=not torch.is_grad_enabled(),
        )
        return mixed, first_value, shift, next_matrix

    def _channel(self, x, mixed, previous, context, layer):
        root = f"blocks.{layer}."
        output_weight = self.weight(root + "att.output.weight").T
        key_weight = self.weight(root + "ffn.key.weight").T
        backends = (self.projection_backend(output_weight), self.projection_backend(key_weight))
        packed_relu = False
        if self.fused_frontend and context.can_fuse:
            packed_relu = (not torch.is_grad_enabled() and
                           can_pack_ffn_key(x.shape[0]*x.shape[1],key_weight,backends[1]))
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
                packed_relu=packed_relu,
            )
        else:
            residual = x + context.project(mixed, output_weight)
            normalized = self._normalize(residual, root + "ln2")
            past, shift = token_shift(normalized, previous, context.mask)
            key = normalized + (past - normalized) * self.weight(root + "ffn.x_k")
            key = context.project(key, key_weight)
        down_weight = self.weight(root + "ffn.value.weight").T
        rows = x.shape[0] * x.shape[1]
        # Preserve the strict split boundaries, while loading either dense
        # layout directly and adding the residual in the GEMM epilogue.
        ordered_down = (self.config.consistency == "strict" and
                        self.projection_backend(down_weight) == "fixed")
        accurate_none_down = (self.config.consistency == "none" and self.width == 2560
                              and down_weight.dtype == torch.bfloat16)
        if (not torch.is_grad_enabled() and context.can_fuse and rows >= 2048
                and self.width in (2560,4096) and (ordered_down or accurate_none_down)):
            operand = key.reshape(-1,key.shape[-1])
            if not packed_relu:
                operand = extension("ops").activation_pack(operand, down_weight.dtype == torch.bfloat16, 1)
            output = extension("ffn_down").run(
                operand, down_weight, residual.reshape(-1,self.width), True).view_as(residual)
            return output, shift
        # A row-major sparse weight must not silently select a less accurate
        # dense cuBLAS reduction. This direct-layout CUDA path avoids repacking
        # those weights and retains the original up projection for BF16.
        if (not torch.is_grad_enabled() and context.can_fuse and x.shape[1] > 1
                and self.config.consistency == "none" and self.width == 4096
                and down_weight.dtype == torch.bfloat16):
            operand = key.reshape(-1,key.shape[-1])
            if not packed_relu:
                operand = extension("ops").activation_pack(operand, True, 1)
            tile = 16 if rows <= 16 else 64128 if rows <= 320 else 132
            output = extension("flash_linear").run_residual(
                operand, down_weight.T, residual.reshape(-1,self.width), tile).view_as(residual)
            return output, shift
        if self.sparse_ffn and not torch.is_grad_enabled() and key.shape[0] * key.shape[1] <= 4:
            output = residual + sparse_ffn(key, down_weight)
        elif (packed_relu and key.dtype == torch.bfloat16 and self.width != 2560
              and self.projection_backend(down_weight) == "torch"):
            # The inference up projection already produced ReLU² operands.
            # Add the residual in the down GEMM epilogue, without another
            # FP32 projection output and elementwise-add launch.
            # FP16 retains its established reduction: canonical unsplit down
            # arithmetic exceeded the full-model FP64 logits RMSE limit.
            # The same gate rejects BF16 width 2560 on long real-model input.
            output = extension("flash_linear").run_residual(
                key.reshape(-1,key.shape[-1]), down_weight.T,
                residual.reshape(-1,residual.shape[-1]),
                132 if residual.shape[-1] == 4096 else 128,
            ).view_as(residual)
        else:
            output = residual + context.project(key, down_weight, relu_squared=not packed_relu)
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
        embedding_tokens = tokens if attention_mask is None else tokens.masked_fill(~mask,0)
        x = _Embedding.apply(embedding_tokens, self.weight("emb.weight"))
        x = self._normalize(x, "blocks.0.ln0")
        next_state = State([], [], [])
        first_value = None
        for layer in range(self.layers):
            args = (x, first_value, state.attention[layer], state.channel[layer], state.wkv[layer])
            if checkpoint_blocks and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                if self.config.precision == "bf16" and x.shape[0]*x.shape[1] >= 4096:
                    # Replay attention and FFN separately so their saved tensors
                    # are not simultaneously resident during the large backward.
                    mixed, first_value, attention, matrix = checkpoint(
                        self._attention, x, state.attention[layer], state.wkv[layer],
                        first_value, context, layer, use_reentrant=False,
                        preserve_rng_state=False)
                    x, channel = checkpoint(
                        self._channel, x, mixed, state.channel[layer], context, layer,
                        use_reentrant=False, preserve_rng_state=False)
                    result = x, first_value, attention, channel, matrix
                else:
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
        return (logits if attention_mask is None else logits.masked_fill(~mask[..., None],0)), state
