"""Benchmark-only adapter for unmodified FlashRWKV2 public BF16 operators.

The block composition follows upstream tests/tmix/wkv7/test_block.py at
311d3c0ebf1ebcf5a62853c45c43b8ead3fa1c37 (MIT). This is a comparison backend,
not part of rwkv_native. Flash retains its native BF16 activation boundaries.
See FlashRWKV2.LICENSE in this directory for the upstream license.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rwkv_native import RuntimeConfig, State


class FlashRWKV7(nn.Module):
    def __init__(self, weights, *, precision="bf16", consistency="strict", device="cuda"):
        super().__init__()
        if precision != "bf16":
            raise ValueError("FlashRWKV2 is compared only in BF16 tracks")
        import flashrwkv2 as fr
        from flashrwkv2.post_norm.tilelang import norm
        from flashrwkv2.tmix.wkv7.statetune import (
            prepare_tmix_wkv7_statetune_metadata, statetune_tmix_wkv7_recurrent_fp32io16)

        self.fr, self.norm = fr, norm
        self.recurrent = statetune_tmix_wkv7_recurrent_fp32io16
        self.prepare_statetune = prepare_tmix_wkv7_statetune_metadata
        self.config = RuntimeConfig(precision, consistency)
        self.width = weights["emb.weight"].shape[1]
        self.heads = self.width // 64
        self.layers = max(int(k.split(".")[1]) for k in weights if k.startswith("blocks.")) + 1
        self.weights = nn.ParameterDict({
            k.replace(".", "__"): nn.Parameter(v.to(device=device, dtype=torch.bfloat16).contiguous())
            for k, v in weights.items()
        })
        self._metadata = {}

    def weight(self, name):
        return self.weights[name.replace(".", "__")]

    def initial_state(self, batch):
        options = dict(device=self.weight("emb.weight").device, dtype=torch.float32)
        return State(
            [torch.zeros(batch, self.width, **options) for _ in range(self.layers)],
            [torch.zeros(batch, self.width, **options) for _ in range(self.layers)],
            [torch.zeros(batch, self.heads, 64, 64, **options) for _ in range(self.layers)],
        )

    def project(self, hidden, weight):
        # The common benchmark supplies the logical [C,V] head weight.
        if torch.is_grad_enabled():
            return self.fr.pretrain_head_linear_bf16(hidden.contiguous(), weight.T)
        shape = hidden.shape[:-1]
        logits = self.fr.infer_head_linear_all_forward_varlen(
            hidden.reshape(-1, self.width).contiguous(), weight.T)
        return logits.view(*shape, -1)

    def _norm(self, x, prefix, res=None, return_res=False):
        return self.norm(x.contiguous(), self.weight(prefix + ".weight"),
                         self.weight(prefix + ".bias"), res=res, return_res=return_res)

    def _ticket(self, batch, length):
        key = batch, length, torch.cuda.current_stream(self.weight("emb.weight").device).cuda_stream
        if key not in self._metadata:
            slots = torch.arange(batch, device=self.weight("emb.weight").device, dtype=torch.int32)
            cu = torch.arange(batch + 1, device=slots.device, dtype=torch.int32) * length
            ticket = self.fr.prepare_tmix_wkv7_recurrent_metadata(
                cu, slots, total_tokens=batch * length, state_pool_size=batch)
            offsets = torch.arange(batch + 1, device=slots.device, dtype=torch.int32)
            starts, ends = cu[:-1].contiguous(), cu[1:].contiguous()
            prepared = self.prepare_statetune(offsets, starts, ends, total_tokens=batch * length)
            self._metadata[key] = cu, slots, ticket, (offsets, starts, ends, prepared)
        return self._metadata[key]

    def _train_layer(self, x, res, first, attention, channel, matrix, *, layer, metadata):
        fr = self.fr
        root = f"blocks.{layer}."
        p = lambda name: self.weight(root + "att." + name)
        normalized, total = self._norm(x, root + "ln1", res, True)
        mixed = fr.statetune_tmix_tokenshift_bf16(
            normalized, attention.to(torch.bfloat16).contiguous(),
            *(p("x_" + name).flatten() for name in "rwkvag"))
        prepared = fr.pretrain_tmix_wkv_prepare_bf16(
            *mixed[:6], *(p(name + ".weight") for name in ("receptance", "key", "value")),
            *(p(name).flatten() if name in ("w0", "a0", "k_k", "k_a") else p(name)
              for name in ("w0", "w1", "w2", "a0", "a1", "a2", "g1", "g2", "k_k", "k_a")),
            v_first=first, v0=p("v0").flatten(), v1=p("v1"), v2=p("v2"), head_size=64)
        offsets, starts, ends, prepared_metadata = metadata
        recurrent, final = self.recurrent(
            matrix.contiguous(), offsets, starts, ends,
            *(value.view(-1, self.heads, 64) for value in prepared[:6]),
            prepared_metadata=prepared_metadata)
        r, _, k, v, _, _, gate, first = prepared
        readout = fr.pretrain_tmix_readout_bf16(
            recurrent.view_as(x), r, k, v, p("r_k").view(self.heads, 64),
            p("ln_x.weight"), p("ln_x.bias"), gate, head_size=64,
            output_weight=p("output.weight"))
        normalized, total = self._norm(readout, root + "ln2", total, True)
        cmix, channel = fr.statetune_cmix_bf16(
            normalized, channel.to(torch.bfloat16).contiguous(),
            self.weight(root + "ffn.x_k").flatten(), self.weight(root + "ffn.key.weight"),
            self.weight(root + "ffn.value.weight"))
        return cmix, total, first, mixed[-1], channel, final

    def _infer_layer(self, x, res, first, attention, channel, matrix, *, layer, metadata):
        fr = self.fr
        cu, slots, ticket = metadata[:3]
        root = f"blocks.{layer}."
        p = lambda name: self.weight(root + "att." + name)
        # Upstream inference mutates pools. Copy inside the timed forward to
        # preserve the same functional input-state contract as native.
        attention = attention.to(torch.bfloat16).clone(memory_format=torch.contiguous_format)
        channel = channel.to(torch.bfloat16).clone(memory_format=torch.contiguous_format)
        matrix = matrix.clone(memory_format=torch.contiguous_format)
        state = fr.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(matrix)
        total, *mixed = fr.infer_tmix_postnorm_tokenshift_forward_varlen(
            x, res, self.weight(root + "ln1.weight"), self.weight(root + "ln1.bias"),
            *(p("x_" + name).flatten() for name in "rwkvag"),
            shift_state_pool=attention, cu_seqlens=cu, state_indices=slots,
            validated_metadata=ticket)
        prepared = fr.infer_tmix_wkv_prepare_forward_varlen(
            *mixed, *(p(name + ".weight") for name in ("receptance", "key", "value")),
            *(p(name).T for name in ("w1", "a1", "g1", "v1", "w2", "a2", "g2", "v2")),
            *(p(name).flatten() for name in ("v0", "k_k", "a0", "k_a")),
            v_first=first, head_size=64)
        recurrent = fr.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *(value.view(-1, self.heads, 64) for value in prepared[:6]),
            state=state, cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
            decay_bias=p("w0").view(self.heads, 64))
        r, _, k, v, _, _, gate, first = prepared
        output = fr.infer_tmix_readout_forward_varlen(
            recurrent.view(-1, self.width), r, k, v, p("r_k"),
            p("ln_x.weight"), p("ln_x.bias"), gate, p("output.weight"), head_size=64)
        total, cmix = fr.infer_cmix_forward_varlen(
            output, total, self.weight(root + "ln2.weight"), self.weight(root + "ln2.bias"),
            self.weight(root + "ffn.x_k").flatten(), self.weight(root + "ffn.key.weight"),
            self.weight(root + "ffn.value.weight").T,
            shift_state_pool=channel, cu_seqlens=cu, state_indices=slots,
            validated_metadata=ticket)
        return cmix, total, first, attention, channel, matrix

    def forward_features(self, tokens, state=None, *, checkpoint_blocks=False):
        batch, length = tokens.shape
        state = self.initial_state(batch) if state is None else state
        x = torch.nn.functional.embedding(tokens, self.weight("emb.weight"))
        x = self._norm(x, "blocks.0.ln0")
        training = torch.is_grad_enabled()
        if not training:
            x = x.view(-1, self.width)
        res, first = torch.zeros_like(x), None
        result = State([], [], [])
        metadata = self._ticket(batch, length)
        for layer in range(self.layers):
            args = (x, res, first, state.attention[layer], state.channel[layer], state.wkv[layer])
            if training:
                kwargs = dict(layer=layer, metadata=metadata[3])
                if checkpoint_blocks:
                    values = checkpoint(self._train_layer, *args, **kwargs,
                                        use_reentrant=False, preserve_rng_state=False)
                else:
                    values = self._train_layer(*args, **kwargs)
            else:
                values = self._infer_layer(*args, layer=layer, metadata=metadata)
            x, res, first, attention, channel, matrix = values
            result.attention.append(attention)
            result.channel.append(channel)
            result.wkv.append(matrix)
        hidden = self._norm(x, "ln_out", res)
        return hidden.view(batch, length, self.width), result

    def forward(self, tokens, state=None, *, last_only=False):
        hidden, state = self.forward_features(tokens, state)
        if last_only:
            hidden = hidden[:, -1:]
        return self.project(hidden, self.weight("head.weight").T), state
