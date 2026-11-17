"""Single-token inference fusions preserving each projection's K arithmetic."""
import torch
from .._extensions import extension
from .gemm import _fixed_schedule


def ordered_splits(weight, backend):
    k, n = weight.shape
    if backend in ('lt', 'lt_equivalent'):
        if (k,n) not in ((2560,2560),(2560,96),(2560,10240)):
            raise ValueError('decode fusion requires a known fixed Lt schedule')
        return 8 if n == 96 else 1
    if backend != 'fixed':
        raise ValueError('decode fusion requires an ordered projection backend')
    return _fixed_schedule(k,n,1)[0]


def input_projections(operands, weights, backends):
    if torch.is_grad_enabled():
        raise RuntimeError('decode fusion requires gradients disabled')
    splits=[ordered_splits(w,b) for w,b in zip(weights,backends)]
    packed_weights=[w.T.contiguous() for w in weights]
    if operands[0].shape[1] == 2560:
        flat=extension('decode_inputs').run(operands,packed_weights,splits,16)
    else:
        flat=extension('decode_inputs_wide').run(operands,packed_weights,splits)
    return flat.split([w.shape[1] for w in weights])


def expand_rank(values,weights):
    if torch.is_grad_enabled():
        raise RuntimeError('decode fusion requires gradients disabled')
    out=extension('decode_rank').run([v.reshape(-1).contiguous() for v in values],
                                    [w.T.contiguous() for w in weights])
    return tuple(out[i].view(*v.shape[:-1],weights[i].shape[1]) for i,v in enumerate(values))


def ffn_up(operand,weight,backend):
    if torch.is_grad_enabled():
        raise RuntimeError('decode fusion requires gradients disabled')
    return extension('decode_up').run(operand,weight.T.contiguous(),
                                      ordered_splits(weight,backend),
                                      64 if operand.shape[1] == 2560 else 16,
                                      1,operand.new_empty(0))
