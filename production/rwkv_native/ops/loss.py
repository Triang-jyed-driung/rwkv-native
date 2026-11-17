"""Stream the vocabulary CE and accumulate gradients without full-matrix adds."""
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .._extensions import extension
from .gemm import project_rows
from .adjoints import projection_backward


class _HeadCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_rows, backend, adjoint_mode):
        operand = hidden.reshape(-1,hidden.shape[-1]).to(weight.dtype).contiguous()
        target = targets.reshape(-1).contiguous()
        count = (target != -100).sum()
        losses = []
        for start in range(0,operand.shape[0],chunk_rows):
            logits = project_rows(operand[start:start+chunk_rows],weight.T,backend)
            losses.append(F.cross_entropy(logits,target[start:start+chunk_rows],reduction='sum'))
        ctx.save_for_backward(operand,weight,target,count)
        ctx.shape,ctx.dtype = hidden.shape,hidden.dtype
        ctx.chunk_rows,ctx.backend,ctx.adjoint_mode = chunk_rows,backend,adjoint_mode
        return torch.stack(losses).sum()/count

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        operand,weight,targets,count = ctx.saved_tensors
        scaled = (operand.dtype == torch.bfloat16 and ctx.adjoint_mode == 'store'
                  and operand.shape[0] >= 4096)
        if scaled:
            operand_half, _, weight_half, range_factors = extension('scaled_half').pack(
                operand, operand.new_empty(0, dtype=torch.float32), weight)
        factor = (grad/count).contiguous()
        dx = torch.empty(operand.shape,device=operand.device,dtype=torch.float32) if ctx.needs_input_grad[0] else None
        dw = torch.empty_like(weight, dtype=torch.float32 if scaled else weight.dtype) if ctx.needs_input_grad[1] else None
        parts = 2
        add = False
        # Forward chunking retains the loss and strict projection policy.
        # BF16 head-gradient accumulation is bandwidth-heavy at 128 rows;
        # backward may group sixteen chunks without a bitwise-gradient contract.
        backward_rows = ((1024 if weight.shape[1] >= 4096 else 2048) if operand.dtype == torch.bfloat16
                         and operand.shape[0] >= 4096 and ctx.chunk_rows == 128
                         else ctx.chunk_rows)
        for start in reversed(range(0,operand.shape[0],backward_rows)):
            end = min(start+backward_rows,operand.shape[0])
            x = operand[start:end]
            logits = project_rows(x,weight.T,ctx.backend)
            g = extension('head_ce').gradient_(logits,targets[start:end],factor)
            if ctx.adjoint_mode=='fp32':
                gx,gw = projection_backward(x,weight.T,g,'fp32',need_dx=dx is not None,need_dw=dw is not None)
                if dx is not None:dx[start:end].copy_(gx)
                if dw is not None:
                    if add:dw.add_(gw.T.to(weight.dtype))
                    else:dw.copy_(gw.T)
            elif scaled:
                # Keep the accumulated head gradient in FP32 across chunks.
                # Forward logits still use the original BF16 projection.
                packed,coefficients,dx_scale = extension('head_gradient_pack').pack(g,range_factors,add)
                if dw is not None:
                    extension('scaled_half').dw(packed,operand_half[start:end],coefficients,dw)
                if dx is not None:
                    products = extension('adjoints').mm(packed,weight_half)
                    extension('head_gradient_pack').unpack(products,dx_scale,dx[start:end])
            else:
                if dw is not None:
                    packed,repeated,coefficients,scale = extension('gradient_pack').pack_accumulating(x,g,add)
                    # Physical [vocab,width] output matches the parameter's
                    # layout, avoiding a final full-weight transpose/copy.
                    extension('adjoints').accumulate(packed,repeated,coefficients,dw,add)
                else:
                    packed,_,_,scale = extension('gradient_pack').pack(x,g,False)
                if dx is not None:
                    products = extension('adjoints').mm(packed,weight)
                    extension('gradient_pack').combine_into(products,scale,parts,dx[start:end])
            add = True
        if scaled and dw is not None:
            dw = dw.to(weight.dtype)
        return dx.view(ctx.shape).to(ctx.dtype) if dx is not None else None,dw,None,None,None,None


def head_cross_entropy(hidden,weight,targets,*,chunk_rows=128,backend='fixed',adjoint_mode='store'):
    """Mean CE over nonignored targets; -100 is ignored, as in PyTorch CE.

    Uses the model's existing projection policy in forward. Supports first-order
    autograd with FP32 features and contiguous FP16/BF16 [vocab,width] weights.
    """
    if (hidden.ndim<2 or hidden.dtype!=torch.float32 or not hidden.is_cuda
            or weight.ndim!=2 or not weight.is_contiguous() or weight.device!=hidden.device
            or weight.dtype not in (torch.float16,torch.bfloat16) or hidden.shape[-1]!=weight.shape[1]
            or targets.device!=hidden.device or targets.dtype!=torch.int64
            or targets.numel()!=hidden.numel()//hidden.shape[-1] or targets.numel()==0
            or type(chunk_rows) is not int or chunk_rows<1 or adjoint_mode not in ('store','fp32')):
        raise ValueError('expected FP32 CUDA features, low-precision head, matching int64 targets and positive chunk rows')
    return _HeadCrossEntropy.apply(hidden,weight,targets,chunk_rows,backend,adjoint_mode)
