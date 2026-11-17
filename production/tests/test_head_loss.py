import pytest
import torch
from torch.nn import functional as F

from rwkv_native._extensions import extension
from rwkv_native.ops.loss import head_cross_entropy


@pytest.mark.parametrize('classes',[13,4096,65536])
@pytest.mark.parametrize('ignored',[False,True])
def test_ce_gradient_reuses_logits_and_matches_autograd(classes,ignored):
    torch.manual_seed(613)
    logits=(torch.randn(5,classes,device='cuda')*3).requires_grad_()
    targets=torch.randint(classes,(5,),device='cuda')
    if ignored:targets[1::2]=-100
    scale=torch.tensor(-.37,device='cuda')
    expected,=torch.autograd.grad(F.cross_entropy(logits,targets,reduction='sum')*scale,logits)
    buffer=logits.detach().clone()
    actual=extension('head_ce').gradient_(buffer,targets,scale)
    assert actual.data_ptr()==buffer.data_ptr()
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=3e-5)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('backend',['fixed','torch'])
@pytest.mark.parametrize('rows,classes,chunk',[(19,128,7),(19,65536,8),(5,128,32)])
def test_streamed_head_loss_matches_reference(dtype,backend,rows,classes,chunk):
    torch.manual_seed(614)
    x=torch.randn(rows,64,device='cuda',requires_grad=True)
    w=(torch.randn(classes,64,device='cuda',dtype=dtype)*.1).requires_grad_()
    targets=torch.randint(classes,(rows,),device='cuda');targets[2]=-100
    actual=head_cross_entropy(x,w,targets,chunk_rows=chunk,backend=backend)
    dx,dw=torch.autograd.grad(actual,(x,w))
    # FP64 oracle represents the same quantized projection operands and STE.
    xx=x.detach().to(dtype).cpu().double().requires_grad_()
    ww=w.detach().cpu().double().requires_grad_()
    reference=F.cross_entropy(xx@ww.T,targets.cpu())
    ex,ew=torch.autograd.grad(reference,(xx,ww))
    torch.testing.assert_close(actual.cpu().double(),reference,atol=3e-6,rtol=1e-6)
    torch.testing.assert_close(dx.cpu().double(),ex,atol=2e-7,rtol=3e-4)
    # Weight gradients are stored in the requested low precision; chunk
    # accumulation changes rounding, so use a normwise gradient bound.
    relative=(dw.cpu().double()-ew).square().mean().sqrt()/ew.square().mean().sqrt()
    assert relative < (.002 if dtype==torch.float16 else .015)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_head_loss_graph_replay_resets_accumulators(dtype):
    torch.manual_seed(615)
    x=torch.randn(2,17,64,device='cuda',requires_grad=True)
    w=torch.randn(128,64,device='cuda',dtype=dtype,requires_grad=True)
    targets=torch.randint(128,(2,17),device='cuda')
    def run():
        x.grad=w.grad=None
        loss=head_cross_entropy(x,w,targets,chunk_rows=7)
        (loss*.37).backward()
        return loss
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):expected=run()
    stream.synchronize()
    expected=expected.detach().clone();dx=x.grad.clone();dw=w.grad.clone()
    x.grad=w.grad=None
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):loss=run()
    for _ in range(3):
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(loss,expected,atol=0,rtol=0)
        torch.testing.assert_close(x.grad,dx,atol=0,rtol=0)
        torch.testing.assert_close(w.grad,dw,atol=0,rtol=0)


@pytest.mark.parametrize('input_grad,weight_grad',[(True,False),(False,True)])
def test_head_frozen_inputs_and_all_ignored_targets(input_grad,weight_grad):
    x=torch.randn(7,64,device='cuda',requires_grad=input_grad)
    w=torch.randn(128,64,device='cuda',dtype=torch.bfloat16,requires_grad=weight_grad)
    targets=torch.full((7,),-100,device='cuda',dtype=torch.int64)
    loss=head_cross_entropy(x,w,targets,chunk_rows=3)
    assert torch.isnan(loss)
    loss.backward()
    for value,needed in ((x,input_grad),(w,weight_grad)):
        assert (value.grad is not None)==needed
        if needed:assert torch.count_nonzero(value.grad)==0


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_fp32_adjoint_mode_remains_supported(dtype):
    torch.manual_seed(616)
    x=torch.randn(13,64,device='cuda',requires_grad=True)
    w=(torch.randn(128,64,device='cuda',dtype=dtype)*.1).requires_grad_()
    targets=torch.randint(128,(13,),device='cuda')
    loss=head_cross_entropy(x,w,targets,chunk_rows=5,adjoint_mode='fp32')
    dx,dw=torch.autograd.grad(loss,(x,w))
    xx=x.detach().to(dtype).cpu().double().requires_grad_()
    ww=w.detach().cpu().double().requires_grad_()
    expected=F.cross_entropy(xx@ww.T,targets.cpu())
    ex,ew=torch.autograd.grad(expected,(xx,ww))
    torch.testing.assert_close(dx.cpu().double(),ex,atol=2e-7,rtol=3e-4)
    error=(dw.cpu().double()-ew).square().mean().sqrt()/ew.square().mean().sqrt()
    assert error < (.002 if dtype==torch.float16 else .015)
