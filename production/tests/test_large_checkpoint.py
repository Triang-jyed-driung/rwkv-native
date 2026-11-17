"""Large-token sublayer replay preserves masked state and parameter adjoints."""
import pytest
import torch
from conftest import tiny_weights
from rwkv_native import FastRWKV7

@pytest.mark.parametrize('consistency',['strict','none'])
@pytest.mark.parametrize('masked',[False,True])
def test_large_sublayer_checkpoint(consistency,masked):
    torch.manual_seed(802)
    model=FastRWKV7(tiny_weights(),precision='bf16',consistency=consistency,backend='fixed')
    ids=torch.randint(128,(1,4097),device='cuda')
    mask=torch.ones_like(ids,dtype=torch.bool) if masked else None
    if masked:
        mask[:,:17]=False;mask[:,101::17]=False;mask[:,-19:]=False
    state=model.initial_state(1)
    for group in (state.attention,state.channel,state.wkv):
        for value in group:value.normal_(0,.01)
    previous=None
    for recompute in (False,True):
        model.zero_grad(set_to_none=True)
        output,final=model.forward_features(ids,state,mask,checkpoint_blocks=recompute)
        loss=output.square().mean()+sum(v.square().mean() for group in (final.attention,final.channel,final.wkv) for v in group)
        loss.backward()
        values=[output.detach(),*[v.detach() for group in (final.attention,final.channel,final.wkv) for v in group]]
        gradients={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
        if previous is not None:
            for a,b in zip(values,previous[0]):assert torch.equal(a,b)
            assert gradients.keys()==previous[1].keys()
            for name,g in gradients.items():
                assert torch.isfinite(g).all()
                assert torch.equal(g,previous[1][name]),name
        previous=([v.clone() for v in values],gradients)
