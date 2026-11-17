from argparse import Namespace
from pathlib import Path
import sys

import pytest
import torch

from conftest import tiny_weights
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from training_fp64_check import build_reference


@pytest.mark.parametrize('name',['blocks.0.att.w2','blocks.1.att.a2','blocks.1.att.v0',
                                 'blocks.1.ffn.key.weight','head.weight','emb.weight'])
def test_fp64_reference_gradient_matches_finite_difference(tmp_path,name):
    weights={k:v.double() for k,v in tiny_weights().items()}
    args=Namespace(checkpoint=tmp_path/'weights.pth',reference=tmp_path/'ref.pt',
                   batch=1,tokens=3,layers=2,head_rows=2,reference_wkv_chunk=2)
    torch.save(weights,args.checkpoint)
    build_reference(args)
    reference=torch.load(args.reference,weights_only=True)
    grad=reference['gradients'][name]
    index=grad.abs().reshape(-1).argmax().item()
    expected=grad.reshape(-1)[index].item()
    epsilon=1e-5
    value=weights[name].reshape(-1)[index].item()
    losses=[]
    for delta in (-epsilon,epsilon):
        weights[name].reshape(-1)[index]=value+delta
        torch.save(weights,args.checkpoint)
        build_reference(args)
        losses.append(torch.load(args.reference,weights_only=True)['loss'])
    actual=(losses[1]-losses[0])/(2*epsilon)
    assert actual==pytest.approx(expected,rel=2e-4,abs=1e-7)
