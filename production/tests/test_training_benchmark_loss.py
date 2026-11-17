"""Head chunking must compute all-token CE independently of batch layout."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from benchmark_model import chunked_training_loss


class Head:
    def __init__(self, weight):
        self.head = weight
        self.rows = []

    def weight(self, name):
        assert name == 'head.weight'
        return self.head

    def project(self, hidden, weight):
        self.rows.append(hidden.shape[0])
        return hidden @ weight


@pytest.mark.parametrize('batch,length', [(1,24), (4,6), (6,4)])
@pytest.mark.parametrize('chunk', [5, 24, 32])
def test_chunked_head_loss_and_gradients(batch, length, chunk):
    torch.manual_seed(715)
    x = torch.randn(batch,length,7,dtype=torch.float64,requires_grad=True)
    weight = torch.randn(13,7,dtype=torch.float64,requires_grad=True)
    target = torch.randint(13,(batch,length))
    head = Head(weight)
    actual = chunked_training_loss(head,x,target,chunk)
    expected = torch.nn.functional.cross_entropy((x@weight.T).reshape(-1,13),target.flatten())
    dx,dw = torch.autograd.grad(actual,(x,weight),retain_graph=True)
    ex,ew = torch.autograd.grad(expected,(x,weight))
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(dx,ex,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(dw,ew,atol=1e-12,rtol=1e-12)
    assert max(head.rows) <= chunk
