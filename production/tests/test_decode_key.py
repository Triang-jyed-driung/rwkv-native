import pytest
import torch

from rwkv_native._extensions import extension
from rwkv_native.ops.gemm import _lt_plan, prepare_lt, project_rows


@pytest.mark.parametrize("rows", [1, 17, 96])
@pytest.mark.parametrize("scale", [0.01, 1.0, 16.0])
def test_decode_key_matches_existing_prefill_arithmetic(rows, scale):
    torch.manual_seed(932)
    x = torch.randn(rows, 2048, device="cuda", dtype=torch.float16) * scale
    weight = torch.randn(8192, 2048, device="cuda", dtype=torch.float16) * 0.01
    supported, reason = prepare_lt(weight.T)
    if not supported:
        pytest.skip(reason)
    expected = _lt_plan(x, weight.T, True).run(x, weight.T)
    actual = extension("ffn_key").run(x, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        actual.double(), x.double() @ weight.double().T, atol=5e-6 * scale, rtol=3e-5
    )
    torch.testing.assert_close(
        extension("ffn_key").run(x[:1], weight), expected[:1], atol=0, rtol=0
    )
    torch.testing.assert_close(
        project_rows(x[:1], weight.T, "lt_equivalent"), expected[:1], atol=0, rtol=0
    )
