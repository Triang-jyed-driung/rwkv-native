import pytest
import torch

from rwkv_native.ops.gemm import prepare_lt, project_rows


@pytest.mark.parametrize("shape", [(2048, 2048), (2048, 8192), (8192, 2048)])
def test_large_row_retarget_keeps_per_row_arithmetic(shape):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Recorded cuBLASLt plans apply to SM120")
    torch.manual_seed(914)
    k, n = shape
    w = (torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.01).T
    ok, reason = prepare_lt(w)
    if not ok:
        pytest.skip(reason)
    x = torch.randn(4097, k, device="cuda", dtype=torch.float16) * 0.1
    actual = project_rows(x, w, "lt_equivalent")
    reference = torch.cat([project_rows(part, w, "lt_equivalent") for part in x.split(768)], 0)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    for i in (0, 96, 767, 768, 3071, 4096):
        torch.testing.assert_close(
            actual[i : i + 1], project_rows(x[i : i + 1], w, "lt_equivalent"), atol=0, rtol=0
        )
