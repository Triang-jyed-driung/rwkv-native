import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k,n", [(2560, 2560), (4096, 16384), (16384, 4096), (2560, 65536)])
def test_decode_m16_matches_original_mma_and_larger_batches(dtype, k, n):
    from rwkv_native._extensions import extension
    from rwkv_native.ops.gemm import _fixed_schedule, project_rows
    torch.manual_seed(20261117)
    x = torch.randn(17, k, device="cuda", dtype=dtype) * .1
    weight = torch.randn(n, k, device="cuda", dtype=dtype) * .01
    splits, _ = _fixed_schedule(k, n, 1)
    original = extension("fixed").run(x, weight, splits, 64)
    for rows in (1, 16, 17):
        actual = project_rows(x[:rows], weight.T, "fixed")
        torch.testing.assert_close(actual, original[:rows], rtol=0, atol=0)

from rwkv_native.ops.gemm import prepare_lt, project_rows


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_preallocated_projection_handles_padded_columns_and_chunk_tail(dtype):
    x=torch.randn(3073,65,device="cuda",dtype=dtype)
    w=torch.randn(17,65,device="cuda",dtype=dtype).T
    actual=project_rows(x,w,"fixed")
    expected=torch.cat([project_rows(x[:3072],w,"fixed"),project_rows(x[3072:],w,"fixed")])
    assert actual.is_contiguous()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_packed_projection_operand_is_not_widened_then_narrowed(monkeypatch,dtype):
    import rwkv_native.ops.gemm as gemm
    x=torch.randn(7,64,device="cuda",dtype=dtype,requires_grad=True)
    w=torch.randn(64,32,device="cuda",dtype=dtype,requires_grad=True)
    original=gemm.project_rows
    def tracked(operand,weight,backend):
        assert operand.data_ptr()==x.data_ptr()
        assert operand.dtype==dtype
        return original(operand,weight,backend)
    monkeypatch.setattr(gemm,"project_rows",tracked)
    y=gemm.matmul(x,w,backend="torch")
    y.sum().backward()
    assert x.grad.dtype==dtype and w.grad.dtype==dtype


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("shape",[(2560,2560),(2560,10240),(2560,96)])
def test_sm89_fixed_lt_is_batch_invariant(dtype,shape):
    if torch.cuda.get_device_capability()!=(8,9):pytest.skip("SM89 plan")
    k,n=shape
    torch.manual_seed(117)
    w=(torch.randn(n,k,device="cuda")*.01).to(dtype).T
    supported,reason=prepare_lt(w)
    if not supported:pytest.skip(reason)
    x=(torch.randn(4096,k,device="cuda")*.1).to(dtype)
    for rows in (1,17,128,192,256,320,769,4096):
        y=project_rows(x[:rows],w,"lt_equivalent")
        for i in (0,rows-1):
            torch.testing.assert_close(y[i:i+1],project_rows(x[i:i+1],w,"lt_equivalent"),atol=0,rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k,n", [(2560,2560),(2560,10240),(10240,2560),(2560,96),
                                (4096,4096),(4096,16384),(16384,4096),(4096,128)])
def test_prefill_pipeline_preserves_single_row_arithmetic(dtype, k, n):
    torch.manual_seed(1117)
    weight = (torch.randn(n,k,device="cuda")*.01).to(dtype)
    x = (torch.randn(769,k,device="cuda")*.1).to(dtype)
    for rows in (127,128,192,193,256,769):
        y = project_rows(x[:rows],weight.T,"fixed")
        for index in (0,rows-1):
            single = project_rows(x[index:index+1],weight.T,"fixed")
            torch.testing.assert_close(y[index:index+1],single,atol=0,rtol=0)


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
