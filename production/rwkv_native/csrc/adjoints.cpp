// FP32 accumulation and device-side alpha, with final FP16/BF16 gradient storage.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
using at::Tensor;
static void check(cublasStatus_t s) {
    TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cuBLAS status ", int(s));
}

Tensor scaled_dw(Tensor x, Tensor g, Tensor coefficients) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && g.dim() == 2 && x.is_contiguous() &&
                    g.is_contiguous(),
                "expected contiguous CUDA matrices");
    TORCH_CHECK(g.device() == x.device() && g.scalar_type() == x.scalar_type() &&
                    x.size(0) == g.size(0),
                "operand mismatch");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "expected FP16/BF16 operands");
    TORCH_CHECK(coefficients.device() == x.device() && coefficients.scalar_type() == at::kFloat &&
                    coefficients.is_contiguous() && coefficients.numel() == 2,
                "expected device FP32 [inverse scale, zero]");
    TORCH_CHECK(x.size(0) <= INT_MAX && x.size(1) > 0 && x.size(1) <= INT_MAX && g.size(1) > 0 &&
                    g.size(1) <= INT_MAX,
                "invalid GEMM dimensions");
    c10::cuda::CUDAGuard guard(x.device());
    if (!x.size(0))
        return at::zeros({x.size(1), g.size(1)}, x.options());
    auto out = at::empty({x.size(1), g.size(1)}, x.options());
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasPointerMode_t pointer_mode;
    cublasMath_t math_mode;
    check(cublasGetPointerMode(handle, &pointer_mode));
    check(cublasGetMathMode(handle, &math_mode));
    check(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    auto status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetMathMode(handle, math_mode);
        check(status);
    }
    const auto dtype = x.scalar_type() == at::kHalf ? CUDA_R_16F : CUDA_R_16BF;
    status =
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, g.size(1), x.size(1), x.size(0),
                     coefficients.data_ptr<float>(), g.data_ptr(), dtype, g.size(1), x.data_ptr(),
                     dtype, x.size(1), coefficients.data_ptr<float>() + 1, out.data_ptr(), dtype,
                     g.size(1), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    auto restore_pointer = cublasSetPointerMode(handle, pointer_mode);
    auto restore_math = cublasSetMathMode(handle, math_mode);
    check(status);
    check(restore_pointer);
    check(restore_math);
    return out;
}

// Explicit FP32 accumulation, independent of PyTorch's TF32/reduction flags.
Tensor strict_mm(Tensor x, Tensor w) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && w.dim() == 2 && w.device() == x.device() &&
                    x.scalar_type() == w.scalar_type(),
                "expected matching CUDA matrices");
    const auto type = x.scalar_type();
    TORCH_CHECK(type == at::kFloat || type == at::kHalf || type == at::kBFloat16,
                "unsupported GEMM dtype");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(1);
    TORCH_CHECK(K > 0 && N > 0 && w.size(0) == K && M <= INT_MAX && K <= INT_MAX && N <= INT_MAX,
                "invalid GEMM shape");
    const bool xr = x.stride(1) == 1 && x.stride(0) == K, xc = x.stride(0) == 1 && x.stride(1) == M;
    const bool wr = w.stride(1) == 1 && w.stride(0) == N, wc = w.stride(0) == 1 && w.stride(1) == K;
    TORCH_CHECK((xr || xc) && (wr || wc), "expected dense row/column-major matrices");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({M, N}, x.options().dtype(at::kFloat));
    if (!M)
        return out;
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasPointerMode_t pointer_mode;
    cublasMath_t math_mode;
    check(cublasGetPointerMode(handle, &pointer_mode));
    check(cublasGetMathMode(handle, &math_mode));
    check(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    auto status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasSetMathMode(handle, math_mode);
        check(status);
    }
    const auto dtype = type == at::kHalf       ? CUDA_R_16F
                       : type == at::kBFloat16 ? CUDA_R_16BF
                                               : CUDA_R_32F;
    const float alpha = 1.f, beta = 0.f;
    status = cublasGemmEx(handle, wr ? CUBLAS_OP_N : CUBLAS_OP_T, xr ? CUBLAS_OP_N : CUBLAS_OP_T, N,
                          M, K, &alpha, w.data_ptr(), dtype, wr ? N : K, x.data_ptr(), dtype,
                          xr ? K : M, &beta, out.data_ptr(), CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    auto restore_pointer = cublasSetPointerMode(handle, pointer_mode);
    auto restore_math = cublasSetMathMode(handle, math_mode);
    check(status);
    check(restore_pointer);
    check(restore_math);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &scaled_dw);
    m.def("mm", &strict_mm);
}
