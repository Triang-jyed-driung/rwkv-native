#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "ffn_key_generated.cuh"

at::Tensor decode_ffn_key(at::Tensor x, at::Tensor weight) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 2048 && x.is_contiguous() &&
                    x.scalar_type() == at::kHalf,
                "expected contiguous CUDA FP16 [M,2048]");
    TORCH_CHECK(weight.device() == x.device() && weight.scalar_type() == at::kHalf &&
                    weight.is_contiguous() && weight.sizes() == at::IntArrayRef({8192, 2048}),
                "expected contiguous FP16 [8192,2048] weight");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
                "expected 16-byte aligned operands");
    c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({x.size(0), 8192}, x.options().dtype(at::kFloat));
    if (x.size(0)) {
        TORCH_CHECK((x.size(0) + 15) / 16 <= 65535, "too many rows");
        ffn_key_single_kernel<<<dim3(128, (x.size(0) + 15) / 16), 128, 10240,
                                c10::cuda::getCurrentCUDAStream()>>>(
            (const half_t *)weight.data_ptr(), (const half_t *)x.data_ptr(), y.data_ptr<float>(),
            x.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}
