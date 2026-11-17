#include <torch/extension.h>
at::Tensor fixed_projection(at::Tensor, at::Tensor, int64_t, int64_t);
at::Tensor fixed_projection_out(at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &fixed_projection);
    m.def("run_out", &fixed_projection_out);
}
