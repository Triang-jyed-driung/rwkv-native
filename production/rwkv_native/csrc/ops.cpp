#include <torch/extension.h>
at::Tensor fixed_layer_norm(at::Tensor, at::Tensor, at::Tensor);
at::Tensor fixed_relu_pack(at::Tensor, bool);
at::Tensor fixed_activation_pack(at::Tensor, bool, int64_t);
at::Tensor fixed_value_blend(at::Tensor, at::Tensor, at::Tensor, at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("norm", &fixed_layer_norm);
    m.def("relu_pack", &fixed_relu_pack);
    m.def("activation_pack", &fixed_activation_pack);
    m.def("value_blend", &fixed_value_blend);
}
