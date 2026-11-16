#include <torch/extension.h>
#include <vector>

using at::Tensor;
Tensor activation_forward(Tensor, int64_t);
Tensor activation_backward(Tensor, Tensor, int64_t);
std::vector<Tensor> layer_norm_backward(Tensor, Tensor, Tensor, double);
std::vector<Tensor> vres_backward(Tensor, Tensor, Tensor, Tensor, Tensor);
std::vector<Tensor> readout_backward(std::vector<Tensor>, Tensor, int64_t);
Tensor readout_forward(std::vector<Tensor>, int64_t);
std::vector<Tensor> mix_backward(Tensor, Tensor, std::vector<Tensor>, std::vector<Tensor>, Tensor);
std::vector<Tensor> key_gate_backward(std::vector<Tensor>, std::vector<Tensor>, int64_t);
std::vector<Tensor> key_gate_forward(std::vector<Tensor>, int64_t);
std::vector<Tensor> wkv_forward(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                int64_t, bool);
std::vector<Tensor> wkv_backward(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                 Tensor, Tensor, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("activation_forward", &activation_forward);
    m.def("activation_backward", &activation_backward);
    m.def("layer_norm_backward", &layer_norm_backward);
    m.def("vres_backward", &vres_backward);
    m.def("readout_backward", &readout_backward);
    m.def("readout_forward", &readout_forward);
    m.def("mix_backward", &mix_backward);
    m.def("key_gate_backward", &key_gate_backward);
    m.def("key_gate_forward", &key_gate_forward);
    m.def("forward", &wkv_forward);
    m.def("backward", &wkv_backward);
}
