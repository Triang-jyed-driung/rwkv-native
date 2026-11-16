#include <torch/extension.h>
std::vector<at::Tensor> packed_attention_frontend(std::vector<at::Tensor>, at::Tensor, bool, bool);
std::vector<at::Tensor> packed_attention_prefill(at::Tensor, at::Tensor, std::vector<at::Tensor>);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &packed_attention_frontend);
    m.def("prefill", &packed_attention_prefill);
}
