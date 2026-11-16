#include <torch/extension.h>
std::vector<at::Tensor> fixed_frontend(std::vector<at::Tensor>, at::Tensor, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &fixed_frontend); }
