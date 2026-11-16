#include <torch/extension.h>
at::Tensor pre_group(std::vector<at::Tensor>, std::vector<at::Tensor>, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &pre_group); }
