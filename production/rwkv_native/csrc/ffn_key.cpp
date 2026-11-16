#include <torch/extension.h>
at::Tensor decode_ffn_key(at::Tensor, at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &decode_ffn_key); }
