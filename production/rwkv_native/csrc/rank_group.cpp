#include <torch/extension.h>
std::vector<at::Tensor> pack_rank_inputs(std::vector<at::Tensor>);
at::Tensor grouped_rank_projection(std::vector<at::Tensor>, std::vector<at::Tensor>, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack", &pack_rank_inputs);
    m.def("run", &grouped_rank_projection);
}
