#include <torch/extension.h>
using at::Tensor;
std::vector<Tensor> wkv_prefetched_readout(std::vector<Tensor>,Tensor,Tensor,Tensor,std::vector<Tensor>,Tensor,bool);
std::vector<Tensor> wkv_register_prepared(std::vector<Tensor>,Tensor,Tensor,std::vector<Tensor>,Tensor,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("prefetched_readout",&wkv_prefetched_readout);
 m.def("register_prepared",&wkv_register_prepared);
}
