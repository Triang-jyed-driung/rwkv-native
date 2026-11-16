// Static export launch table.
#include "rank_group_3_m16.cuh"
#include "rank_group_3_m64.cuh"
#include "rank_group_4_m16.cuh"
#include "rank_group_4_m64.cuh"
static void launch_rank_group(const std::vector<at::Tensor>& xs,const std::vector<at::Tensor>& weights,at::Tensor output,int64_t tile,cudaStream_t stream){
 if(xs.size()==3 && tile==16)rank_group_3_m16<<<dim3(96,(xs[0].size(0)+tile-1)/tile),128,10240,stream>>>((const half_t*)weights[0].data_ptr(),(const half_t*)weights[1].data_ptr(),(const half_t*)weights[2].data_ptr(),(const half_t*)weights[2].data_ptr(),(const half_t*)xs[0].data_ptr(),(const half_t*)xs[1].data_ptr(),(const half_t*)xs[2].data_ptr(),(const half_t*)xs[2].data_ptr(),output.data_ptr<float>(),xs[0].size(0));
 if(xs.size()==3 && tile==64)rank_group_3_m64<<<dim3(96,(xs[0].size(0)+tile-1)/tile),128,16384,stream>>>((const half_t*)weights[0].data_ptr(),(const half_t*)weights[1].data_ptr(),(const half_t*)weights[2].data_ptr(),(const half_t*)weights[2].data_ptr(),(const half_t*)xs[0].data_ptr(),(const half_t*)xs[1].data_ptr(),(const half_t*)xs[2].data_ptr(),(const half_t*)xs[2].data_ptr(),output.data_ptr<float>(),xs[0].size(0));
 if(xs.size()==4 && tile==16)rank_group_4_m16<<<dim3(128,(xs[0].size(0)+tile-1)/tile),128,10240,stream>>>((const half_t*)weights[0].data_ptr(),(const half_t*)weights[1].data_ptr(),(const half_t*)weights[2].data_ptr(),(const half_t*)weights[3].data_ptr(),(const half_t*)xs[0].data_ptr(),(const half_t*)xs[1].data_ptr(),(const half_t*)xs[2].data_ptr(),(const half_t*)xs[3].data_ptr(),output.data_ptr<float>(),xs[0].size(0));
 if(xs.size()==4 && tile==64)rank_group_4_m64<<<dim3(128,(xs[0].size(0)+tile-1)/tile),128,16384,stream>>>((const half_t*)weights[0].data_ptr(),(const half_t*)weights[1].data_ptr(),(const half_t*)weights[2].data_ptr(),(const half_t*)weights[3].data_ptr(),(const half_t*)xs[0].data_ptr(),(const half_t*)xs[1].data_ptr(),(const half_t*)xs[2].data_ptr(),(const half_t*)xs[3].data_ptr(),output.data_ptr<float>(),xs[0].size(0));
}
