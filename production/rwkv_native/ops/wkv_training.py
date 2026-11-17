"""FP32 training forward with shape-selected producer/consumer staging."""
import torch
from .._extensions import extension

def forward(vectors,state,mask,checkpoint_size,save):
    r=vectors[0]
    if (save and r.dtype==torch.float32 and checkpoint_size==16
            and r.shape[0]*r.shape[2]<128 and r.shape[1]>=1024):
        return extension("wkv_cooperative").run(list(vectors),state.contiguous(),mask,4,True)
    return extension("core").forward(*vectors,state,mask,checkpoint_size,save)
