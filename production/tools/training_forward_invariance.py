"""Strict training-scope forward checks; backward equality is not required."""
import argparse,gc,hashlib,json
from pathlib import Path
import torch
from benchmark_model import first_layers
from benchmark_common import source_hash
from inference_fp64_check import identity
from rwkv_native import FastRWKV7,State


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',choices=('fp16','bf16'),required=True)
    p.add_argument('--chunk-rows',type=int,default=128)
    a=p.parse_args()
    if a.chunk_rows<1:p.error('positive chunk rows required')
    torch.set_num_threads(4)
    ref=torch.load(a.reference,mmap=True,map_location='cpu',weights_only=True)
    for k,v in identity(a.checkpoint).items():assert ref[k]==v,k
    B,T,L=[ref[k] for k in ('B','T','layers')]
    weights=torch.load(a.checkpoint,mmap=True,map_location='cpu',weights_only=True)
    model=FastRWKV7(first_layers(weights,L),precision=a.precision,consistency='strict')
    C,V=model.width,weights['head.weight'].shape[0]
    ids=ref['ids'].cuda();seed=model.initial_state(1)
    def initial(batch):return State(*[[x.expand(batch,*x.shape[1:]) for x in group] for group in (seed.attention,seed.channel,seed.wkv)])
    def capture(state):return [[x.detach().cpu() for x in getattr(state,name)] for name in ('attention','channel','wkv')]
    def check_state(state,expected,sample=None):
        for group,old in zip(capture(state),expected):
            for value,target in zip(group,old):assert torch.equal(value,target if sample is None else target[sample:sample+1])
    with torch.enable_grad(),torch.autograd.graph.save_on_cpu():
        hidden,state=model.forward_features(ids,initial(B),checkpoint_blocks=True)
        logits=model.project(hidden,model.weight('head.weight').T)
        expected=logits.detach().cpu();expected_hidden=hidden.detach().cpu();expected_state=capture(state)
        del logits,hidden,state
    gc.collect();torch.cuda.empty_cache();print('training forward captured',flush=True)
    with torch.no_grad():
        hidden,state=model.forward_features(ids,initial(B))
        assert torch.equal(hidden.cpu(),expected_hidden),'inference hidden'
        logits=model.project(hidden,model.weight('head.weight').T)
        assert torch.equal(logits.cpu(),expected),'inference logits';check_state(state,expected_state)
        del logits,state
        flat=hidden.reshape(-1,C);target=expected.reshape(-1,V)
        for start in range(0,len(flat),a.chunk_rows):
            logits=model.project(flat[start:start+a.chunk_rows],model.weight('head.weight').T)
            assert torch.equal(logits.cpu(),target[start:start+a.chunk_rows]),('head rows',start)
        del logits,flat,target,hidden
        gc.collect();torch.cuda.empty_cache();print('inference and chunked head bitwise',flush=True)
        for row in range(B):
            logits,state=model(ids[row:row+1],initial(1))
            assert torch.equal(logits.cpu(),expected[row:row+1]),('batch',row)
            check_state(state,expected_state,row);del logits,state
        if B==1:
            # A one-sample call alone does not prove batch-size independence.
            paired=torch.cat((ids,(ids+1)%V))
            logits,state=model(paired,initial(2))
            assert torch.equal(logits[:1].cpu(),expected),'batch one versus two'
            for values,old in zip(capture(state),expected_state):
                for value,target in zip(values,old):assert torch.equal(value[:1],target)
            del paired,logits,state
    assert all(torch.count_nonzero(x).item()==0 for group in (seed.attention,seed.channel,seed.wkv) for x in group)
    result=dict(**identity(a.checkpoint),width=C,B=B,T=T,layers=L,precision=a.precision,consistency='strict',chunk_rows=a.chunk_rows,
                runtime_sha256=source_hash(False),reference=str(a.reference.resolve()),train_vs_inference_bitwise=True,
                batch_vs_individual_bitwise=True,batch_comparison_sizes=sorted({1,B,max(2,B)}),chunked_head_bitwise=True,
                input_state_unchanged=True,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
