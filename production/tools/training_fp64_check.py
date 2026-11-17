"""Independent real-weight FP64 training oracle and parameter-gradient RMSE.

The reference body uses CPU PyTorch operations and checkpointed recurrence;
its FP64 vocabulary head runs on CUDA to bound CPU time. It imports no native
operators. Validation uses the selected rwkv_native package, enabling frozen and
candidate checks with exactly the same input tokens and all-token CE objective.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def selected_weights(path, layers):
    values = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
    return {k:v for k,v in values.items()
            if not k.startswith('blocks.') or int(k.split('.')[1]) < layers}


def build_reference(args):
    torch.manual_seed(20261116)
    original = selected_weights(args.checkpoint,args.layers)
    width = original['emb.weight'].shape[1]
    vocab = original['emb.weight'].shape[0]
    heads = width//64
    ids = torch.randint(vocab,(args.batch,args.tokens),device='cuda').cpu()
    targets = torch.randint(vocab,(args.batch,args.tokens),device='cuda').cpu()
    weights = {k:v.detach().double().requires_grad_(True) for k,v in original.items()}
    del original

    def norm(x, prefix):
        return F.layer_norm(x,(width,),weights[prefix+'.weight'].flatten(),
                            weights[prefix+'.bias'].flatten(),1e-5)

    def recurrent_chunk(r, decay, k, v, a, b, matrix):
        outputs = []
        for token in range(r.shape[1]):
            projected = (matrix*a[:,token,...,:,None]).sum(-2)
            matrix = (matrix*decay[:,token,...,:,None]
                      + b[:,token,...,:,None]*projected[...,None,:]
                      + k[:,token,...,:,None]*v[:,token,...,None,:])
            outputs.append((matrix*r[:,token,...,:,None]).sum(-2))
        return torch.stack(outputs,1),matrix

    def block(x, first, layer):
        root = f'blocks.{layer}.'
        q = lambda name: weights[root+name]
        xx = norm(x,root+'ln1')
        attention_state = xx[:,-1]
        delta = torch.cat((torch.zeros_like(xx[:,:1]),xx[:,:-1]),1)-xx
        mixed = {name:xx+delta*q('att.x_'+name) for name in 'rwkvag'}
        r,k,v = [mixed[name]@q('att.'+projection+'.weight').T for name,projection in
                 (('r','receptance'),('k','key'),('v','value'))]
        rate = torch.sigmoid(q('att.a0')+mixed['a']@q('att.a1')@q('att.a2'))
        gate = torch.sigmoid(mixed['g']@q('att.g1'))@q('att.g2')
        if first is None: first = v
        else:
            v = v+(first-v)*torch.sigmoid(q('att.v0')+mixed['v']@q('att.v1')@q('att.v2'))
        normalized_key = (k*q('att.k_k')).reshape(args.batch,args.tokens,heads,64)
        normalized_key = normalized_key/normalized_key.square().sum(-1,keepdim=True).sqrt().clamp_min(1e-12)
        a = -normalized_key
        b = normalized_key*rate.reshape(args.batch,args.tokens,heads,64)
        decay = torch.exp(-0.6065306597126334*torch.sigmoid(
            q('att.w0')+torch.tanh(mixed['w']@q('att.w1'))@q('att.w2')))
        decay = decay.reshape(args.batch,args.tokens,heads,64)
        k = k*(1+(rate-1)*q('att.k_a'))
        r,k,v = [value.reshape(args.batch,args.tokens,heads,64) for value in (r,k,v)]
        matrix = torch.zeros(args.batch,heads,64,64,dtype=torch.float64)
        outputs = []
        for start in range(0,args.tokens,args.reference_wkv_chunk):
            stop = start+args.reference_wkv_chunk
            output,matrix = checkpoint(recurrent_chunk,
                *[value[:,start:stop] for value in (r,decay,k,v,a,b)],matrix,
                use_reentrant=False,preserve_rng_state=False)
            outputs.append(output)
        recurrent = torch.cat(outputs,1)
        centered = recurrent-recurrent.mean(-1,keepdim=True)
        normalized = centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+64e-5)
        dot = (r*k*q('att.r_k').reshape(1,heads,64)).sum(-1,keepdim=True)
        readout = (normalized.reshape(args.batch,args.tokens,width)*q('att.ln_x.weight')
                   + q('att.ln_x.bias')+(dot*v).reshape(args.batch,args.tokens,width))*gate
        x = x+readout@q('att.output.weight').T
        xx = norm(x,root+'ln2')
        channel_state = xx[:,-1]
        delta = torch.cat((torch.zeros_like(xx[:,:1]),xx[:,:-1]),1)-xx
        key = (xx+delta*q('ffn.x_k'))@q('ffn.key.weight').T
        x = x+key.clamp_min(0).square()@q('ffn.value.weight').T
        return x,first,attention_state,channel_state,matrix

    x = norm(F.embedding(ids,weights['emb.weight']),'blocks.0.ln0')
    first = None
    state = dict(attention=[],channel=[],wkv=[])
    for layer in range(args.layers):
        x,first,attention,channel,matrix = checkpoint(block,x,first,layer,
            use_reentrant=False,preserve_rng_state=False)
        for name,value in zip(state,(attention,channel,matrix)):
            state[name].append(value.detach())
        print(f'FP64 reference forward layer {layer+1}/{args.layers}',flush=True)
    hidden = norm(x,'ln_out')
    head = weights['head.weight'].to('cuda')

    def head_loss(value,target):
        logits = value.to('cuda')@head.T
        return F.cross_entropy(logits,target.to('cuda'),reduction='sum')

    flat = hidden.reshape(-1,width)
    target = targets.flatten()
    losses = [checkpoint(head_loss,flat[start:start+args.head_rows],target[start:start+args.head_rows],
                         use_reentrant=False,preserve_rng_state=False)
              for start in range(0,flat.shape[0],args.head_rows)]
    loss = torch.stack(losses).sum()/flat.shape[0]
    print('FP64 reference backward',flush=True)
    loss.backward()
    stat = args.checkpoint.stat()
    document = dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_bytes=stat.st_size,
                    checkpoint_mtime_ns=stat.st_mtime_ns,B=args.batch,T=args.tokens,layers=args.layers,
                    seed=20261116,head_rows=args.head_rows,loss_chunk_axis='flattened_tokens',
                    reference_method='independent_fp64_cpu_body_cuda_head',
                    ids=ids,targets=targets,hidden=hidden.detach(),state=state,loss=loss.item(),
                    gradients={k:v.grad for k,v in weights.items() if v.grad is not None})
    args.reference.parent.mkdir(parents=True,exist_ok=True)
    torch.save(document,args.reference)
    print(json.dumps(dict(reference=str(args.reference),parameters=len(document['gradients']),loss=loss.item())),flush=True)


def validate(args):
    from rwkv_native import FastRWKV7,State
    from benchmark_model import chunked_training_loss
    from benchmark_common import source_hash

    ref = torch.load(args.reference,mmap=True,weights_only=True)
    stat = args.checkpoint.stat()
    for key,expected in [('checkpoint',str(args.checkpoint.resolve())),('checkpoint_bytes',stat.st_size),
                         ('checkpoint_mtime_ns',stat.st_mtime_ns),('B',args.batch),('T',args.tokens),
                         ('layers',args.layers),('head_rows',args.head_rows)]:
        assert ref[key]==expected,(key,ref[key],expected)
    original = selected_weights(args.checkpoint,args.layers)
    model = FastRWKV7(original,precision=args.precision,consistency=args.consistency)
    ids,targets = ref['ids'].cuda(),ref['targets'].cuda()
    seed = model.initial_state(1)
    initial = State(*[[value.expand(args.batch,*value.shape[1:]) for value in group]
                      for group in (seed.attention,seed.channel,seed.wkv)])
    hidden,state = model.forward_features(ids,initial,checkpoint_blocks=True)
    state_errors = {}
    for group in ('attention','channel','wkv'):
        squared,count = 0.,0
        for actual,expected in zip(getattr(state,group),ref['state'][group]):
            squared += (actual.detach().cpu().double()-expected).square().sum().item()
            count += expected.numel()
        state_errors[group] = (squared/count)**.5
    logit_squared,logit_count = 0.,0
    with torch.no_grad():
        ref_head = original['head.weight'].cuda().double()
        flat = hidden.reshape(-1,hidden.shape[-1])
        ref_flat = ref['hidden'].reshape_as(flat)
        for start in range(0,flat.shape[0],args.head_rows):
            actual = model.project(flat[start:start+args.head_rows],model.weight('head.weight').T)
            expected = ref_flat[start:start+args.head_rows].cuda()@ref_head.T
            logit_squared += (actual.double()-expected).square().sum().item()
            logit_count += actual.numel()
        del ref_head,actual,expected,flat,ref_flat
    gc.collect();torch.cuda.empty_cache()
    loss = chunked_training_loss(model,hidden,targets,args.head_rows)
    loss.backward()
    errors = {}
    total_squared,total_count = 0.,0
    gradients = {name.replace('__','.'):parameter.grad for name,parameter in model.weights.items()
                 if parameter.grad is not None}
    assert gradients.keys()==ref['gradients'].keys(), (gradients.keys()^ref['gradients'].keys())
    for name,grad in gradients.items():
        actual = grad.detach().cpu().double()
        expected = ref['gradients'][name]
        assert torch.isfinite(actual).all(),name
        squared = (actual-expected).square().sum().item()
        errors[name] = (squared/actual.numel())**.5
        total_squared += squared
        total_count += actual.numel()
    result = dict(precision=args.precision,consistency=args.consistency,B=args.batch,T=args.tokens,
                  layers=args.layers,head_rows=args.head_rows,loss_chunk_axis='flattened_tokens',
                  runtime_sha256=source_hash(False),reference=str(args.reference),
                  loss=loss.item(),reference_loss=ref['loss'],logits_rmse=(logit_squared/logit_count)**.5,
                  state_rmse=state_errors,gradients_rmse=(total_squared/total_count)**.5,
                  gradient_parameter_rmse=errors,parameters_checked=len(errors),gradient_elements=total_count)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='gradient_parameter_rmse'}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--build-reference',action='store_true')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--tokens',type=int,default=17)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--head-rows',type=int,default=128)
    p.add_argument('--reference-wkv-chunk',type=int,default=16)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--precision',choices=('fp16','bf16'),default='bf16')
    p.add_argument('--consistency',choices=('strict','none'),default='strict')
    args = p.parse_args()
    if min(args.batch,args.tokens,args.layers,args.head_rows,args.reference_wkv_chunk,args.cpu_threads)<1:
        p.error('positive shapes, chunk sizes and thread count required')
    if not args.build_reference and args.output is None:p.error('--output required for validation')
    torch.set_num_threads(args.cpu_threads)
    if args.build_reference:build_reference(args)
    else:validate(args)


if __name__=='__main__':main()
