"""Independent CPU FP64 inference oracle at the exact measured model scope.

Reference generation never initializes CUDA. Stream checkpoint layers and retain
normalized hidden states instead of a potentially multi-GB vocabulary tensor.
Validation runs the actual full forward, then computes reference head errors in
CPU vocabulary tiles after releasing the model. No runtime operator is used by
the oracle. Both precisions and both head scopes share one reference.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F


GROUPS = ('attention', 'channel', 'wkv')


def identity(checkpoint):
    stat = checkpoint.stat()
    return dict(checkpoint=str(checkpoint.resolve()), checkpoint_bytes=stat.st_size,
                checkpoint_mtime_ns=stat.st_mtime_ns)


def norm(x, weights, prefix):
    return F.layer_norm(x, (x.shape[-1],), weights[prefix+'.weight'].flatten(),
                        weights[prefix+'.bias'].flatten(), 1e-5)


@torch.no_grad()
def reference_forward(weights, ids, layers):
    """Original checkpoint values, zero initial state, all input tokens valid."""
    batch, tokens = ids.shape
    width = weights['emb.weight'].shape[1]
    heads = width // 64
    x = F.embedding(ids, weights['emb.weight']).double()
    first = None
    state = {name: [] for name in GROUPS}
    for layer in range(layers):
        prefix = f'blocks.{layer}.'
        q = {key[len(prefix):]: value.double() for key, value in weights.items()
             if key.startswith(prefix)}
        if layer == 0:
            x = norm(x, q, 'ln0')
        xx = norm(x, q, 'ln1')
        state['attention'].append(xx[:, -1].clone())
        delta = torch.cat((torch.zeros_like(xx[:, :1]), xx[:, :-1]), 1) - xx
        mixed = {name: xx + delta*q['att.x_'+name] for name in 'rwkvag'}
        r, k, v = [mixed[name] @ q['att.'+projection+'.weight'].T for name, projection
                   in (('r', 'receptance'), ('k', 'key'), ('v', 'value'))]
        rate = torch.sigmoid(q['att.a0'] + mixed['a'] @ q['att.a1'] @ q['att.a2'])
        gate = torch.sigmoid(mixed['g'] @ q['att.g1']) @ q['att.g2']
        if first is None:
            first = v
        else:
            v = v + (first-v)*torch.sigmoid(q['att.v0'] + mixed['v'] @ q['att.v1'] @ q['att.v2'])
        key = (k*q['att.k_k']).reshape(batch, tokens, heads, 64)
        key = key / key.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        a, b = -key, key*rate.reshape(batch, tokens, heads, 64)
        decay = torch.exp(-0.6065306597126334*torch.sigmoid(
            q['att.w0'] + torch.tanh(mixed['w'] @ q['att.w1']) @ q['att.w2']))
        decay = decay.reshape(batch, tokens, heads, 64)
        k = k*(1+(rate-1)*q['att.k_a'])
        r, k, v = [value.reshape(batch, tokens, heads, 64) for value in (r, k, v)]
        matrix = torch.zeros(batch, heads, 64, 64, dtype=torch.float64)
        outputs = []
        for token in range(tokens):
            projected = (matrix*a[:, token, ..., :, None]).sum(-2)
            matrix = (matrix*decay[:, token, ..., :, None]
                      + b[:, token, ..., :, None]*projected[..., None, :]
                      + k[:, token, ..., :, None]*v[:, token, ..., None, :])
            outputs.append((matrix*r[:, token, ..., :, None]).sum(-2))
        y = torch.stack(outputs, 1)
        state['wkv'].append(matrix)
        centered = y-y.mean(-1, keepdim=True)
        normalized = centered*torch.rsqrt(centered.square().mean(-1, keepdim=True)+64e-5)
        dot = (r*k*q['att.r_k'].reshape(1, heads, 64)).sum(-1, keepdim=True)
        readout = (normalized.reshape(batch, tokens, width)*q['att.ln_x.weight']
                   + q['att.ln_x.bias'] + (dot*v).reshape(batch, tokens, width))*gate
        x = x + readout @ q['att.output.weight'].T
        xx = norm(x, q, 'ln2')
        state['channel'].append(xx[:, -1].clone())
        delta = torch.cat((torch.zeros_like(xx[:, :1]), xx[:, :-1]), 1)-xx
        key = (xx+delta*q['ffn.x_k']) @ q['ffn.key.weight'].T
        x = x + key.clamp_min(0).square() @ q['ffn.value.weight'].T
        del q
        print(json.dumps(dict(phase='reference', layer=layer+1, layers=layers)), flush=True)
    q = {key: value.double() for key, value in weights.items() if key.startswith('ln_out.')}
    return norm(x, q, 'ln_out'), state


def build_reference(args):
    if args.reference.exists():
        raise FileExistsError(f'Refusing to overwrite reference: {args.reference}')
    weights = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=True)
    vocab, width = weights['emb.weight'].shape
    if args.ids_from_reference:
        previous = torch.load(args.ids_from_reference, mmap=True, map_location='cpu', weights_only=True)
        assert Path(previous['checkpoint']).resolve() == args.checkpoint.resolve()
        ids = previous['ids'].clone()
        assert tuple(ids.shape) == (args.batch, args.tokens)
        del previous
    else:
        ids = torch.randint(vocab, (args.batch, args.tokens),
                            generator=torch.Generator(device='cpu').manual_seed(20261117))
    hidden, state = reference_forward(weights, ids, args.layers)
    document = dict(**identity(args.checkpoint), B=args.batch, T=args.tokens, width=width,
                    layers=args.layers, ids=ids, hidden=hidden, state=state,
                    initial_state='zeros', attention_mask='all_valid',
                    reference_method='independent_fp64_cpu_inference',
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.reference.with_suffix('.tmp')
    torch.save(document, temporary)
    temporary.replace(args.reference)
    assert not torch.cuda.is_initialized(), 'CPU reference unexpectedly initialized CUDA'
    print(json.dumps(dict(reference=str(args.reference), cuda_initialized=False)), flush=True)


def rmse(actual, expected):
    assert actual.shape == expected.shape
    difference = actual.double()-expected.double()
    return difference.square().sum().item(), difference.numel()


@torch.no_grad()
def validate(args):
    from rwkv_native import FastRWKV7, State
    from benchmark_common import source_hash

    reference = torch.load(args.reference, mmap=True, map_location='cpu', weights_only=True)
    scope = dict(**identity(args.checkpoint), B=args.batch, T=args.tokens, layers=args.layers)
    for key, value in scope.items():
        assert reference[key] == value, (key, reference[key], value)
    assert reference['reference_method'] == 'independent_fp64_cpu_inference'
    weights = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=True)
    selected = {key: value for key, value in weights.items()
                if not key.startswith('blocks.') or int(key.split('.')[1]) < args.layers}
    if args.stage_weights:
        from staged_inference_model import make_staged_model
        model = make_staged_model(selected, precision=args.precision, consistency=args.consistency)
    else:
        model = FastRWKV7(selected, precision=args.precision, consistency=args.consistency)
    ids = reference['ids'].cuda()
    seed = model.initial_state(1)
    initial = State(*[[value.expand(args.batch, *value.shape[1:]) for value in group]
                      for group in (seed.attention, seed.channel, seed.wkv)])
    logits, state = model(ids, initial, last_only=args.consistency == 'none')
    actual = logits.cpu()
    state_errors, state_elements = {}, {}
    for group in GROUPS:
        values = getattr(state, group)
        assert len(values) == len(reference['state'][group]) == args.layers
        errors = [rmse(value.cpu(), expected) for value, expected in zip(values, reference['state'][group])]
        count = sum(n for _, n in errors)
        state_errors[group] = (sum(s for s, _ in errors)/count)**.5
        state_elements[group] = count
    del model, ids, logits, state, selected, values, seed, initial
    gc.collect()
    torch.cuda.empty_cache()
    hidden = reference['hidden'][:, -1:] if args.consistency == 'none' else reference['hidden']
    squared, count = 0., 0
    for start in range(0, weights['head.weight'].shape[0], args.head_columns):
        stop = start+args.head_columns
        expected = hidden @ weights['head.weight'][start:stop].double().T
        error, elements = rmse(actual[..., start:stop], expected)
        squared += error
        count += elements
    result = dict(**scope, width=reference['width'], workload='inference',
                  precision=args.precision, consistency=args.consistency,
                  runtime_sha256=source_hash(False), reference=str(args.reference.resolve()),
                  reference_method=reference['reference_method'], initial_state='zero',
                  mask='none', state_storage='batch_broadcast',
                  head_scope='last_token' if args.consistency == 'none' else 'all_tokens',
                  logits_rmse=(squared/count)**.5, logits_elements=count,
                  state_rmse=state_errors, state_elements=state_elements,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result['parameter_storage'] = 'layerwise_cpu_staging' if args.stage_weights else 'resident'
    if args.stage_weights:
        import staged_inference_model
        result['parameter_staging_sha256'] = hashlib.sha256(Path(staged_inference_model.__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--build-reference', action='store_true')
    parser.add_argument('--stage-weights', action='store_true', help='untimed inference accuracy validation only')
    parser.add_argument('--ids-from-reference', type=Path)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--layers', type=int, default=32)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--head-columns', type=int, default=1024)
    parser.add_argument('--precision', choices=('fp16', 'bf16'), default='bf16')
    parser.add_argument('--consistency', choices=('strict', 'none'), default='strict')
    args = parser.parse_args()
    if min(args.batch, args.tokens, args.layers, args.cpu_threads, args.head_columns) < 1:
        parser.error('positive shapes and tile/thread counts required')
    if not args.build_reference and args.output is None:
        parser.error('--output required for validation')
    torch.set_num_threads(args.cpu_threads)
    if args.build_reference:
        build_reference(args)
    else:
        validate(args)


if __name__ == '__main__':
    main()
