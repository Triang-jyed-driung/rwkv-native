"""Strict real-weight forward invariance at a measured inference scope.

Check every logit and terminal state against training, individual batch samples,
and sequential chunks. Saved training tensors are offloaded to CPU; backward is
outside this forward-only contract. This checker is not a performance benchmark.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from inference_fp64_check import GROUPS, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=('fp16', 'bf16'), required=True)
    parser.add_argument('--stream-chunk', type=int, default=127)
    args = parser.parse_args()
    if args.stream_chunk < 1:
        parser.error('--stream-chunk must be positive')
    from rwkv_native import FastRWKV7, State
    from benchmark_common import source_hash

    torch.set_num_threads(4)
    reference = torch.load(args.reference, mmap=True, map_location='cpu', weights_only=True)
    for key, value in identity(args.checkpoint).items():
        assert reference[key] == value, key
    batch, tokens, layers = (reference[key] for key in ('B', 'T', 'layers'))
    weights = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=True)
    selected = {key: value for key, value in weights.items()
                if not key.startswith('blocks.') or int(key.split('.')[1]) < layers}
    model = FastRWKV7(selected, precision=args.precision, consistency='strict')
    ids = reference['ids'].cuda()
    seed = model.initial_state(1)

    def initial(size):
        return State(*[[value.expand(size, *value.shape[1:]) for value in group]
                       for group in (seed.attention, seed.channel, seed.wkv)])

    def capture_state(state):
        return {name: [value.detach().cpu() for value in getattr(state, name)] for name in GROUPS}

    def equal_state(actual, expected, sample=None):
        for group in GROUPS:
            assert len(getattr(actual, group)) == len(expected[group]) == layers
            for layer, (value, target) in enumerate(zip(getattr(actual, group), expected[group])):
                if sample is not None:
                    target = target[sample:sample+1]
                assert torch.equal(value.detach().cpu(), target), (group, layer, sample)

    class StagedGroup:
        """Keep one input layer on CUDA; verify it before releasing storage."""
        def __init__(self, values):
            self.values, self.value, self.index = values, None, None

        def release(self):
            if self.value is not None:
                assert torch.equal(self.value.cpu(), self.values[self.index]), ('input state modified', self.index)
                self.value = None

        def __getitem__(self, index):
            self.release()
            self.index = index
            self.value = self.values[index].cuda()
            return self.value

    def continue_from_cpu(token_ids, previous):
        # This is an untimed correctness check. Full-batch kernels and state
        # values are unchanged; stage inputs layer by layer so an additional
        # complete terminal state does not exhaust a maximum-depth benchmark.
        staged = State(*[StagedGroup(previous[name]) for name in GROUPS])
        output = model(token_ids, staged)
        for name in GROUPS:
            getattr(staged, name).release()
        return output

    with torch.no_grad():
        logits, terminal = model(ids, initial(batch))
        expected = logits.cpu()
        expected_state = capture_state(terminal)
        del logits, terminal
    print('inference captured', flush=True)
    with torch.enable_grad(), torch.autograd.graph.save_on_cpu():
        hidden, terminal = model.forward_features(ids, initial(batch), checkpoint_blocks=True)
        logits = model.project(hidden, model.weight('head.weight').T)
        assert torch.equal(logits.detach().cpu(), expected), 'training logits differ'
        equal_state(terminal, expected_state)
        del hidden, terminal, logits
    gc.collect()
    torch.cuda.empty_cache()
    print('training forward bitwise', flush=True)
    with torch.no_grad():
        for sample in range(batch):
            logits, terminal = model(ids[sample:sample+1], initial(1))
            assert torch.equal(logits.cpu(), expected[sample:sample+1]), ('batch', sample)
            equal_state(terminal, expected_state, sample)
            del logits, terminal
        if batch == 1:
            # A one-sample versus itself comparison does not test batch changes.
            paired_ids = torch.cat((ids, (ids+1) % weights['emb.weight'].shape[0]), 0)
            logits, terminal = model(paired_ids, initial(2))
            assert torch.equal(logits[:1].cpu(), expected), 'batch 1 versus batch 2'
            for group in GROUPS:
                for actual, target in zip(getattr(terminal, group), expected_state[group]):
                    assert torch.equal(actual[:1].cpu(), target), ('batch 2 state', group)
            del paired_ids, logits, terminal
        print('individual batch samples bitwise', flush=True)
        # Make a real split for T > 1, including short rectangular cases.
        chunk = min(args.stream_chunk, max(1, tokens//2))
        state_cpu = None
        for start in range(0, tokens, chunk):
            stop = min(tokens, start+chunk)
            if state_cpu is None:
                state = initial(batch)
                previous = capture_state(state)
                logits, terminal = model(ids[:, start:stop], state)
                equal_state(state, previous)
                del state, previous
            else:
                logits, terminal = continue_from_cpu(ids[:, start:stop], state_cpu)
            assert torch.equal(logits.cpu(), expected[:, start:stop]), ('stream', start, stop)
            state_cpu = capture_state(terminal)
            del logits, terminal
        assert all(torch.equal(value, target) for name in GROUPS
                   for value, target in zip(state_cpu[name], expected_state[name]))
        if tokens == 1:
            # Exercise a real continuation and the decode/prefill boundary.
            continuation = (ids+1) % weights['emb.weight'].shape[0]
            combined_ids = torch.cat((ids, continuation), 1)
            complete_logits, complete_state = model(combined_ids, initial(batch))
            assert torch.equal(complete_logits[:, :1].cpu(), expected), 'decode versus prefill'
            continuation_logits = complete_logits[:, 1:].cpu()
            continuation_state = capture_state(complete_state)
            del complete_logits, complete_state
            next_logits, next_state = continue_from_cpu(continuation, state_cpu)
            assert torch.equal(next_logits.cpu(), continuation_logits), 'continuation'
            equal_state(next_state, continuation_state)
            del continuation, combined_ids, next_logits, next_state, continuation_logits, continuation_state
        assert all(torch.count_nonzero(value).item() == 0 for name in GROUPS for value in getattr(seed, name))
    result = dict(**identity(args.checkpoint), width=model.width, B=batch, T=tokens, layers=layers,
                  precision=args.precision, consistency='strict', head_scope='all_tokens',
                  initial_state='zero', mask='none', state_storage='batch_broadcast',
                  runtime_sha256=source_hash(False), reference=str(args.reference.resolve()),
                  train_vs_inference_bitwise=True, batch_vs_individual_bitwise=True,
                  batch_comparison_sizes=[batch, 1] if batch > 1 else [1, 2],
                  streaming_bitwise=True, stream_chunk=chunk, stream_comparison_tokens=max(2, tokens),
                  input_state_unchanged=True,
                  streaming_input_state_storage='layerwise_cpu_staging',
                  training_saved_tensors='cpu', checkpoint_blocks=True,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
