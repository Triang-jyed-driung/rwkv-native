"""Record real-model outputs, state and gradients before/after structural edits."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rwkv_native import FastRWKV7


def fingerprint(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
args = parser.parse_args()
model = FastRWKV7.from_checkpoint(
    args.checkpoint,
    dtype=torch.float16,
)
records = []
for batch, tokens, masked in (
    (1, 1, False),
    (1, 8, True),
    (4, 8, True),
    (96, 1, False),
    (32, 256, False),
):
    ids = (
        64
        + torch.arange(tokens, device="cuda")[None]
        + 17 * torch.arange(batch, device="cuda")[:, None]
    ) % 65536
    mask = torch.ones_like(ids, dtype=torch.bool) if masked else None
    if mask is not None:
        mask[:, 1::3] = False
    initial = model.initial_state(batch)
    training = batch == 1 and tokens == 8
    with torch.set_grad_enabled(training):
        logits, state = model(ids, initial, mask, last_only=not training)
        record = {"B": batch, "T": tokens, "masked": masked, "logits": fingerprint(logits)}
        record["state"] = {
            name: [fingerprint(x) for x in getattr(state, name)]
            for name in ("attention", "channel", "wkv")
        }
        if training:
            loss = torch.nn.functional.cross_entropy(logits[mask], (ids[mask] + 1) % 65536)
            loss.backward()
            record["loss"] = loss.item()
            record["gradients"] = {
                name: fingerprint(p.grad)
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            model.zero_grad(set_to_none=True)
    records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"B={batch} T={tokens} masked={masked} captured", flush=True)
    del logits, state, initial
