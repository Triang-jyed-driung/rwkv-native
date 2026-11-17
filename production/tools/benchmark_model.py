"""Real checkpoint latency and training forward/backward, excluding optimizer.step."""

import argparse
import gc
import hashlib
import math
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from rwkv_native import FastRWKV7, State
from benchmark_common import measure, source_hash


CASES = [("decode", 1, 1), ("decode", 192, 1),
         ("prefill", 1, 256), ("prefill", 1, 4096),
         ("rectangular_prefill", 20, 16), ("rectangular_prefill", 60, 64),
         ("train", 16, 1024), ("train", 1, 16384)]


def first_layers(weights, layers):
    """Keep embedding/output head and the original first N blocks, without renumbering."""
    return {k: v for k, v in weights.items()
            if not k.startswith("blocks.") or int(k.split(".")[1]) < layers}


def chunked_training_loss(model, hidden, targets, chunk_rows, *, fused=True):
    """All-token CE with the same head row limit for every batch/length shape."""
    if fused and hasattr(model, 'training_loss'):
        return model.training_loss(hidden,targets,chunk_rows=chunk_rows)
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    if chunk_rows < 1 or targets.numel() != hidden.shape[0]:
        raise ValueError("positive chunk rows and matching token targets required")

    def loss_chunk(value, target):
        logits = model.project(value, model.weight("head.weight").T)
        return torch.nn.functional.cross_entropy(logits, target, reduction="sum")

    losses = [checkpoint(loss_chunk, hidden[start:start+chunk_rows],
                         targets[start:start+chunk_rows], use_reentrant=False,
                         preserve_rng_state=False)
              for start in range(0, hidden.shape[0], chunk_rows)]
    return torch.stack(losses).sum() / hidden.shape[0]


def check_gradients(model):
    """Compare all used parameter gradients with/without block recomputation, untimed.

    This checks real-weight checkpoint integration, not an independent derivative
    oracle. WKV's separate float64-reference tests cover the recurrence adjoint.
    """
    torch.manual_seed(20261116)
    vocab = model.weight("emb.weight").shape[0]
    ids = torch.randint(vocab, (1, 17), device="cuda")
    target = torch.randint(vocab, (1,), device="cuda")
    reference = {}
    max_error, max_rmse, max_relative_rmse, checked = 0., 0., 0., 0
    with torch.enable_grad():
        for recompute in (False, True):
            model.zero_grad(set_to_none=True)
            hidden, _ = model.forward_features(ids, checkpoint_blocks=recompute)
            logits = model.project(hidden[:, -1], model.weight("head.weight").T)
            loss = torch.nn.functional.cross_entropy(logits, target) + hidden.square().mean() * .001
            loss.backward()
            seen = set()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().float().cpu()
                if not torch.isfinite(grad).all():
                    raise AssertionError(f"nonfinite gradient: {name}")
                seen.add(name)
                if not recompute:
                    reference[name] = grad
                else:
                    tol = .003 if model.config.precision == "fp16" else .02
                    difference = grad-reference[name]
                    rmse = math.sqrt(difference.square().mean().item())
                    magnitude = math.sqrt(reference[name].square().mean().item())
                    # Backward need not be bitwise or elementwise invariant.
                    # Check recomputation with a normwise tolerance; independent
                    # FP64 checks provide the optimization accuracy criterion.
                    if rmse > 2e-6 + tol*magnitude:
                        raise AssertionError(f"{name}: checkpoint gradient RMSE {rmse} exceeds {2e-6+tol*magnitude}")
                    max_error = max(max_error, difference.abs().max().item())
                    max_rmse = max(max_rmse,rmse)
                    max_relative_rmse = max(max_relative_rmse,rmse/max(magnitude,1e-30))
                    checked += 1
            if recompute and seen != reference.keys():
                raise AssertionError("checkpointing changed the set of parameter gradients")
    model.zero_grad(set_to_none=True)
    return {"status": "passed", "method": "checkpointed_vs_uncheckpointed_rmse",
            "B": 1, "T": 17, "parameters_checked": checked, "max_abs_error": max_error,
            "max_parameter_rmse": max_rmse, "max_relative_parameter_rmse": max_relative_rmse}


def run_case(model, workload, b, t, args):
    training = workload == "train"
    vocab = model.weight("emb.weight").shape[0]
    torch.manual_seed(20261116)
    ids = torch.randint(vocab, (b, t), device="cuda")
    targets = torch.randint(vocab, (b, t), device="cuda") if training else None
    # Identical zero state for each sample, with batch-broadcast storage. The
    # functional model materializes only the current layer when needed. Avoid
    # keeping a second full B192 state pool resident merely to represent zeros.
    seed_state = model.initial_state(1)
    state = State(*[[x.expand(b, *x.shape[1:]) for x in group]
                    for group in (seed_state.attention, seed_state.channel, seed_state.wkv)])

    def forward():
        if not training:
            return model(ids, state, last_only=model.config.consistency == "none")
        model.zero_grad(set_to_none=True)
        hidden, _ = model.forward_features(ids, state, checkpoint_blocks=True)
        # Recompute head/CE per chunk in backward, bounding retained vocabulary logits.
        return chunked_training_loss(model, hidden, targets, args.loss_chunk,fused=args.head_loss!='generic')

    def backward(loss):
        loss.backward()

    name = f"model-l{model.layers}-{model.config.precision}-{model.config.consistency}-{workload}-b{b}-t{t}"
    tuning = None
    with torch.set_grad_enabled(training):
        if args.implementation == "flash":
            # Preparation executes the actual forward/backward outside timing.
            # Warmup and JIT compilation alone do not constitute tuning.
            args.measure_phase = "flash_tuning"
            def prepare():
                result = forward()
                if training:
                    backward(result)
            tuning = model.fr.prepare_tuning([prepare], mode=args.flash_tuning)
            tuning["mode"] = args.flash_tuning
            for entry in tuning["records"]:
                path = entry.get("path")
                if path and Path(path).is_file():
                    entry["profile_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        metrics = measure(forward, backward if training else None, args, name,
                          prepare_capture=lambda: model.zero_grad(set_to_none=True))
    if training and getattr(args, "verify_gradients", False):
        args.measure_phase = "gradient_finiteness"
        # Untimed sanity check; full correctness is covered by the operator/model tests.
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), "nonfinite parameter gradient"
    record = {"workload": workload, "B": b, "T": t, "precision": model.config.precision,
              "consistency": model.config.consistency,
              "last_only": not training and model.config.consistency == "none",
              "head_scope": "last_token" if not training and model.config.consistency == "none" else "all_tokens",
              "checkpoint_blocks": training, "loss_chunk": args.loss_chunk if training else None,
              "loss_chunk_axis": "flattened_tokens" if training else None,
              "head_loss_implementation": ('streamed_cuda' if args.head_loss!='generic' and hasattr(model,'training_loss')
                                           else 'checkpointed_autograd') if training else None,
              "mask": "none", "initial_state": "zero", "state_storage": "batch_broadcast",
              **metrics}
    if tuning is not None:
        record["flash_tuning"] = tuning
    record["tokens_per_second"] = b * t * 1000 / record["total_ms"]["p10"]
    model.zero_grad(set_to_none=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=("native", "flash"), default="native")
    parser.add_argument("--flash-tuning", choices=("inspect", "require", "tune"), default="inspect",
                        help="record actual Flash dispatches before timing; tune searches missing configurable keys")
    parser.add_argument("--layers", type=int,
                        help="explicit measured prefix, retaining embedding/ln_out/head; disables automatic depth fallback")
    parser.add_argument("--cases", nargs="+", choices=[f"{w}:{b}:{t}" for w, b, t in CASES],
                        help="run a subset; fragments do not individually satisfy the full gate")
    parser.add_argument("--probe-shape", help="explicit reduced training shape B:T; recorded as a probe")
    parser.add_argument("--workload", choices=("inference", "train", "all"), default="all")
    parser.add_argument("--precisions", nargs="+", choices=("fp16", "bf16"))
    parser.add_argument("--consistencies", nargs="+", choices=("strict", "none"), default=["strict", "none"])
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--verify-gradients", action="store_true",
                        help="opt in to untimed short checkpoint and gradient-finiteness checks")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--loss-chunk", type=int, default=128,
                        help="maximum flattened token rows per head/CE chunk, independent of B")
    parser.add_argument("--head-loss", choices=("auto","generic"), default="auto",
                        help="use the model's streamed CE when available, or the generic checkpointed head for ablation")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    probe_cases = None
    if args.probe_shape:
        if args.cases or args.workload == "inference" or args.layers is None:
            parser.error("--probe-shape requires explicit --layers and cannot combine with --cases/inference")
        try:
            probe_b,probe_t = map(int,args.probe_shape.split(':'))
            if min(probe_b,probe_t)<1:raise ValueError
        except ValueError:
            parser.error("--probe-shape must be positive B:T")
        probe_cases = [('train',probe_b,probe_t)]
    if args.precisions is None:
        args.precisions = ["bf16"] if args.implementation == "flash" else ["fp16", "bf16"]
    if args.implementation == "flash" and args.precisions != ["bf16"]:
        parser.error("FlashRWKV2 comparison supports only --precisions bf16")
    model_class = FastRWKV7
    if args.implementation == "flash":
        from flash_model import FlashRWKV7
        model_class = FlashRWKV7
    if args.warmup < 1 or args.repeats < 1 or args.loss_chunk < 1 or (args.layers is not None and args.layers < 1):
        parser.error("require warmup >= 1, repeats >= 1, loss-chunk >= 1")
    stat = args.checkpoint.stat()
    document = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_bytes": stat.st_size,
                "checkpoint_mtime_ns": stat.st_mtime_ns, "source_sha256": source_hash(),
                "runtime_sha256": source_hash(False),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "execution": args.execution,
                "warmup": args.warmup, "repeats": args.repeats,
                "suite": "training_shape_probe" if probe_cases else "fixed_baseline",
                "optimizer_step": False, "seed": 20261116, "loss_chunk_axis": "flattened_tokens",
                "implementation": args.implementation, "cases": []}
    if args.implementation == "flash":
        import flashrwkv2
        root = Path(flashrwkv2.__file__).resolve().parent
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        document.update(flash_package=str(root), flash_python_sha256=digest.hexdigest(),
                        flash_adapter_sha256=hashlib.sha256(
                            Path(__file__).with_name("flash_model.py").read_bytes()).hexdigest(),
                        activation_precision="bf16", state_precision="fp32",
                        state_copy="functional copy inside timed forward")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    full_layers = max(int(k.split(".")[1]) for k in weights if k.startswith("blocks.")) + 1
    if args.layers is not None and args.layers > full_layers:
        parser.error("--layers cannot exceed the checkpoint's layer count")
    selected_weights = first_layers(weights, args.layers) if args.layers else weights
    selected_cases = [(w, b, t) for w, b, t in CASES
                      if (not args.cases or f"{w}:{b}:{t}" in args.cases)
                      and (args.workload != "train" or w == "train")
                      and (args.workload != "inference" or w != "train")]
    if probe_cases is not None:selected_cases = probe_cases
    document.update(width=weights["emb.weight"].shape[1], layers=full_layers,
                    vocab=weights["emb.weight"].shape[0])
    for precision in args.precisions:
        for consistency in args.consistencies:
            load_oom = False
            try:
                model = model_class(selected_weights, precision=precision, consistency=consistency)
            except torch.OutOfMemoryError:
                load_oom = True
            if load_oom:
                for workload, b, t in selected_cases:
                    document["cases"].append(dict(
                        workload=workload, B=b, T=t, precision=precision, consistency=consistency,
                        status="oom", phase="model_load", full_layers=full_layers,
                        measured_layers=args.layers or full_layers))
                args.output.write_text(json.dumps(document, indent=2) + "\n")
                print(json.dumps(dict(status="oom", phase="model_load", precision=precision,
                                      consistency=consistency, layers=args.layers or full_layers)), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            gradient_check = None
            fallback_reason = "explicit_common_depth" if model.layers != full_layers else None
            for workload, b, t in selected_cases:
                # On the largest model, weights + gradients alone can exceed VRAM.
                # Avoid a futile full backward, and record this exact lower bound.
                free_bytes, _ = torch.cuda.mem_get_info()
                gradient_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                if (args.layers is None and workload == "train" and model.width >= 4096 and model.layers > 2
                        and gradient_bytes > free_bytes):
                    fallback_reason = "full_parameter_gradients_exceed_free_vram"
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = model_class(first_layers(weights, 2), precision=precision,
                                      consistency=consistency)
                    gradient_check = check_gradients(model) if args.verify_gradients else None
                retry_two_layers = False
                try:
                    if args.verify_gradients and workload == "train" and model.layers != full_layers and gradient_check is None:
                        args.measure_phase = "gradient_check"
                        print(json.dumps(dict(phase="gradient_check",status="starting",layers=model.layers,
                                              precision=precision,consistency=consistency)),flush=True)
                        gradient_check = check_gradients(model) if args.verify_gradients else None
                    record = run_case(model, workload, b, t, args) | {"status": "ok"}
                except torch.OutOfMemoryError:
                    model.zero_grad(set_to_none=True)
                    record = dict(workload=workload, B=b, T=t, precision=precision,
                                  consistency=consistency, status="oom",
                                  phase=getattr(args, "measure_phase", "unknown"))
                    retry_two_layers = (args.layers is None and workload == "train"
                                        and model.width >= 4096 and model.layers > 2)
                # Retry outside the exception handler so its traceback cannot retain the full model.
                if retry_two_layers:
                    fallback_reason = "full_training_oom"
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = model_class(first_layers(weights, 2), precision=precision,
                                      consistency=consistency)
                    gradient_check = check_gradients(model) if args.verify_gradients else None
                    try:
                        record = run_case(model, workload, b, t, args) | {"status": "ok"}
                    except torch.OutOfMemoryError:
                        model.zero_grad(set_to_none=True)
                        record = dict(workload=workload, B=b, T=t, precision=precision,
                                      consistency=consistency, status="oom")
                record.update(full_layers=full_layers, measured_layers=model.layers,
                              scope="full_model" if model.layers == full_layers else "partial_model",
                              fallback_reason=fallback_reason, gradient_check=gradient_check)
                document["cases"].append(record)
                args.output.write_text(json.dumps(document, indent=2) + "\n")
                print(json.dumps({k: v for k, v in record.items() if k not in
                                 ("forward_ms", "backward_ms", "wall_ms", "total_ms")}
                                 | {"p10_ms": record.get("total_ms", {}).get("p10")}), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
