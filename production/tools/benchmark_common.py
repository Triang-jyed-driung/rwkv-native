"""Shared timing and source fingerprints for end-to-end validation."""

import gc
import hashlib
import json
import time
from pathlib import Path

import torch


def distribution(values):
    """The sole latency statistic is p10; retain samples for reproducibility."""
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"p10": percentile(.1), "samples": values}


def measure(forward, backward, args, trace_name, *, prepare_capture=None):
    """Measure training F/B in one iteration; backward is a VJP, not optimizer.step."""
    args.measure_phase = "warmup"
    if args.execution != "graph":
        for _ in range(args.warmup):
            result = forward()
            if backward:
                backward(result)
            del result
    torch.cuda.synchronize()
    graph = None
    training = backward is not None
    inputs_resident = torch.cuda.memory_allocated()
    capture_peak = None
    if args.execution == "graph":
        args.measure_phase = "capture_warmup"
        # Metadata providers can bind validation tickets to a CUDA stream.
        # Prepare them on the exact stream that will be captured, outside
        # capture and outside timing, for all compared implementations.
        try:
            capture_stream = torch.cuda.Stream()
        except RuntimeError as error:
            # Stream creation can report driver OOM as AcceleratorError too.
            # Keep it in the same explicit OOM accounting as graph capture.
            if "CUDA error: out of memory" in str(error):
                raise torch.OutOfMemoryError(str(error)) from error
            raise
        with torch.cuda.stream(capture_stream):
            for _ in range(args.warmup):
                result = forward()
                if backward:
                    backward(result)
                del result
        capture_stream.synchronize()
        if prepare_capture is not None:
            prepare_capture()
        gc.collect()
        torch.cuda.empty_cache()
        inputs_resident = torch.cuda.memory_allocated()
        args.measure_phase = "capture"
        torch.cuda.reset_peak_memory_stats()
        graph = torch.cuda.CUDAGraph()
        # Retain graph outputs/gradients to own their captured storage.
        try:
            with torch.cuda.graph(graph, stream=capture_stream):
                captured_result = forward()
                captured_grads = backward(captured_result) if backward else None
        except RuntimeError as error:
            # Driver-side graph instantiation can report AcceleratorError rather
            # than the allocator's OutOfMemoryError. Preserve the OOM record;
            # unrelated capture/kernel failures must still terminate the run.
            if "CUDA error: out of memory" in str(error):
                raise torch.OutOfMemoryError(str(error)) from error
            raise
        # The capture-stream warmup above is the only warmup. Capture records
        # the graph; the next replay is the single timed sample by default.
        torch.cuda.synchronize()
        capture_peak = torch.cuda.max_memory_allocated()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    forward_ms, backward_ms, total_ms, wall_ms = [], [], [], []
    start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    args.measure_phase = "measure"
    for _ in range(args.repeats):
        before = time.perf_counter()
        start.record()
        if graph is not None:
            graph.replay()
        else:
            result = forward()
            middle.record()
            if backward:
                backward(result)
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - before) * 1000)
        if graph is None:
            forward_ms.append(start.elapsed_time(middle))
            backward_ms.append(middle.elapsed_time(end))
        total_ms.append(start.elapsed_time(end))
        if graph is None:
            del result
    record = {
        # p10 remains a compatibility key for archived readers. With one
        # sample, this metadata explicitly identifies a single observation.
        "latency_statistic": "single_sample" if args.repeats == 1 else "p10",
        "warmup_passes": args.warmup,
        "total_ms": distribution(total_ms),
        "wall_ms": distribution(wall_ms),
        "resident_bytes": baseline, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_increment_bytes": torch.cuda.max_memory_allocated() - baseline,
        "inputs_resident_bytes": inputs_resident,
        "capture_peak_allocated_bytes": capture_peak,
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    if graph is None:
        record["forward_ms"] = distribution(forward_ms)
    elif not training:
        record["forward_ms"] = record["total_ms"]
    if backward and graph is None:
        record["backward_ms"] = distribution(backward_ms)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        path = args.trace_dir / (trace_name + ".json")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            if graph is not None:
                graph.replay()
            else:
                result = forward()
                if backward:
                    backward(result)
            torch.cuda.synchronize()
        if graph is None:
            del result
        prof.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
        record.update(kernel_launches=len(kernels),
                      profiled_kernel_time_ms=sum(e.get("dur", 0) for e in kernels) / 1000,
                      trace=str(path))
    return record


def source_hash(include_benchmarks=True):
    import rwkv_native
    root = Path(__file__).resolve().parents[1]
    package = Path(rwkv_native.__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(p for p in package.rglob("*")
                   if p.suffix in (".py", ".cu", ".cpp", ".cuh") and "third_party" not in p.parts)
    for path in files:
        digest.update(("rwkv_native/" + str(path.relative_to(package))).encode())
        digest.update(path.read_bytes())
    for path in [root / "tools" / name for name in ("benchmark_common.py", "benchmark_model.py")] if include_benchmarks else ():
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


