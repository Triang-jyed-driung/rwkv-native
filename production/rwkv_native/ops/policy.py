"""Choose a projection backend once per weight shape, before graph capture."""

import warnings
from dataclasses import dataclass

import torch

from .gemm import prepare_lt

LOW_RANK_SUFFIXES = tuple(".att." + name + str(stage) for name in "wagv" for stage in (1, 2))
PRE_OUTPUTS = (64, 96, 256, 2048)
EXPANSION_SHAPES = ((96, 2048), (96, 2048), (256, 2048), (64, 2048))


def projection_weights(weights):
    """Yield logical [input,output] matrices from checkpoint parameter names."""
    for key, weight in weights.items():
        name = key.replace("__", ".")
        if weight.ndim != 2 or name == "emb.weight":
            continue
        if name.endswith(".weight"):
            yield weight.T
        elif name.endswith(LOW_RANK_SUFFIXES):
            yield weight


@dataclass
class ProjectionPolicy:
    backends: dict
    report: dict
    lt_shapes: frozenset

    @classmethod
    def build(cls, weights, *, backend, equivalent_tiles, grouped_inputs):
        grouped = {}
        for weight in projection_weights(weights):
            grouped.setdefault(tuple(weight.shape), []).append(weight)
        choices, report = {}, {}
        lt_backend = "lt_equivalent" if equivalent_tiles else "lt"
        for shape, matrices in grouped.items():
            weight = matrices[0]
            aligned = all(w.data_ptr() % 16 == 0 for w in matrices)
            if not aligned:
                selected, reason = "fixed", "unaligned weight uses the fixed CUDA fallback"
            elif (
                grouped_inputs
                and backend != "fixed"
                and weight.dtype == torch.float16
                and shape[0] == 2048
                and shape[1] in PRE_OUTPUTS
            ):
                selected, reason = (
                    "pre_group",
                    "shared K8/K16 sums in grouped and single-weight paths",
                )
            elif backend == "fixed":
                selected, reason = "fixed", "fixed CUDA requested"
            else:
                supported, reason = prepare_lt(weight, equivalent_tiles)
                selected = lt_backend if supported else "fixed"
                if not supported and (
                    "metadata differs" in reason or "retarget validation failed" in reason
                ):
                    warnings.warn(
                        f"{shape}: {reason}; using fixed CUDA for every row count",
                        RuntimeWarning,
                        stacklevel=3,
                    )
            choices[shape] = selected
            label = {"fixed": "fixed CUDA", "pre_group": "grouped SM80 CUDA"}.get(
                selected, "cuBLASLt"
            )
            report[shape] = {"backend": label, "reason": reason}
        return cls(
            choices,
            report,
            frozenset(s for s, b in choices.items() if b.startswith("lt")),
        )
