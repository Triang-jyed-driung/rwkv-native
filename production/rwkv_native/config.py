"""Execution choices; consistency constrains forward arithmetic only."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeConfig:
    precision: str = "fp16"
    consistency: str = "strict"

    def __post_init__(self):
        for name, choices in (
            ("precision", ("fp16", "bf16")),
            ("consistency", ("strict", "none")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")

    @property
    def dtype(self):
        return torch.float16 if self.precision == "fp16" else torch.bfloat16

    @classmethod
    def resolve(cls, *, precision, consistency, dtype):
        # Keep the existing dtype= API, but reject contradictory requests.
        if dtype is not None:
            names = {torch.float16: "fp16", torch.bfloat16: "bf16"}
            if dtype not in names:
                raise ValueError("dtype must be torch.float16 or torch.bfloat16")
            if precision is not None and precision != names[dtype]:
                raise ValueError("precision and dtype disagree")
            precision = names[dtype]
        return cls("fp16" if precision is None else precision, consistency)
