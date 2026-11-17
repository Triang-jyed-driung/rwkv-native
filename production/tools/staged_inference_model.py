"""Untimed no-grad validation with one checkpoint block's parameters on CUDA.

The production model executes its normal forward. Only weight storage changes;
all dtypes and strides are copied from a resident two-block model. This helper
rejects autograd and CUDA graph capture and must not be used for benchmarks.
"""
from types import MethodType
import torch


def make_staged_model(weights, **options):
    from rwkv_native import FastRWKV7
    if torch.is_grad_enabled():
        raise RuntimeError('parameter staging is only for no-grad accuracy validation')
    layers = max(int(k.split('.')[1]) for k in weights if k.startswith('blocks.')) + 1
    seed_weights = {k: v for k, v in weights.items()
                    if not k.startswith('blocks.') or int(k.split('.')[1]) < 2}
    model = FastRWKV7(seed_weights, **options)
    def suffix(name):
        return '.'.join(name.split('.')[2:]) if name.startswith('blocks.') else name
    layouts = {}
    for name in seed_weights:
        parameter = model.weight(name)
        signature = (tuple(parameter.shape), tuple(parameter.stride()), parameter.dtype, parameter.device)
        key = suffix(name)
        if key in layouts and layouts[key] != signature:
            raise ValueError(f'inconsistent layer parameter layout: {name}')
        layouts[key] = signature
    del parameter
    # Retain the production policy selected from all logical matrix shapes.
    # Models with additional shapes in later layers are rejected on access.
    model.weights = torch.nn.ParameterDict()
    model.layers = layers
    model._staged_source_weights = weights
    model._staged_weight_cache = {}
    model._staged_weight_group = None
    model._parameter_storage = 'layerwise_cpu_staging'
    def weight(self, name):
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('parameter staging is only for untimed no-grad validation')
        group = '.'.join(name.split('.')[:2]) if name.startswith('blocks.') else name.rsplit('.', 1)[0]
        if group != self._staged_weight_group:
            self._staged_weight_cache.clear()
            self._staged_weight_group = group
        if name not in self._staged_weight_cache:
            original = self._staged_source_weights[name]
            shape, strides, dtype, device = layouts[suffix(name)]
            if tuple(original.shape) != shape:
                raise ValueError(f'parameter shape differs from resident template: {name}')
            value = torch.empty_strided(shape, strides, dtype=dtype, device=device)
            value.copy_(original)
            self._staged_weight_cache[name] = value
        return self._staged_weight_cache[name]
    model.weight = MethodType(weight, model)
    return model
