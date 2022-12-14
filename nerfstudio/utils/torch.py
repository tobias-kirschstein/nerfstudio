from typing import Union, List

from torch import nn


class disable_gradients_for:

    def __init__(self, module_or_modules: Union[nn.Module, List[nn.Module]]):
        if not isinstance(module_or_modules, list):
            module_or_modules = [module_or_modules]
        self._modules = module_or_modules
        self._original_require_grads = [
            {name: p.requires_grad for name, p in m.named_parameters()} for m in self._modules
        ]

    def __enter__(self):
        for m in self._modules:
            m.requires_grad_(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m, original_require_grads in zip(self._modules, self._original_require_grads):
            for name, param in m.named_parameters():
                param.requires_grad = original_require_grads[name]
