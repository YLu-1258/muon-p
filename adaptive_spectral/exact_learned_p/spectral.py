"""Exact spectral geometry and its local exponent derivative."""
import math

import torch


def spectral_direction(momentum, p, *, driver='gesvd', derivative=False):
    """Unit-RMS direction; differentiate its RMS normalization as well as s**p.

    Momentum and SVD factors are constants in this local hypergradient. Actual
    training calls use FP32. FP64 is preserved for numerical reference tests.
    """
    g = momentum.double() if momentum.dtype == torch.float64 else momentum.float()
    norm = g.norm()
    if norm == 0:
        return torch.zeros_like(g), torch.zeros_like(g) if derivative else None
    g = g / norm
    kwargs = {'driver': driver} if g.is_cuda and driver != 'default' else {}
    u, s, vh = torch.linalg.svd(g, full_matrices=False, **kwargs)
    powers = torch.where(s > 0, s.pow(p), 0)
    scale = math.sqrt(g.numel()) / powers.norm().clamp_min(torch.finfo(g.dtype).tiny)
    d = (u * (powers * scale)) @ vh
    dp = None
    if derivative:
        if p <= 0:
            raise ValueError('Exponent derivative requires p>0')
        logs = s.clamp_min(torch.finfo(s.dtype).tiny).log()
        mean_log = (powers.square() * logs).sum() / powers.square().sum()
        dp = (u * (powers * scale * (logs - mean_log))) @ vh
    if not torch.isfinite(d).all() or (dp is not None and not torch.isfinite(dp).all()):
        raise FloatingPointError('Nonfinite spectral update or exponent derivative')
    return d, dp


def split_parameters(model):
    """Exclude embeddings/output heads by identity, including tied weights."""
    excluded = set()
    for accessor in ('get_input_embeddings', 'get_output_embeddings'):
        module = getattr(model, accessor, lambda: None)()
        if module is not None:
            excluded.update(id(p) for p in module.parameters())
    matrices, fallback = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (matrices if p.ndim == 2 and id(p) not in excluded else fallback).append((name, p))
    return matrices, fallback


class ExactMuon(torch.optim.Optimizer):
    def __init__(self, named_parameters, lr=2e-5, momentum=0.95, weight_decay=0.0, driver='gesvd'):
        self.names = {id(p): name for name, p in named_parameters}
        self.driver = driver
        super().__init__([p for _, p in named_parameters], dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, p, derivative=False, progress=None):
        derivatives = []
        total = sum(len(group['params']) for group in self.param_groups)
        index = 0
        for group in self.param_groups:
            for parameter in group['params']:
                index += 1
                if parameter.grad is None:
                    continue
                if progress:
                    progress(f'SVD {index}/{total}: {self.names[id(parameter)]}')
                grad = parameter.grad.float()
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buffer = state['momentum_buffer']
                buffer.mul_(group['momentum']).add_(grad)
                nesterov = grad + group['momentum'] * buffer
                direction, dp = spectral_direction(nesterov, p, driver=self.driver, derivative=derivative)
                parameter.mul_(1-group['lr']*group['weight_decay'])
                parameter.add_(direction, alpha=-group['lr'])
                if derivative:
                    derivatives.append((parameter, dp, group['lr']))
        return derivatives


def hypergradient(derivatives):
    """Contract current loss gradients with the preceding update's derivatives."""
    if not derivatives:
        raise ValueError('No derivative directions')
    result = torch.zeros((), device=derivatives[0][0].device, dtype=derivatives[0][1].dtype)
    for parameter, dp, lr in derivatives:
        if parameter.grad is not None:
            result.add_((parameter.grad.to(dp.dtype)*dp).sum(), alpha=-lr)
    return result
