"""Independent per-matrix exponents with delayed local hypergradients."""
import math
import statistics

import bootstrap  # noqa: F401
import torch
from torch import nn

from adaptive_spectral.exact_learned_p.spectral import ExactMuon, spectral_direction


def summarize(values):
    numbers = list(values.values())
    return dict(p_mean=statistics.mean(numbers), p_median=statistics.median(numbers),
                p_min=min(numbers), p_max=max(numbers), p_std=statistics.pstdev(numbers))


class MatrixExponents:
    """A vector of independent Adam coordinates, indexed by stable parameter names."""
    def __init__(self, named_parameters, p_init, mode, device):
        self.names = tuple(name for name, _ in named_parameters)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Expected unique matrix parameter names')
        self.indices = {id(parameter): i for i, (_,parameter) in enumerate(named_parameters)}
        self.mode, self.p_init = mode, p_init
        initial_a = math.log(p_init/(1-p_init)) if mode == 'learned' else 0.0
        self.a = nn.Parameter(torch.full((len(self.names),), initial_a, device=device))

    def values(self):
        numbers = self.a.sigmoid().detach().cpu().tolist() if self.mode == 'learned' else [self.p_init]*len(self.names)
        return dict(zip(self.names,numbers))

    @torch.no_grad()
    def hypergradient(self, pending):
        result = torch.zeros_like(self.a)
        for parameter, dp, lr in pending:
            if parameter.grad is not None:
                result[self.indices[id(parameter)]] = -lr*(parameter.grad.to(dp.dtype)*dp).sum()
        return result

    def update(self, pending, previous_p, optimizer):
        grad_p = self.hypergradient(pending)
        if not torch.isfinite(grad_p).all():
            raise FloatingPointError('Nonfinite per-matrix exponent hypergradient')
        previous = torch.tensor([previous_p[name] for name in self.names], device=self.a.device)
        grad_a = grad_p*previous*(1-previous)
        optimizer.zero_grad(set_to_none=True)
        self.a.grad = grad_a
        optimizer.step()
        with torch.no_grad():
            self.a.clamp_(-8,8)
        return (dict(zip(self.names,grad_p.cpu().tolist())),
                dict(zip(self.names,grad_a.cpu().tolist())))


class ExactMatrixMuon(ExactMuon):
    @torch.no_grad()
    def step(self, exponents, derivative=False, progress=None):
        derivatives = []
        total = sum(len(group['params']) for group in self.param_groups)
        index = 0
        for group in self.param_groups:
            for parameter in group['params']:
                index += 1
                if parameter.grad is None:
                    continue
                name = self.names[id(parameter)]
                if progress:
                    progress(f'SVD {index}/{total}: {name} (p={exponents[name]:.4f})')
                grad = parameter.grad.float()
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buffer = state['momentum_buffer']
                buffer.mul_(group['momentum']).add_(grad)
                direction, dp = spectral_direction(grad+group['momentum']*buffer, exponents[name],
                                                   driver=self.driver,derivative=derivative)
                parameter.mul_(1-group['lr']*group['weight_decay'])
                parameter.add_(direction,alpha=-group['lr'])
                if derivative:
                    derivatives.append((parameter,dp,group['lr']))
        return derivatives
