"""Standalone spectral recurrence benchmark; no training dependencies."""
import argparse
import csv
import json
import math
import platform
import time
from fractions import Fraction
from pathlib import Path

import torch


def odd_power(x, degree):
    """U S^degree V.T using the smaller Gram matrix; return matmul count."""
    if degree == 1:
        return x, 0
    gram = x @ x.T
    result = x
    # Deliberately simple, auditable evaluation; counts reflect this implementation.
    for _ in range((degree - 1) // 2):
        result = gram @ result
    return result, 1 + (degree - 1) // 2


def direction(g, exponent, steps, coefficient):
    transposed = g.shape[0] > g.shape[1]
    y = g.T.contiguous() if transposed else g.clone()
    x = y.clone()
    count = 0
    a, b = exponent.numerator, exponent.denominator
    if a == b:
        return g.clone(), 0
    if a == 0:
        for _ in range(steps):
            gram = x @ x.T
            cubic = gram @ x
            x = 3.4445 * x - 4.7750 * cubic + 2.0315 * (gram @ cubic)
            count += 3
    elif a % 2 and b % 2:
        source, cost = odd_power(y, a)
        count += cost
        for _ in range(steps):
            target, cost = odd_power(x, b)
            x = x + coefficient * (source - target)
            count += cost
    else:
        # f(x,y)=x+c*(y^(2a)-x^(2b))*y, an odd bivariate polynomial.
        # In matrix form even powers act on the left of the original y.
        yy = y @ y.T
        source = torch.linalg.matrix_power(yy, a) @ y
        # matrix_power count is calculated by the same binary powering scheme.
        def power_cost(n):
            return n.bit_length() - 1 + n.bit_count() - 1
        count += 2 + power_cost(a)
        for _ in range(steps):
            xx = x @ x.T
            target = torch.linalg.matrix_power(xx, b) @ y
            x = x + coefficient * (source - target)
            count += 2 + power_cost(b)
    return (x.T if transposed else x), count


def synthetic(rows, cols, spectrum, seed):
    generator = torch.Generator().manual_seed(seed)
    rank = min(rows, cols)
    u = torch.linalg.qr(torch.randn(rows, rank, generator=generator, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(cols, rank, generator=generator, dtype=torch.float64))[0]
    if spectrum == 'flat':
        s = torch.ones(rank, dtype=torch.float64)
    else:
        s = torch.logspace(0, -6, rank, dtype=torch.float64)
        if spectrum == 'rank_deficient':
            s[rank // 2:] = 0
    return (u * s) @ v.T


def metrics(output, reference, u, s, vh):
    z = output.double().cpu()
    if not torch.isfinite(z).all():
        return dict(relative_error=math.inf, normalized_error=math.inf,
                    cosine=math.nan, rms_ratio=math.inf, small_mode_error=math.inf)
    norm = z.norm().clamp_min(1e-30)
    refnorm = reference.norm().clamp_min(1e-30)
    # Project onto reference singular vectors, exposing errors hidden by total norm.
    projected = (u.T @ z @ vh.T).diagonal()
    expected = (u.T @ reference @ vh.T).diagonal()
    small = (s > s.max() * 1e-12) & (s < s.max() * 1e-3)
    small_error = ((projected[small] - expected[small]).norm()
                   / expected[small].norm().clamp_min(1e-30)) if small.any() else torch.tensor(float('nan'))
    return dict(relative_error=((z-reference).norm()/refnorm).item(),
                normalized_error=(z/norm-reference/refnorm).norm().item(),
                cosine=((z*reference).sum()/norm/refnorm).item(),
                rms_ratio=(norm/refnorm).item(), small_mode_error=small_error.item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shapes', nargs='+', default=['128x128', '128x512', '512x128'])
    parser.add_argument('--spectra', nargs='+', choices=['flat', 'log', 'rank_deficient'], default=['flat', 'log', 'rank_deficient'])
    parser.add_argument('--exponents', nargs='+', default=['0', '1/7', '2/7', '3/7', '4/7', '5/7', '6/7', '1', '1/3'])
    parser.add_argument('--steps', nargs='+', type=int, default=[6, 12, 24, 48, 96])
    parser.add_argument('--coefficient-fractions', nargs='+', type=float, default=[0.25, 0.5, 0.9])
    parser.add_argument('--dtypes', nargs='+', choices=['float32', 'bfloat16'], default=['float32', 'bfloat16'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input', type=Path, help='Optional torch.save file containing one real 2D momentum tensor; replaces synthetic inputs')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.steps):
        parser.error('repeats and steps must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu for a functional smoke test')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(started_at=time.time(), torch=torch.__version__, python=platform.python_version(),
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    cuda=torch.version.cuda, tf32=False)
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    cases = [('real', 'real')] if args.input else [(shape, spectrum) for shape in args.shapes for spectrum in args.spectra]
    with (args.output/'results.csv').open('w', newline='') as handle, torch.inference_mode():
        writer = None
        for shape, spectrum in cases:
            g = torch.load(args.input, map_location='cpu', weights_only=True).double() if args.input else synthetic(*map(int, shape.split('x')), spectrum, args.seed)
            if g.ndim != 2 or not torch.isfinite(g).all() or g.norm() == 0:
                raise ValueError('Input must be a finite nonzero 2D tensor')
            g = g / g.norm()  # identical Frobenius normalization for every endpoint
            u, s, vh = torch.linalg.svd(g, full_matrices=False)
            for dtype_name in args.dtypes:
                dtype = getattr(torch, dtype_name)
                y = g.to(device=device, dtype=dtype)
                # Warm up actual precision and device before measuring configurations.
                for _ in range(3):
                    _ = y @ y.T
                sync()
                for exponent_text in args.exponents:
                    p = Fraction(exponent_text)
                    if not 0 <= p <= 1:
                        raise ValueError('Exponents must be in [0,1]')
                    powers = torch.where(s > s.max()*1e-12, s.pow(float(p)), 0) if p == 0 else s.pow(float(p))
                    reference = (u * powers) @ vh
                    # Odd case c<2/b; even case c<1/b is conservative for c*y.
                    bound = 2/p.denominator if p.numerator % 2 else 1/p.denominator
                    fractions = args.coefficient_fractions if p not in (0, 1) else [0.0]
                    for fraction in fractions:
                        coefficient = fraction * bound
                        for steps in (args.steps if p != 1 else [0]):
                            direction(y, p, steps, coefficient)
                            sync()
                            if device.type == 'cuda':
                                torch.cuda.reset_peak_memory_stats(device)
                            start = time.perf_counter()
                            for _ in range(args.repeats):
                                output, count = direction(y, p, steps, coefficient)
                            sync()
                            elapsed = (time.perf_counter()-start)*1000/args.repeats
                            row = dict(shape=str(tuple(g.shape)), spectrum=spectrum, dtype=dtype_name,
                                       exponent=str(p), steps=steps, coefficient=coefficient,
                                       matmuls=count, milliseconds=elapsed,
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                                       **metrics(output, reference, u, s, vh))
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            handle.flush()
            print(f'Finished {shape} / {spectrum}', flush=True)
    print(f'Results: {args.output / "results.csv"}')


if __name__ == '__main__':
    main()
