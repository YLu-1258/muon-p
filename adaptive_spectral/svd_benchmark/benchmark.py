"""Time reduced FP32 SVD spectral updates at Llama-3.2-1B matrix sizes."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import time

import torch


CASES = [('attention_qo', 2048, 2048, 32),
         ('attention_kv', 512, 2048, 32),
         ('mlp_gate_up', 8192, 2048, 32),
         ('mlp_down', 2048, 8192, 16)]


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def spectral_update(momentum, p, driver, derivative):
    g = momentum.float()
    g = g / g.norm().clamp_min(1e-30)
    kwargs = {} if driver == 'default' else {'driver': driver}
    u, s, vh = torch.linalg.svd(g, full_matrices=False, **kwargs)
    powers = s.pow(p)
    update = ((u * powers) @ vh).to(momentum.dtype)
    dp = None
    if derivative:
        # d(s^p)/dp at s=0 is zero for p>0; clamp only inside log.
        weights = powers * s.clamp_min(torch.finfo(s.dtype).tiny).log()
        dp = ((u * weights) @ vh).to(momentum.dtype)
    return update, dp, (g, u, s, vh)


def run(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu --smoke for a functional check')
    if device.type != 'cuda' and any(d != 'default' for d in args.drivers):
        raise ValueError('CPU requires --drivers default')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    cases = [(name, max(4, m//128), max(4, n//128), count)
             for name, m, n, count in CASES] if args.smoke else CASES
    configs = [(case, driver, mode) for driver in args.drivers
               for mode in args.modes for case in cases]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'metadata.json').exists():
        raise ValueError('Output already contains a run; choose a fresh --output directory')
    metadata = dict(started_at=time.time(), pid=os.getpid(), device=str(device),
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    torch=torch.__version__, cuda=torch.version.cuda, tf32=False,
                    p=args.p, seed=args.seed, repeats=args.repeats, warmup=args.warmup,
                    drivers=args.drivers, modes=args.modes, smoke=args.smoke,
                    cases=cases, training_steps=args.training_steps,
                    total=len(configs), input_dtype='bfloat16')
    atomic_json(args.output/'metadata.json', metadata)
    state = dict(state='running', completed=0, total=len(configs), current='initializing', updated_at=time.time())

    def publish(**kwargs):
        state.update(kwargs, updated_at=time.time())
        atomic_json(args.output/'progress.json', state)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    fields = ['case', 'rows', 'cols', 'count', 'driver', 'mode', 'status',
              'median_ms', 'min_ms', 'max_ms', 'peak_allocated_mib',
              'extra_peak_mib', 'svd_reconstruction_error', 'error']
    publish()
    print(f'Watch: watch --no-title -n 2 python {Path(__file__).with_name("status.py")} {args.output}', flush=True)
    failures = 0
    try:
        with (args.output/'results.csv').open('w', newline='') as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            handle.flush()
            for index, ((name, m, n, count), driver, mode) in enumerate(configs):
                row = dict(case=name, rows=m, cols=n, count=count, driver=driver, mode=mode)
                label = f'{name} {m}x{n} | {driver} | {mode}'
                publish(current=label, phase='allocating')
                momentum = output = factors = dp = None
                try:
                    # Reset seed so all drivers and modes see the same case input.
                    torch.manual_seed(args.seed + next(i for i,c in enumerate(cases) if c[0] == name))
                    momentum = torch.randn(m, n, device=device, dtype=torch.bfloat16)
                    for repeat in range(args.warmup):
                        publish(phase=f'warmup {repeat+1}/{args.warmup}')
                        output, dp, factors = spectral_update(momentum, args.p, driver, mode == 'update_derivative')
                        sync()
                        del output, dp, factors
                        output = dp = factors = None
                    sync()
                    baseline = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    samples = []
                    for repeat in range(args.repeats):
                        publish(phase=f'measurement {repeat+1}/{args.repeats}')
                        start = time.perf_counter()
                        output, dp, factors = spectral_update(momentum, args.p, driver, mode == 'update_derivative')
                        sync()
                        samples.append((time.perf_counter()-start)*1000)
                        if repeat != args.repeats-1:
                            del output, dp, factors
                            output = dp = factors = None
                    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
                    publish(phase='checking reconstruction')
                    g, u, s, vh = factors
                    residual = (((u*s)@vh-g).norm()/g.norm()).item()
                    del g, u, s, vh
                    if not torch.isfinite(output).all() or (dp is not None and not torch.isfinite(dp).all()):
                        raise ValueError('Nonfinite update or exponent derivative')
                    row.update(status='ok', median_ms=statistics.median(samples), min_ms=min(samples),
                               max_ms=max(samples), peak_allocated_mib=peak/2**20 if peak else '',
                               extra_peak_mib=(peak-baseline)/2**20 if peak else '',
                               svd_reconstruction_error=residual)
                except (RuntimeError, ValueError) as exc:
                    failures += 1
                    row.update(status='failed', error=str(exc).replace('\n', ' ')[:500])
                finally:
                    del momentum, output, dp, factors
                writer.writerow(row)
                handle.flush()
                publish(completed=index+1)
                print(f'{index+1}/{len(configs)} {label}: {row["status"]}', flush=True)
        publish(state='complete' if not failures else 'complete_with_failures', phase='finished')
    except BaseException as exc:
        publish(state='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise
    return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--drivers', nargs='+', choices=['default', 'gesvdj', 'gesvd'], default=['gesvdj', 'gesvd'])
    parser.add_argument('--modes', nargs='+', choices=['update', 'update_derivative'], default=['update', 'update_derivative'])
    parser.add_argument('--p', type=float, default=1/3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--training-steps', type=int, default=600)
    parser.add_argument('--smoke', action='store_true', help='Use tiny scaled shapes; no valid Llama timing estimates')
    parser.add_argument('--output', type=Path, default=Path('adaptive_spectral/svd_benchmark/results/default'))
    args = parser.parse_args()
    if not 0 < args.p <= 1 or args.repeats < 1 or args.warmup < 0 or args.training_steps < 1:
        parser.error('Require 0<p<=1, repeats>=1, warmup>=0, training-steps>=1')
    raise SystemExit(run(args))
