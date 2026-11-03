"""Render a read-only benchmark snapshot for the terminal watch command."""
import argparse
import csv
import json
from pathlib import Path
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'


def snapshot(root):
    meta, state = read_json(root/'metadata.json'), read_json(root/'progress.json')
    if not meta:
        return f'SVD BENCHMARK\nRun: {root}\nWaiting for metadata.json\n'
    try:
        text = (root/'results.csv').read_text()
        rows = list(csv.DictReader(text[:text.rfind('\n')+1].splitlines()))
    except OSError:
        rows = []
    complete = len(rows)
    fraction = complete/meta['total']
    running = state.get('state') == 'running'
    end = time.time() if running else state.get('updated_at', time.time())
    elapsed = end-meta['started_at']
    eta = '~'+duration(elapsed*(meta['total']-complete)/complete) if running and complete else '--'
    lines = ['SVD BENCHMARK | LLAMA-3.2-1B', f'Run: {root.resolve()}',
             f"GPU: {meta.get('gpu') or meta['device']}   PID: {meta['pid']}   State: {state.get('state', 'starting').upper()}",
             f'Progress: {complete}/{meta["total"]} [{"#"*int(30*fraction)}{"-"*(30-int(30*fraction))}] {fraction:.1%}',
             f'Elapsed: {duration(elapsed)}   ETA: {eta}',
             f"Current: {state.get('current', '--')} | {state.get('phase', '--')}",
             f"Last status: {duration(time.time()-state.get('updated_at',time.time()))} ago",
             '', f'{"CASE":<17} {"DRIVER":<8} {"MODE":<18} {"MEDIAN MS":>10} {"EXTRA MiB":>10} {"SVD ERR":>10}']
    for row in rows:
        if row['status'] != 'ok':
            lines.append(f'{row["case"]} {row["driver"]} {row["mode"]}: FAILED {row["error"][:100]}')
        else:
            lines.append(f'{row["case"]:<17} {row["driver"]:<8} {row["mode"]:<18} '
                         f'{float(row["median_ms"]):10.2f} {row["extra_peak_mib"]:>10.10} {float(row["svd_reconstruction_error"]):10.2g}')
    lines += ['', 'PROJECTED SEQUENTIAL OPTIMIZER COST (112 matrices per step)']
    if meta['smoke']:
        lines.append('SMOKE TEST: tiny matrices; training estimates disabled.')
    else:
        for driver in meta['drivers']:
            for mode in meta['modes']:
                matching = [r for r in rows if r['driver']==driver and r['mode']==mode and r['status']=='ok']
                if len(matching) != len(meta['cases']):
                    lines.append(f'{driver} / {mode}: awaiting all four successful matrix cases')
                    continue
                seconds = sum(float(r['median_ms'])*int(r['count']) for r in matching)/1000
                lines.append(f'{driver} / {mode}: {seconds:.2f} s/step; {meta["training_steps"]} steps: {duration(seconds*meta["training_steps"])}')
    lines += ['', 'Includes conversion, normalization, SVD, reconstruction and BF16 output.',
              'Derivative mode also reconstructs d(update)/dp; neither mode includes training/meta passes.',
              'Synthetic inputs; driver costs can differ on actual momentum spectra.',
              'ETA is approximate: cases have different costs. A long SVD may leave status unchanged.',
              'RUNNING is last reported state; SIGKILL or host failure can leave it stale.',
              'Ctrl-C stops only the watcher.']
    if state.get('error'):
        lines.append('Error: '+state['error'])
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path('adaptive_spectral/svd_benchmark/results/default'))
    print(snapshot(parser.parse_args().output), end='')
