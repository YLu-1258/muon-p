"""Read-only terminal snapshot. Run under watch; requires only standard Python."""
import argparse
import csv
import json
import math
from fractions import Fraction
from pathlib import Path
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def snapshot(root):
    try:
        meta = json.loads((root / 'metadata.json').read_text())
    except (OSError, ValueError):
        return f'APPROXIMATION BENCHMARK\nRun: {root}\nWaiting for metadata.json\n'
    csv_path = root / 'results.csv'
    try:
        # Ignore a partially written final record while the benchmark appends.
        content = csv_path.read_text()
        content = content[:content.rfind('\n') + 1]
        rows = list(csv.DictReader(content.splitlines()))
        updated = csv_path.stat().st_mtime
    except OSError:
        rows, updated = [], time.time()
    per_case = sum(1 if Fraction(p) == 1 else len(meta['steps']) *
                   (1 if Fraction(p) == 0 else len(meta['coefficient_fractions']))
                   for p in meta['exponents'])
    cases = 1 if meta.get('input') else len(meta['shapes']) * len(meta['spectra'])
    total = cases * len(meta['dtypes']) * per_case
    done = len(rows)
    fraction = min(1, done / total) if total else 0
    started = meta.get('started_at')
    elapsed = (updated if done >= total else time.time()) - started if started else None
    eta = '~' + duration(elapsed * (total-done)/done) if elapsed is not None and 0 < done < total else '--'
    bar = '#' * int(30*fraction) + '-' * (30-int(30*fraction))
    lines = ['APPROXIMATION BENCHMARK', f'Run: {root.resolve()}',
             f"GPU: {meta.get('gpu') or meta.get('device')}   Seed: {meta['seed']}",
             f'[{bar}] {fraction:6.1%}   {done:,}/{total:,} measurements',
             f'Elapsed: {duration(elapsed) if elapsed is not None else "--"}   ETA: {eta}   Last result: {duration(time.time()-updated) if rows else "--"} ago',
             'State: ' + ('COMPLETE' if done >= total else 'INCOMPLETE (may be running or stopped)'),
             '', 'LATEST MEASUREMENTS',
             f'{"SHAPE":<14} {"SPECTRUM":<15} {"DTYPE":<9} {"p":>5} {"ITER":>5} {"c":>8} {"REL ERR":>10} {"COSINE":>9} {"MS":>9}']
    for row in rows[-8:]:
        lines.append(f'{row["shape"]:<14} {row["spectrum"]:<15} {row["dtype"]:<9} '
                     f'{row["exponent"]:>5} {row["steps"]:>5} {float(row["coefficient"]):8.4f} '
                     f'{float(row["relative_error"]):10.3g} {float(row["cosine"]):9.5f} {float(row["milliseconds"]):9.2f}')
    lines += ['', 'BEST DIRECTIONAL ERROR SO FAR (equal-norm error; all completed cases)',
              f'{"p":>5} {"DTYPE":<9} {"ERROR":>10} {"ITER":>5} {"c":>8} {"SHAPE / SPECTRUM"}']
    best = {}
    for row in rows:
        error = float(row['normalized_error'])
        key = (row['exponent'], row['dtype'])
        if math.isfinite(error) and (key not in best or error < float(best[key]['normalized_error'])):
            best[key] = row
    for key in sorted(best, key=lambda key: (Fraction(key[0]), key[1])):
        row = best[key]
        lines.append(f'{key[0]:>5} {key[1]:<9} {float(row["normalized_error"]):10.3g} '
                     f'{row["steps"]:>5} {float(row["coefficient"]):8.4f} {row["shape"]} / {row["spectrum"]}')
    lines += ['', 'Best rows are individual cases, not aggregate coefficient recommendations.',
              'ETA uses completed measurement counts; costs vary by configuration.',
              'Progress advances after each measurement. Ctrl-C stops only this watcher.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path('results'))
    print(snapshot(parser.parse_args().output), end='')
