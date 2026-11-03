"""Colored read-only terminal dashboard. No torch or training dependencies."""
import argparse
import json
from pathlib import Path
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'


def number(value, precision=4):
    return '--' if value is None else f'{value:.{precision}f}'


def sparkline(values):
    if not values:
        return '--'
    values = values[-48:]
    low, high = min(values), max(values)
    levels = '▁▂▃▄▅▆▇█'
    return ''.join(levels[min(7,int(7*(v-low)/(high-low)))] if high>low else levels[3] for v in values)


def render(root, color=True):
    def paint(text, code):
        return f'\033[{code}m{text}\033[0m' if color else text
    try:
        state = json.loads((root/'progress.json').read_text())
    except (OSError, ValueError):
        return paint('EXACT SVD | LEARNED p', '1;36')+f'\nRun: {root}\nWaiting for progress.json\n'
    try:
        text = (root/'metrics.jsonl').read_text()
        metrics = [json.loads(line) for line in text[:text.rfind('\n')+1].splitlines() if line]
    except (OSError, ValueError):
        metrics = []
    steps = [row for row in metrics if row['event']=='step']
    validations = [row for row in metrics if row['event']=='validation']
    status = state['state']
    code = '32' if status=='complete' else '31' if status in ('failed','cancelled') else '36'
    fraction = min(1,state['step']/state['steps'])
    bar = '█'*int(32*fraction)+'░'*(32-int(32*fraction))
    end = time.time() if status=='running' else state['updated_at']
    elapsed = duration(end-state['started_at'])
    lines = [paint('EXACT SVD  /  LEARNED SPECTRAL EXPONENT', '1;36'),
             f'Run: {root.resolve()}',
             paint(f"{status.upper()}   mode={state['mode']}   PID={state['pid']}   GPU={state.get('gpu') or state.get('device','--')}",code),
             paint(f"[{bar}] {fraction:6.1%}   {state['step']}/{state['steps']} optimizer steps",'1;32'),
             f"Elapsed {elapsed}   ETA ~{duration(state['eta_seconds']) if 'eta_seconds' in state else '--'}   Last status {duration(time.time()-state['updated_at'])} ago",
             paint('Phase: '+state['phase'],'33'), '',
             paint(f"p = {number(state.get('p'),6)}    delayed p updates = {state.get('p_updates',0)}",'1;35'),
             f"Train CE {number(state.get('train_loss'))}    delayed dL/dp {number(state.get('hypergradient_p'),6)}",
             paint(f"Selection CE {number(state.get('validation_loss'))}    Best selection CE {number(state.get('best_validation'))}",'1;32'),
             f"Last step {number(state.get('step_seconds'),2)} s    Optimizer {number(state.get('optimizer_seconds'),2)} s    Peak allocated {number(state.get('peak_allocated_gib'),2)} GiB",
             f"Response tokens: train={state.get('train_tokens',0):,}   Extra meta passes: 0",
             f"Splits: {state.get('split_sizes','loading')}   Matrix parameters: {state.get('matrix_count','--')}", '',
             paint('p trajectory       '+sparkline([r['p_next'] for r in steps]),'35'),
             paint('Selection CE       '+sparkline([r['loss'] for r in validations]),'32'), '',
             paint('RECENT STEPS','1;36'),
             f'{"STEP":>6} {"TRAIN CE":>10} {"p BEFORE":>10} {"p USED":>10} {"OPT s":>9} {"STEP s":>9}']
    for row in steps[-6:]:
        lines.append(f'{row["step"]:6} {row["train_loss"]:10.4f} '
                     f'{row.get("p_before",row["p_used"]):10.6f} {row["p_used"]:10.6f} {row["optimizer_seconds"]:9.2f} {row["step_seconds"]:9.2f}')
    lines += ['', 'CE = response-token cross-entropy; model-selection data never adapts p.',
              'p updates reuse the next training gradient before clipping; first step only updates W.',
              'Sparklines scale each history independently.',
              'RUNNING is last reported state; a killed process can leave stale status.',
              'Ctrl-C stops this watcher only.']
    if state.get('error'):
        lines += [paint('ERROR: '+state['error'].replace('\n',' ')[:250],'1;31')]
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path('adaptive_spectral/exact_learned_p/runs/learned'))
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()
    print(render(args.output, not args.no_color), end='')
