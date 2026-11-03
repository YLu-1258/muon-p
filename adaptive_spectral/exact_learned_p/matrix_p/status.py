"""Colored per-matrix exponent dashboard; standard Python only."""
import argparse
import json
from pathlib import Path
import time

import bootstrap  # noqa: F401
from adaptive_spectral.exact_learned_p.status import duration, number, sparkline


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def render(root,color=True,all_matrices=False):
    def paint(text,code):
        return f'\033[{code}m{text}\033[0m' if color else text
    state=read_json(root/'progress.json')
    if not state:
        return f'EXACT SVD | ONE p PER MATRIX\nWaiting for {root / "progress.json"}\n'
    try:
        text=(root/'metrics.jsonl').read_text()
        steps=[json.loads(line) for line in text[:text.rfind('\n')+1].splitlines() if line]
        steps=[row for row in steps if row['event']=='step']
    except (OSError,ValueError):
        steps=[]
    status=state['state']
    code='32' if status=='complete' else '31' if status in ('failed','cancelled') else '36'
    fraction=state['step']/state['steps']
    end=time.time() if status=='running' else state['updated_at']
    values=state.get('p',{})
    initial=state.get('p_init',1/3)
    gradients=state.get('hypergradient_p') or {}
    names=sorted(values,key=lambda name:abs(values[name]-initial),reverse=True)
    bins=[0]*10
    for p in values.values():
        bins[min(9,int(p*10))]+=1
    lines=[paint('EXACT SVD | ONE INDEPENDENT p PER ELIGIBLE MATRIX','1;36'),
           f'Run: {root.resolve()}',
           paint(f"{status.upper()}  mode={state['mode']}  PID={state['pid']}  GPU={state.get('gpu') or state.get('device','--')}",code),
           paint(f"[{'█'*int(30*fraction)}{'░'*(30-int(30*fraction))}] {fraction:6.1%}  {state['step']}/{state['steps']} steps",'1;32'),
           f"Elapsed {duration(end-state['started_at'])}  ETA ~{duration(state['eta_seconds']) if 'eta_seconds' in state else '--'}  Last status {duration(time.time()-state['updated_at'])} ago",
           paint('Phase: '+state['phase'],'33'),'',
           paint(f"p mean={number(state.get('p_mean'),5)}  median={number(state.get('p_median'),5)}  min={number(state.get('p_min'),5)}  max={number(state.get('p_max'),5)}  std={number(state.get('p_std'),5)}",'1;35'),
           f"Independent exponents: {len(values)}   Delayed p-vector updates: {state.get('p_updates',0)}   Extra meta passes: 0",
           f"Train CE {number(state.get('train_loss'))}  Selection CE {number(state.get('validation_loss'))}  Best CE {number(state.get('best_validation'))}",
           f"Step {number(state.get('step_seconds'),2)} s  Optimizer {number(state.get('optimizer_seconds'),2)} s  Peak {number(state.get('peak_allocated_gib'),2)} GiB  Tokens {state.get('train_tokens',0):,}",
           '',paint('p HISTOGRAM [0,1]   counts in bins of width 0.1','1;35'),
           ' '.join(f'{i/10:.1f}–{(i+1)/10:.1f}:{count}' for i,count in enumerate(bins)),
           'Mean p history '+paint(sparkline([row['p_mean'] for row in steps]),'35'),'',
           paint('MATRICES BY DISTANCE FROM INITIAL p' + (' (ALL)' if all_matrices else ' (TOP 8)'),'1;36'),
           f'{"PARAMETER":<62} {"p":>9} {"CHANGE":>10} {"dL/dp":>12}']
    for name in names if all_matrices else names[:8]:
        lines.append(f'{name[-62:]:<62} {values[name]:9.5f} {values[name]-initial:+10.5f} {number(gradients.get(name),7):>12}')
    lines+=['',paint('RECENT STEPS','1;36'),
            f'{"STEP":>6} {"TRAIN CE":>10} {"MEAN p":>10} {"MIN p":>10} {"MAX p":>10} {"OPT s":>9}']
    for row in steps[-5:]:
        lines.append(f'{row["step"]:6} {row["train_loss"]:10.4f} {row["p_mean"]:10.5f} {row["p_min"]:10.5f} {row["p_max"]:10.5f} {row["optimizer_seconds"]:9.2f}')
    lines+=['','p values are per matrix; summaries are unweighted over eligible matrices.',
            'Gradient columns evaluate the preceding update; p columns show the new update.',
            'First step only updates W; selection validation never supplies gradients to p.',
            'RUNNING is last reported state; a killed process can leave stale status.',
            'Ctrl-C stops this watcher only.']
    if state.get('error'):
        lines.append(paint('ERROR: '+state['error'].replace('\n',' ')[:250],'1;31'))
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=Path('adaptive_spectral/exact_learned_p/matrix_p/runs/learned'))
    parser.add_argument('--no-color',action='store_true')
    parser.add_argument('--all-matrices',action='store_true')
    args=parser.parse_args()
    print(render(args.output,not args.no_color,args.all_matrices),end='')
