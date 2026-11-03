"""Colored four-method dashboard for concurrent one-GPU-per-method runs."""
import argparse
from pathlib import Path
import time

from comparison import read_json
from status import duration, number


def snapshot(root,color=True):
    def paint(text,code):
        return f'\033[{code}m{text}\033[0m' if color else text
    suite=read_json(root/'suite.json')
    if not suite:
        return f'EXACT SVD | FOUR-METHOD COMPARISON\nWaiting for {root / "suite.json"}\n'
    end=time.time() if suite['state'] in ('running','plotting','preflight') else suite['updated_at']
    lines=[paint('EXACT SVD | FOUR METHODS IN PARALLEL','1;36'),
           f"State: {suite['state'].upper()}   Elapsed: {duration(end-suite['started_at'])}   One GPU per method",
           f'Run: {root.resolve()}','',
           f'{"METHOD":<37} {"GPU":<5} {"STATE":<10} {"STEPS":>9} {"VAL CE":>9} {"BEST CE":>9} {"MEAN p":>9} {"ETA":>10}']
    active=[]
    for method in suite['methods']:
        progress=read_json(root/method['name']/'progress.json')
        state=progress.get('state',method['state']) if method['state']=='running' else method['state']
        counts=f"{progress.get('step',0)}/{progress.get('steps','--')}"
        ps=progress.get('p')
        mean=progress.get('p_mean') if isinstance(ps,dict) else ps
        eta=duration(progress['eta_seconds']) if 'eta_seconds' in progress else '--'
        code='32' if state=='complete' else '31' if state in ('failed','cancelled') else '33' if state=='pending' else '36'
        lines.append(paint(f"{method['label']:<37} {method.get('gpu','--'):<5} {state.upper():<10} {counts:>9} "
                           f"{number(progress.get('validation_loss')):>9} {number(progress.get('best_validation')):>9} "
                           f"{number(mean,5):>9} {eta:>10}",code))
        active.append((method,progress))
    lines+=['',paint('CURRENT PHASES','1;33')]
    for method,progress in active:
        freshness=duration(time.time()-progress['updated_at'])+' ago' if 'updated_at' in progress else '--'
        lines.append(f"GPU {method.get('gpu','--')} / {method['name']}: {progress.get('phase','waiting')}  [status {freshness}]")
        if progress.get('error'):
            lines.append(paint('  '+progress['error'].replace('\n',' ')[:160],'31'))
    matrix=next((progress for method,progress in active if method['name']=='matrix_learned'),{})
    if isinstance(matrix.get('p'),dict):
        lines+=['',paint(f"MATRIX p: min={number(matrix.get('p_min'),5)} median={number(matrix.get('p_median'),5)} max={number(matrix.get('p_max'),5)} std={number(matrix.get('p_std'),5)}",'1;35')]
        changed=sorted(matrix['p'],key=lambda name:abs(matrix['p'][name]-matrix.get('p_init',1/3)),reverse=True)
        for name in changed[:4]:
            lines.append(f'  {name:<65} p={matrix["p"][name]:.5f}')
    if suite['state']=='complete':
        lines+=['',paint('FOUR-METHOD COMPARISON READY','1;32'),str(root/'comparison'/'loss_curves.png'),
                str(root/'comparison'/'p_trajectory.png'),str(root/'comparison'/'matrix_p_heatmap.png'),
                str(root/'comparison'/'summary.csv')]
    if suite.get('error'):
        lines+=['',paint('ERROR: '+suite['error'],'1;31')]
    lines+=['','All methods use exact SVD. RUNNING is the last reported state and can become stale.',
            'Ctrl-C stops this watcher only; interrupt the launcher to stop all four workers.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',type=Path,default=Path('adaptive_spectral/exact_learned_p/runs/comparison-four'))
    parser.add_argument('--no-color',action='store_true')
    args=parser.parse_args()
    print(snapshot(args.output,not args.no_color),end='')
