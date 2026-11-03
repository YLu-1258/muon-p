"""Run four matched methods concurrently, each isolated to one GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from comparison import METHODS
from progress import atomic_json


def gpu_assignments(requested, visible=None):
    devices = requested if requested is not None else (visible.split(',') if visible is not None else ['0','1','2','3'])
    devices = [device.strip() for device in devices]
    if len(devices) != 4 or len(set(devices)) != 4 or any(not device or device == '-1' for device in devices):
        raise ValueError('Exactly four distinct GPU identifiers are required; use --gpus 0 1 2 3')
    return devices


def stop_children(children):
    active=[child for child in children.values() if child.poll() is None]
    for child in active:
        child.terminate()
    deadline=time.monotonic()+10
    for child in active:
        try:
            child.wait(timeout=max(0.1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog='Other arguments are passed identically to train.py.')
    parser.add_argument('--output',type=Path,default=Path('adaptive_spectral/exact_learned_p/runs/comparison-four'))
    parser.add_argument('--gpus',nargs=4,metavar=('MUON','FIXED','GLOBAL','MATRIX'),help='CUDA identifiers in method order; defaults to CUDA_VISIBLE_DEVICES or 0 1 2 3')
    args,shared=parser.parse_known_args()
    forbidden={'--mode','--p-init','--resume','--output','--device'}
    if any(token.split('=')[0] in forbidden for token in shared):
        parser.error('The suite manages mode, p-init, resume, output and device')
    smoke='--smoke' in shared
    try:
        devices=gpu_assignments(args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'))
    except ValueError as exc:
        parser.error(str(exc))
    import matplotlib
    del matplotlib
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Suite output must be empty; choose a fresh --output')
    script=Path(__file__).resolve().with_name('train.py')
    state=dict(state='preflight',started_at=time.time(),shared_arguments=shared,smoke=smoke,
               methods=[dict(name=name,label=label,state='pending',gpu=gpu)
                        for (name,label,*_),gpu in zip(METHODS,devices)])
    def publish():
        state['updated_at']=time.time()
        atomic_json(args.output/'suite.json',state)
    def env_for(gpu):
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=gpu
        env.setdefault('OMP_NUM_THREADS','1')
        return env
    def cancelled(signum,frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,cancelled)
    publish()
    print(f'Watch: watch --color --no-title -n 2 {sys.executable} {script.with_name("suite_status.py")} {args.output}',flush=True)
    children,logs={},[]
    try:
        # Validate the four isolated CUDA contexts before loading any models.
        if not smoke:
            code=('import json,torch; '
                  'assert torch.cuda.is_available() and torch.cuda.device_count()==1, "Expected one visible GPU"; '
                  'print(json.dumps({"name":torch.cuda.get_device_name(0),'
                  '"uuid":str(getattr(torch.cuda.get_device_properties(0),"uuid","") or ""),'
                  '"capability":torch.cuda.get_device_capability(0)}))')
            seen_uuids=set()
            for gpu,method in zip(devices,state['methods']):
                checked=subprocess.run([sys.executable,'-c',code],env=env_for(gpu),capture_output=True,text=True)
                if checked.returncode:
                    raise RuntimeError(f'GPU {gpu} preflight failed: {checked.stderr.strip()}')
                method['hardware']=checked.stdout.strip()
                identifier=json.loads(checked.stdout).get('uuid')
                if identifier and identifier in seen_uuids:
                    raise ValueError('GPU identifiers resolve to the same physical GPU')
                if identifier:
                    seen_uuids.add(identifier)
                publish()
        state['state']='running'
        for (name,label,mode,p,_),gpu,method in zip(METHODS,devices,state['methods']):
            trainer=script.parent/'matrix_p'/'train.py' if name=='matrix_learned' else script
            command=[sys.executable,str(trainer),*shared,'--mode',mode,'--p-init',str(p),
                     '--device','cuda:0','--output',str(args.output/name)]
            log=(args.output/(name+'.log')).open('w')
            logs.append(log)
            children[name]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env_for(gpu))
            method.update(state='running',pid=children[name].pid)
            publish()
            print(f'GPU {gpu}: {label}; log: {args.output/(name+".log")}',flush=True)
        while True:
            unfinished=False
            for method in state['methods']:
                code=children[method['name']].poll()
                if code is None:
                    unfinished=True
                elif code:
                    method.update(state='failed',returncode=code)
                    publish()
                    raise RuntimeError(f"{method['name']} exited {code}; see {args.output / (method['name']+'.log')}")
                elif method['state']!='complete':
                    method['state']='complete'
                    publish()
            if not unfinished:
                break
            time.sleep(0.5)
        state['state']='plotting'
        publish()
        subprocess.run([sys.executable,str(script.with_name('plot_comparison.py')),str(args.output)],check=True)
        state['state']='complete'
        publish()
    except BaseException as exc:
        stop_children(children)
        for method in state['methods']:
            if method['state']=='running':
                method['state']='cancelled'
        state.update(state='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        publish()
        raise
    finally:
        for log in logs:
            log.close()


if __name__=='__main__':
    main()
