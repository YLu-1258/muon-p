"""Atomic progress snapshots shared with a standard-library terminal viewer."""
import json
import os
from pathlib import Path
import time


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class Progress:
    def __init__(self, output, steps, mode):
        self.output = Path(output)
        self.state = dict(state='running', phase='starting', step=0, steps=steps,
                          mode=mode, pid=os.getpid(), started_at=time.time())
        self.update()

    def update(self, **kwargs):
        self.state.update(kwargs, updated_at=time.time())
        atomic_json(self.output/'progress.json', self.state)

    def metric(self, row):
        with (self.output/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
