"""GPU selection rules are checked without requiring CUDA hardware."""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json
import subprocess

from run_comparison import gpu_assignments, main


class GPUSelectionTests(unittest.TestCase):
    def test_defaults_inherited_and_explicit_gpu_order(self):
        self.assertEqual(gpu_assignments(None),['0','1','2','3'])
        self.assertEqual(gpu_assignments(None,'4,5,6,7'),['4','5','6','7'])
        self.assertEqual(gpu_assignments(['7','6','5','4'],'0'),['7','6','5','4'])

    def test_duplicate_missing_or_disabled_devices_rejected(self):
        for devices in (['0'],['0','1','2','2'],['0','1','2',''],['0','1','2','-1']):
            with self.assertRaises(ValueError):
                gpu_assignments(devices)

    def test_all_workers_launch_with_distinct_visible_devices_and_local_cuda_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'suite'
            commands=[]
            def preflight(command,**kwargs):
                gpu=kwargs.get('env',{}).get('CUDA_VISIBLE_DEVICES')
                return subprocess.CompletedProcess(command,0,json.dumps({'uuid':gpu,'name':'mock GPU'}),'')
            def launch(command,**kwargs):
                commands.append((command,kwargs['env']))
                child=MagicMock()
                child.pid=100+len(commands)
                child.poll.return_value=0
                return child
            argv=['run_comparison.py','--gpus','4','5','6','7','--steps','2','--output',str(output)]
            with patch('sys.argv',argv), patch('run_comparison.subprocess.run',side_effect=preflight), \
                 patch('run_comparison.subprocess.Popen',side_effect=launch):
                main()
            self.assertEqual(len(commands),4)
            for i,(command,env) in enumerate(commands):
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],str(i+4))
                self.assertEqual(command[command.index('--device')+1],'cuda:0')
                self.assertEqual(Path(command[command.index('--output')+1]).name,
                                 ['muon','fixed_third','learned','matrix_learned'][i])
            self.assertIn('matrix_p',commands[-1][0][1])
            self.assertEqual(json.loads((output/'suite.json').read_text())['state'],'complete')

    def test_worker_failure_stops_all_remaining_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'suite'
            children=[]
            def launch(command,**kwargs):
                child=MagicMock()
                child.pid=100+len(children)
                child.poll.return_value=1 if not children else None
                child.terminate.side_effect=lambda: setattr(child.poll,'return_value',-15)
                children.append(child)
                return child
            argv=['run_comparison.py','--smoke','--gpus','0','1','2','3','--output',str(output)]
            with patch('sys.argv',argv), patch('run_comparison.subprocess.Popen',side_effect=launch):
                with self.assertRaises(RuntimeError):
                    main()
            self.assertEqual(len(children),4)
            for child in children[1:]:
                child.terminate.assert_called_once()
                child.wait.assert_called_once()
            state=json.loads((output/'suite.json').read_text())
            self.assertEqual(state['state'],'failed')
            self.assertEqual([method['state'] for method in state['methods']],
                             ['failed','cancelled','cancelled','cancelled'])


if __name__=='__main__':
    unittest.main()
