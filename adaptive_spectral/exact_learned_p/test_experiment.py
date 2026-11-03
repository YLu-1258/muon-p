"""Numerical checks for the scientific quantities used by the experiment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data import split_indices
from spectral import spectral_direction, hypergradient, split_parameters
from status import render
from train import TinyLM, arguments, run, loss_pass
from progress import Progress
from comparison import read_metrics


class SpectralTests(unittest.TestCase):
    def test_normalized_derivative_matches_finite_difference(self):
        generator = torch.Generator().manual_seed(5)
        for shape in ((5,5),(3,8),(8,3)):
            matrix = torch.randn(*shape, generator=generator, dtype=torch.float64)
            for p in (0.1,1/3,0.8):
                direction, dp = spectral_direction(matrix,p,driver='default',derivative=True)
                plus,_ = spectral_direction(matrix,p+1e-5,driver='default')
                minus,_ = spectral_direction(matrix,p-1e-5,driver='default')
                torch.testing.assert_close(dp,(plus-minus)/2e-5,atol=1e-8,rtol=1e-6)
                self.assertAlmostEqual(direction.square().mean().item(),1.0,places=10)
                self.assertAlmostEqual((direction*dp).sum().item(),0.0,places=9)

    def test_one_step_hypergradient_matches_meta_loss_difference(self):
        torch.manual_seed(3)
        initial = torch.randn(4,6,dtype=torch.float64)
        momentum = torch.randn_like(initial)
        target = torch.randn_like(initial)
        p,lr = 0.4,0.01
        direction,dp = spectral_direction(momentum,p,driver='default',derivative=True)
        committed = (initial-lr*direction).requires_grad_()
        loss = ((committed-target).sin().square()).mean()
        loss.backward()
        analytic = hypergradient([(committed,dp,lr)]).item()
        def meta_loss(exponent):
            d,_=spectral_direction(momentum,exponent,driver='default')
            return ((initial-lr*d-target).sin().square()).mean().item()
        finite = (meta_loss(p+1e-5)-meta_loss(p-1e-5))/2e-5
        self.assertAlmostEqual(analytic,finite,places=7)

    def test_zero_momentum_and_endpoints(self):
        d,dp = spectral_direction(torch.zeros(3,5),0.3,derivative=True)
        self.assertEqual(d.count_nonzero().item(),0)
        self.assertEqual(dp.count_nonzero().item(),0)
        matrix=torch.randn(4,7,dtype=torch.float64)
        d,_=spectral_direction(matrix,1.0,driver='default')
        torch.testing.assert_close(d,matrix/matrix.square().mean().sqrt())

    def test_data_splits_and_tied_parameter_exclusion(self):
        splits=split_indices(100,15,42)
        flattened=sum(splits.values(),[])
        self.assertEqual(len(set(flattened)),100)
        self.assertEqual(splits,split_indices(100,15,42))
        model=TinyLM()
        model.lm_head.weight=model.embed.weight
        matrices,fallback=split_parameters(model)
        self.assertNotIn(id(model.embed.weight),[id(p) for _,p in matrices])
        self.assertIn(id(model.embed.weight),[id(p) for _,p in fallback])

    def test_dashboard_waiting_partial_and_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertIn('Waiting',render(root,False))
            state=dict(state='failed',phase='stopped',step=1,steps=4,mode='learned',
                       pid=1,started_at=1,updated_at=2,error='test failure')
            (root/'progress.json').write_text(json.dumps(state))
            (root/'metrics.jsonl').write_text('{"event":')
            snapshot=render(root,False)
            self.assertIn('FAILED',snapshot)
            self.assertIn('test failure',snapshot)
            self.assertNotIn('\033',snapshot)

    def test_delayed_updates_reuse_training_backward_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def execute(name,steps,resume=None):
                output=root/name
                cli=['train.py','--smoke','--steps',str(steps),'--save-every','1','--output',str(output)]
                if resume:
                    cli+=['--resume',str(resume)]
                with patch('sys.argv',cli):
                    args=arguments()
                output.mkdir()
                with patch('train.loss_pass',wraps=loss_pass) as calls:
                    run(args,Progress(output,args.steps,args.mode))
                backwards=sum(call.kwargs.get('backward',False) for call in calls.call_args_list)
                checkpoint=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False)
                return output,checkpoint,backwards
            full,reference,backwards=execute('full',4)
            self.assertEqual(backwards,4)  # no additional backward passes for p
            rows=[r for r in read_metrics(full/'metrics.jsonl') if r['event']=='step']
            self.assertEqual([r['p_updates'] for r in rows],[0,1,2,3])
            self.assertIsNone(rows[0]['hypergradient_p'])
            self.assertEqual(rows[0]['p_used'],rows[0]['p_before'])
            self.assertNotEqual(rows[1]['p_used'],rows[1]['p_before'])
            self.assertEqual(len(reference['pending']),2)
            first,_,_=execute('first',2)
            _,resumed,backwards=execute('resumed',4,first/'checkpoint.pt')
            self.assertEqual(backwards,2)
            for name,tensor in reference['model'].items():
                torch.testing.assert_close(tensor,resumed['model'][name],atol=0,rtol=0)
            torch.testing.assert_close(reference['a'],resumed['a'],atol=0,rtol=0)
            torch.testing.assert_close(reference['pending'][0][1],resumed['pending'][0][1],atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
