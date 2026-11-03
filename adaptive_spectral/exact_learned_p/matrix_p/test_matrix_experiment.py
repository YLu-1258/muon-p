"""Per-matrix routing, delayed-gradient and restart verification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from matrix_optimizer import MatrixExponents, ExactMatrixMuon, spectral_direction
from train import TinyLM, arguments, run, loss_pass, Progress, split_parameters
from comparison import read_metrics
from status import render


class MatrixTests(unittest.TestCase):
    def test_opposing_signals_do_not_cancel_and_training_gradients_survive(self):
        left=torch.nn.Parameter(torch.zeros(3,4))
        right=torch.nn.Parameter(torch.zeros(3,4))
        left.grad=torch.ones_like(left)
        right.grad=torch.ones_like(right)
        controller=MatrixExponents([('left',left),('right',right)],1/3,'learned','cpu')
        optimizer=torch.optim.Adam([controller.a],lr=0.05)
        previous=controller.values()
        pending=[(left,torch.ones_like(left),0.01),(right,-torch.ones_like(right),0.01)]
        gradients,_=controller.update(pending,previous,optimizer)
        current=controller.values()
        self.assertLess(gradients['left'],0)
        self.assertGreater(gradients['right'],0)
        self.assertGreater(current['left'],previous['left'])
        self.assertLess(current['right'],previous['right'])
        torch.testing.assert_close(left.grad,torch.ones_like(left))
        torch.testing.assert_close(right.grad,torch.ones_like(right))

    def test_coupled_loss_partial_derivatives_match_finite_differences(self):
        torch.manual_seed(2)
        initial=[torch.randn(4,4,dtype=torch.float64),torch.randn(4,3,dtype=torch.float64)]
        momentum=[torch.randn_like(tensor) for tensor in initial]
        target=torch.randn(4,3,dtype=torch.float64)
        ps=[0.2,0.7]
        lrs=[0.02,0.03]
        named,pending=[],[]
        for i in range(2):
            d,dp=spectral_direction(momentum[i],ps[i],driver='default',derivative=True)
            parameter=torch.nn.Parameter(initial[i]-lrs[i]*d)
            named.append((str(i),parameter))
            pending.append((parameter,dp,lrs[i]))
        ((named[0][1]@named[1][1]-target).sin().square().mean()).backward()
        controller=MatrixExponents(named,1/3,'learned','cpu')
        analytic=controller.hypergradient(pending)
        def loss(exponents):
            ws=[initial[i]-lrs[i]*spectral_direction(momentum[i],exponents[i],driver='default')[0] for i in range(2)]
            return ((ws[0]@ws[1]-target).sin().square().mean()).item()
        for i in range(2):
            plus,minus=ps.copy(),ps.copy()
            plus[i]+=1e-5
            minus[i]-=1e-5
            finite=(loss(plus)-loss(minus))/2e-5
            self.assertAlmostEqual(analytic[i].item(),finite,places=7)

    def test_optimizer_uses_each_matrix_exponent(self):
        torch.manual_seed(3)
        named=[('left',torch.nn.Parameter(torch.randn(3,5))),('right',torch.nn.Parameter(torch.randn(5,3)))]
        before=[parameter.detach().clone() for _,parameter in named]
        ps={'left':0.1,'right':0.8}
        for _,parameter in named:
            parameter.grad=torch.randn_like(parameter)
        expected=[before[i]-0.01*spectral_direction(parameter.grad,ps[name],driver='default')[0]
                  for i,(name,parameter) in enumerate(named)]
        optimizer=ExactMatrixMuon(named,lr=0.01,driver='default')
        pending=optimizer.step(ps,derivative=True)
        self.assertEqual(len(pending),2)
        for (_,parameter),reference in zip(named,expected):
            torch.testing.assert_close(parameter,reference,atol=1e-6,rtol=1e-5)

    def test_delayed_pass_count_and_exact_resume(self):
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
                saved=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False)
                return output,saved,backwards
            full,reference,backwards=execute('full',4)
            self.assertEqual(backwards,4)
            rows=[r for r in read_metrics(full/'metrics.jsonl') if r['event']=='step']
            self.assertEqual([r['p_updates'] for r in rows],[0,1,2,3])
            self.assertEqual(rows[0]['p_before'],rows[0]['p_used'])
            self.assertGreater(rows[-1]['p_std'],0)
            self.assertEqual(len(reference['pending']),2)
            self.assertEqual(reference['matrix_names'],['hidden.weight','hidden_second.weight'])
            first,_,_=execute('first',2)
            _,resumed,backwards=execute('resumed',4,first/'checkpoint.pt')
            self.assertEqual(backwards,2)
            for name,tensor in reference['model'].items():
                torch.testing.assert_close(tensor,resumed['model'][name],atol=0,rtol=0)
            torch.testing.assert_close(reference['a'],resumed['a'],atol=0,rtol=0)
            self.assertEqual(reference['previous_p'],resumed['previous_p'])
            for original,restored in zip(reference['pending'],resumed['pending']):
                self.assertEqual(original[0],restored[0])
                torch.testing.assert_close(original[1],restored[1],atol=0,rtol=0)

    def test_tied_embeddings_excluded_and_dashboard_reads_all_values(self):
        model=TinyLM()
        model.lm_head.weight=model.embed.weight
        matrices,other=split_parameters(model)
        self.assertEqual(len(matrices),2)
        self.assertNotIn(id(model.embed.weight),[id(p) for _,p in matrices])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertIn('Waiting',render(root,False))
            state=dict(state='complete',step=2,steps=2,mode='learned',pid=1,started_at=1,updated_at=2,
                       phase='finished',p={'left':0.2,'right':0.8},p_init=1/3)
            (root/'progress.json').write_text(json.dumps(state))
            (root/'metrics.jsonl').write_text('{"event":')
            text=render(root,False,True)
            self.assertIn('left',text)
            self.assertIn('right',text)
            self.assertNotIn('\033',text)


if __name__=='__main__':
    unittest.main()
