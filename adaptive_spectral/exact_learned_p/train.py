"""Single-GPU exact-SVD pilot with a delayed training-gradient exponent update."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from data import collate, prepare, sample
from progress import Progress, atomic_json
from spectral import ExactMuon, hypergradient, split_parameters


class TinyLM(nn.Module):
    """Offline CPU smoke fixture, not a scientific model baseline."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.hidden = nn.Linear(16, 16)
        self.hidden_second = nn.Linear(16, 16)
        self.lm_head = nn.Linear(16, 32, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, labels, attention_mask=None):
        hidden = self.embed(input_ids).cumsum(dim=1)
        logits = self.lm_head(self.hidden_second(self.hidden(hidden).tanh()).tanh())
        return SimpleNamespace(loss=F.cross_entropy(logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1)))


def response_count(examples):
    return sum(sum(label != -100 for label in row['labels'][1:]) for row in examples)


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def loss_pass(model, examples, microbatch, pad_id, device, backward=False):
    """Response-token-weighted mean, also across accumulation microbatches."""
    total = response_count(examples)
    if total == 0:
        raise ValueError('Batch has no response tokens')
    loss_sum = 0.0
    for offset in range(0, len(examples), microbatch):
        chunk = examples[offset:offset+microbatch]
        weight = response_count(chunk)/total
        with autocast(device):
            loss = model(**collate(chunk, pad_id, device)).loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite model loss')
        if backward:
            (loss*weight).backward()
        loss_sum += loss.detach().float().item()*weight
    return loss_sum, total


def save_checkpoint(path, model, muon, fallback, a, p_optimizer, step, generator, counters, config, pending, previous_p):
    payload = dict(model=model.state_dict(), muon=muon.state_dict(), fallback=fallback.state_dict(),
                   algorithm='delayed_training_v1', a=a.detach(), p_optimizer=p_optimizer.state_dict(), step=step,
                   generator=generator.get_state(), counters=counters,
                   pending=[(muon.names[id(parameter)], dp, lr) for parameter,dp,lr in pending],
                   previous_p=previous_p,
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], config=config)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args, progress):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This pilot is single-process; do not launch with torchrun/DDP')
    device = torch.device('cpu' if args.smoke else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; --smoke runs an offline CPU check')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    progress.update(phase='loading model and tokenizer', device=str(device))
    tokenizer = None
    resolved_model_revision = None
    if args.smoke:
        model, pad_id = TinyLM().to(device), 0
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        if tokenizer.eos_token is None:
            raise ValueError('Tokenizer must define eos_token')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id
        # FP32 parameters avoid losing small updates when p changes. BF16 autocast
        # handles model forward/backward; momentum/SVD/update accumulation are FP32.
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                    torch_dtype=torch.float32, attn_implementation='sdpa').to(device)
        model.config.use_cache = False
        resolved_model_revision = getattr(model.config, '_commit_hash', None)
        if not args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    progress.update(phase='preparing train/selection data')
    datasets, split_info = prepare(args, tokenizer)
    atomic_json(args.output/'splits.json', split_info)
    matrices, others = split_parameters(model)
    if not matrices or not others:
        raise ValueError('Expected both matrix and fallback parameter groups')
    manifest = dict(config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    torch=torch.__version__, cuda=torch.version.cuda,
                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    resolved_model_revision=resolved_model_revision,
                    split_sizes={k:len(v) for k,v in datasets.items()},
                    matrices=[dict(name=n, shape=list(p.shape)) for n,p in matrices],
                    fallback=[n for n,_ in others],
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    data_sampling='uniform with replacement; no epoch claim',
                    algorithm='delayed_training_v1',
                    update_normalization='unit RMS, differentiated analytically')
    atomic_json(args.output/'manifest.json', manifest)
    muon = ExactMuon(matrices, lr=args.lr, momentum=args.momentum,
                     weight_decay=args.weight_decay, driver=args.driver)
    fallback = torch.optim.AdamW([p for _,p in others], lr=args.fallback_lr,
                                betas=(0.9, 0.95), weight_decay=args.weight_decay)
    initial_a = math.log(args.p_init/(1-args.p_init)) if args.mode == 'learned' else 0.0
    a = nn.Parameter(torch.tensor(initial_a, device=device))
    p_optimizer = torch.optim.Adam([a], lr=args.p_lr)
    train_rng = torch.Generator().manual_seed(args.seed+1)
    start_step = 0
    counters = dict(train_tokens=0, p_updates=0, best_validation=None, elapsed_seconds=0.0)
    pending, previous_p = [], None
    if args.resume:
        progress.update(phase='loading checkpoint')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint.get('algorithm') != 'delayed_training_v1':
            raise ValueError('Checkpoint uses the previous held-out meta algorithm; start a fresh delayed run')
        for key, value in checkpoint['config'].items():
            if key not in ('output','resume','steps','save_every','eval_every','device') and manifest['config'][key] != value:
                raise ValueError(f'Resume configuration mismatch for {key}: {value!r} != {manifest["config"][key]!r}')
        model.load_state_dict(checkpoint['model'])
        muon.load_state_dict(checkpoint['muon'])
        fallback.load_state_dict(checkpoint['fallback'])
        with torch.no_grad():
            a.copy_(checkpoint['a'])
        p_optimizer.load_state_dict(checkpoint['p_optimizer'])
        start_step, counters = checkpoint['step'], checkpoint['counters']
        train_rng.set_state(checkpoint['generator'].cpu())
        by_name = dict(matrices)
        pending = [(by_name[name], dp.to(device), lr) for name,dp,lr in checkpoint['pending']]
        previous_p = checkpoint['previous_p']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        random.setstate(checkpoint['python_rng'])
        if checkpoint['cuda_rng'] and device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        del checkpoint
    if start_step >= args.steps:
        raise ValueError('Checkpoint already reached requested steps')

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def evaluate(step):
        progress.update(phase='model-selection validation')
        model.eval()
        with torch.no_grad():
            value, _ = loss_pass(model, datasets['validation'], args.microbatch, pad_id, device)
        model.train()
        old_best = counters['best_validation']
        counters['best_validation'] = value if old_best is None else min(old_best, value)
        progress.update(validation_loss=value, best_validation=counters['best_validation'])
        progress.metric(dict(event='validation', step=step, loss=value, time=time.time(),
                             elapsed_seconds=elapsed_before + time.monotonic()-run_started,
                             train_tokens=counters['train_tokens']))
        return value

    current_p = lambda: a.sigmoid().item() if args.mode == 'learned' else args.p_init
    run_started = time.monotonic()
    elapsed_before = counters['elapsed_seconds']
    progress.update(gpu=manifest['gpu'], p=current_p(), step=start_step,
                    matrix_count=len(matrices), split_sizes=manifest['split_sizes'], **counters)
    model.train()
    evaluate(start_step)
    active_started = time.monotonic()
    for step in range(start_step+1, args.steps+1):
        sync()
        started = time.perf_counter()
        p_before = current_p()
        lr_factor = min(1.0, step/max(1, args.warmup_steps))
        for group in muon.param_groups:
            group['lr'] = args.lr*lr_factor
        for group in fallback.param_groups:
            group['lr'] = args.fallback_lr*lr_factor
        model.zero_grad(set_to_none=True)
        progress.update(phase=f'train forward/backward (step {step})', active_step=step)
        examples = sample(datasets['train'], args.batch_size, train_rng)
        train_loss, tokens = loss_pass(model, examples, args.microbatch, pad_id, device, backward=True)
        # The fresh, unclipped gradient at W_t evaluates the direction derivative
        # saved when W_t was reached. Consume it before clipping/momentum changes.
        grad_p = grad_a = None
        if pending:
            progress.update(phase='delayed p update using current training gradient')
            grad = hypergradient(pending)
            if not torch.isfinite(grad):
                raise FloatingPointError('Nonfinite delayed exponent hypergradient')
            grad_p = grad.item()
            grad_a = grad_p*previous_p*(1-previous_p)
            p_optimizer.zero_grad(set_to_none=True)
            a.grad = torch.tensor(grad_a, device=device)
            p_optimizer.step()
            with torch.no_grad():
                a.clamp_(-8,8)
            counters['p_updates'] += 1
        pending.clear()
        p_used = current_p()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, error_if_nonfinite=True).item()
        sync()
        svd_start = time.perf_counter()
        pending = muon.step(p_used, derivative=args.mode == 'learned',
                            progress=lambda phase: progress.update(phase=phase))
        previous_p = p_used if pending else None
        fallback.step()
        sync()
        optimizer_seconds = time.perf_counter()-svd_start
        # No additional model pass: these derivatives wait for the next normal
        # training backward. That gradient will also update W as usual.
        model.zero_grad(set_to_none=True)
        counters['train_tokens'] += tokens
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
        if args.save_every and (step % args.save_every == 0 or step == args.steps):
            progress.update(phase='saving resumable checkpoint')
            counters['elapsed_seconds'] = elapsed_before + time.monotonic()-run_started
            save_checkpoint(args.output/'checkpoint.pt', model, muon, fallback, a, p_optimizer,
                            step, train_rng, counters, manifest['config'], pending, previous_p)
        sync()
        seconds = time.perf_counter()-started
        counters['elapsed_seconds'] = elapsed_before + time.monotonic()-run_started
        row = dict(event='step', step=step, time=time.time(), train_loss=train_loss,
                   p_before=p_before, p_used=p_used, p_next=current_p(), hypergradient_p=grad_p,
                   hypergradient_a=grad_a, gradient_norm=gradient_norm, lr=args.lr*lr_factor,
                   step_seconds=seconds, optimizer_seconds=optimizer_seconds,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                   **counters)
        progress.metric(row)
        average = (time.monotonic()-active_started)/(step-start_step)
        progress.update(step=step, p=current_p(), train_loss=train_loss,
                        hypergradient_p=grad_p, step_seconds=seconds, optimizer_seconds=optimizer_seconds,
                        peak_allocated_gib=row['peak_allocated_gib'], eta_seconds=average*(args.steps-step),
                        phase='step complete', **counters)
        print(f'{step}/{args.steps} loss={train_loss:.4f} p={p_before:.5f}->{p_used:.5f} SVD+fallback={optimizer_seconds:.2f}s total={seconds:.2f}s', flush=True)
    progress.update(state='complete', phase='finished', eta_seconds=0)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--dataset-revision', default='main')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=['learned','fixed'], default='learned')
    parser.add_argument('--p-init', type=float, default=1/3)
    parser.add_argument('--driver', choices=['gesvd','gesvdj','default'], default='gesvd')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--microbatch', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--fallback-lr', type=float, default=2e-5)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=20)
    parser.add_argument('--p-lr', type=float, default=0.05)
    parser.add_argument('--val-size', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--save-every', type=int, default=100, help='0 disables model checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Offline tiny model + synthetic data on CPU')
    parser.add_argument('--resume', type=Path, help='Trusted checkpoint; use a fresh output directory and matching configuration')
    parser.add_argument('--output', type=Path, default=Path('adaptive_spectral/exact_learned_p/runs/learned'))
    args = parser.parse_args()
    if args.smoke:
        args.steps = min(args.steps, 4)
        args.batch_size = 4
        args.eval_every = 1
        args.warmup_steps = 0
        args.driver = 'default'
    for name in ('steps','batch_size','microbatch','max_length','val_size','eval_every'):
        if getattr(args,name) < 1:
            parser.error(f'{name} must be positive')
    if not 0 <= args.p_init <= 1 or (args.mode == 'learned' and not 0 < args.p_init < 1):
        parser.error('p must be in [0,1]; learned p must start strictly inside')
    if not 0 <= args.momentum < 1 or min(args.lr,args.fallback_lr,args.p_lr,args.clip_norm) <= 0:
        parser.error('Require momentum in [0,1) and positive learning rates/clip norm')
    if min(args.save_every,args.warmup_steps,args.weight_decay) < 0:
        parser.error('save-every, warmup-steps, weight-decay must be nonnegative')
    return args


if __name__ == '__main__':
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise SystemExit('Output directory must be empty; choose a fresh --output')
    progress = Progress(args.output, args.steps, args.mode)
    print(f'Watch: watch --color --no-title -n 2 python {Path(__file__).with_name("status.py")} {args.output}', flush=True)
    try:
        run(args, progress)
    except BaseException as exc:
        progress.update(state='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        phase='stopped', error=str(exc))
        raise
