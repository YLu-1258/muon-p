"""Definitions and validation shared by the suite launcher and plotter."""
import json


METHODS = [('muon', 'Muon (exact SVD, p=0)', 'fixed', 0.0, '#268bd2'),
           ('fixed_third', 'Muon-p (exact SVD, p=1/3)', 'fixed', 1/3, '#2aa198'),
           ('learned', 'Global learned p (exact SVD)', 'learned', 1/3, '#d33682'),
           ('matrix_learned', 'Per-matrix learned p (exact SVD)', 'learned', 1/3, '#b58900')]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def read_metrics(path):
    try:
        text = path.read_text()
    except OSError:
        return []
    return [json.loads(line) for line in text[:text.rfind('\n')+1].splitlines() if line]


def verify_comparable(root):
    manifests = [read_json(root/name/'manifest.json') for name,*_ in METHODS]
    if not all(manifests):
        raise ValueError('All three manifests are required')
    keys = ('model','revision','dataset_revision','steps','seed','split_seed','batch_size',
            'microbatch','max_length','lr','fallback_lr','momentum','weight_decay','clip_norm',
            'warmup_steps','eval_every','driver','smoke','val_size','no_gradient_checkpointing')
    reference = manifests[0]
    splits = read_json(root/'muon'/'splits.json')
    for (name,_,mode,p,_), manifest in zip(METHODS, manifests):
        expected = 'delayed_matrix_training_v1' if name == 'matrix_learned' else 'delayed_training_v1'
        if manifest.get('algorithm') != expected:
            raise ValueError(f'{name}: wrong experiment algorithm')
        if manifest['config']['mode'] != mode or manifest['config']['p_init'] != p:
            raise ValueError(f'{name}: wrong method configuration')
        for key in keys:
            if manifest['config'][key] != reference['config'][key]:
                raise ValueError(f'{name}: unmatched {key}')
        if manifest['resolved_model_revision'] != reference['resolved_model_revision']:
            raise ValueError(f'{name}: model revisions differ')
        if manifest['matrices'] != reference['matrices']:
            raise ValueError(f'{name}: eligible matrices differ')
        if read_json(root/name/'splits.json') != splits:
            raise ValueError(f'{name}: dataset splits differ')
    return manifests
