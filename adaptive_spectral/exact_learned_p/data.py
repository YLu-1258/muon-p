"""Disjoint development splits; no official GSM8K test data is loaded."""
import random

import torch


def split_indices(length, val_size, seed):
    if val_size < 1 or val_size >= length:
        raise ValueError('Require nonempty train/selection splits')
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return dict(validation=indices[:val_size], train=indices[val_size:])


def encode_example(tokenizer, question, answer, max_length):
    prompt = f'Question: {question}\nAnswer: '
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt+answer+tokenizer.eos_token, add_special_tokens=False)[:max_length]
    labels = ([-100]*len(prefix) + ids[len(prefix):])[:len(ids)]
    return dict(input_ids=ids, labels=labels)


def prepare(args, tokenizer=None):
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed+100)
        raw = []
        for _ in range(48):
            ids = torch.randint(1, 32, (12,), generator=generator).tolist()
            raw.append(dict(input_ids=ids, labels=[-100]*4+ids[4:]))
        splits = split_indices(len(raw), 8, args.split_seed)
    else:
        from datasets import load_dataset
        dataset = load_dataset('openai/gsm8k', 'main', split='train', revision=args.dataset_revision)
        splits = split_indices(len(dataset), args.val_size, args.split_seed)
        raw = [encode_example(tokenizer, row['question'], row['answer'], args.max_length) for row in dataset]
    result, dropped = {}, {}
    for split, indices in splits.items():
        kept = [i for i in indices if any(label != -100 for label in raw[i]['labels'][1:])]
        dropped[split] = len(indices)-len(kept)
        splits[split] = kept
        result[split] = [raw[i] for i in kept]
        if not kept:
            raise ValueError(f'No response tokens remain in {split} after truncation')
    return result, dict(indices=splits, dropped_no_response=dropped)


def collate(examples, pad_id, device):
    length = max(len(row['input_ids']) for row in examples)
    data = dict(input_ids=[], labels=[], attention_mask=[])
    for row in examples:
        n = length-len(row['input_ids'])
        data['input_ids'].append(row['input_ids']+[pad_id]*n)
        data['labels'].append(row['labels']+[-100]*n)
        data['attention_mask'].append([1]*len(row['input_ids'])+[0]*n)
    return {key: torch.tensor(value, device=device) for key, value in data.items()}


def sample(dataset, size, generator):
    return [dataset[i] for i in torch.randint(len(dataset), (size,), generator=generator).tolist()]
