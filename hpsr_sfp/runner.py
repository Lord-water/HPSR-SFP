"""Explicit adaptation and cache extraction using the recorded encoder code."""
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from transformers import AutoTokenizer

from . import LABELS
from .encoder import UsageEncoder, batch, seed, tokenize, usage_loss
from .io import file_hash, records, write_json


def encoder_identity(backbone, adapter=None):
    path = Path(backbone)
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']
    names += sorted(p.name for p in path.glob('*.safetensors'))
    names += sorted(p.name for p in path.glob('*.bin'))
    if not any(path.glob('*.safetensors')) and not any(path.glob('*.bin')):
        raise ValueError('Local backbone weights are missing')
    digest = hashlib.sha256()
    for name in names:
        if (path/name).is_file():
            digest.update(name.encode())
            digest.update(file_hash(path/name).encode())
    digest.update((file_hash(adapter) if adapter else 'unadapted').encode())
    return digest.hexdigest()


def load_encoder(args):
    torch.set_num_threads(2)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu for extraction only')
        torch.cuda.set_per_process_memory_fraction(.32)
        if torch.cuda.mem_get_info()[0] < 6 * 2**30:
            raise RuntimeError('At least 6 GiB of free GPU memory required at launch')
    tok = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    adapter = getattr(args, 'adapter', None)
    seed(42)
    model = UsageEncoder(args.backbone, args.device, adapter=args.command == 'adapt' or bool(adapter))
    if adapter:
        checkpoint = torch.load(adapter, map_location='cpu', weights_only=True)
        if not checkpoint.get('complete', False):
            raise ValueError('Only a completed usage adapter can be used for extraction')
        if checkpoint['label_order'] != LABELS or checkpoint['max_length'] != 256:
            raise ValueError('Adapter configuration mismatch')
        if checkpoint['backbone_id'] != encoder_identity(args.backbone):
            raise ValueError('Adapter/backbone identity mismatch')
        model.restore(checkpoint['state'])
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    return model, tok, pad


def gpu_boundary(device):
    if device == 'cuda' and torch.cuda.mem_get_info()[0] < 2 * 2**30:
        raise RuntimeError('GPU free memory fell below the 2 GiB safety margin')


def adapt(args):
    if args.device != 'cuda':
        raise ValueError('Full adaptation uses CUDA; CPU is reserved for feature extraction')
    rows = records(args.inputs)
    if any(r.get('split', 'train') != 'train' or r['observed_particle'] not in LABELS for r in rows):
        raise ValueError('Adaptation requires training-only observed choices')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    backbone_id = encoder_identity(args.backbone)
    model, tok, pad = load_encoder(args)
    tokens = [tokenize(tok, row, 256) for row in rows]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    steps = math.ceil(len(rows)/16)
    warm = max(1, round(.05 * steps))
    def rate(step):
        return (step+1)/warm if step < warm else max(.1, .5*(1+math.cos(math.pi*(step-warm)/max(steps-warm, 1))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    order = np.random.default_rng(42).permutation(len(rows))
    for start in range(0, len(order), 16):
        gpu_boundary(args.device)
        tick = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = order[start:start+16]
        total = 0.
        for index in indices:
            logits, _ = model(*batch(tokens[int(index)], pad, args.device))
            target = torch.tensor([LABELS.index(rows[int(index)]['observed_particle'])], device=args.device)
            loss = usage_loss(logits, target)/len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite observed-choice loss')
            loss.backward()
            total += float(loss.detach())
        gradient = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.))
        if not math.isfinite(gradient):
            raise FloatingPointError('Non-finite gradient')
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        done = min(start+16, len(rows))
        print(json.dumps(dict(stage='adapt', completed=done, total=len(rows), loss=total)), flush=True)
        if (start//16+1) % 128 == 0 or done == len(rows):
            artifact = dict(state=model.state(), label_order=LABELS, max_length=256, backbone_id=backbone_id,
                            seed=42, completed_contexts=done, total_contexts=len(rows),
                            input_sha256=file_hash(args.inputs), complete=done == len(rows))
            temporary = output/'latest.tmp'
            torch.save(artifact, temporary)
            temporary.replace(output/'latest.pt')
        time.sleep((time.monotonic()-tick)*.35)
    (output/'latest.pt').replace(output/'adapter.pt')
    write_json(output/'config.json', dict(backbone_id=backbone_id, epochs=1, seed=42, max_length=256,
               effective_batch_size=16, lr=1e-4, weight_decay=.01, lora_rank=8, lora_alpha=16, lora_dropout=.05))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def extract_features(args):
    output = Path(args.output)
    if output.suffix != '.npz':
        raise ValueError('Feature output must end in .npz')
    rows = records(args.inputs)
    identity = encoder_identity(args.backbone, args.adapter)
    model, tok, pad = load_encoder(args)
    model.eval()
    h, u = [], []
    with torch.no_grad():
        for index, row in enumerate(rows):
            gpu_boundary(args.device)
            utility, feature = model(*batch(tokenize(tok, row, 256), pad, args.device))
            h.append(feature.cpu().float().numpy())
            u.append(utility.cpu().float().numpy())
            print(json.dumps(dict(stage='features', completed=index+1, total=len(rows))), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, h=np.stack(h), u=np.stack(u),
                            record_ids=np.array([r['record_id'] for r in rows]), label_order=np.array(LABELS),
                            encoder_id=np.array(identity), input_sha256=np.array(file_hash(args.inputs)))
