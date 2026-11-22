"""Run stages explicitly; importing the package never launches training."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import LABELS
from .io import features, file_hash, labels, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['adapt', 'features']:
        sub = commands.add_parser(name)
        sub.add_argument('--backbone', required=True, help='Local pretrained model directory')
        sub.add_argument('--inputs', required=True)
        sub.add_argument('--output', required=True)
        sub.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
        if name == 'features':
            sub.add_argument('--adapter', help='Trusted local adapter checkpoint; omit for Base-Raw')
    sub = commands.add_parser('fit')
    for field in ['train-features', 'dev-features', 'train-labels', 'dev-labels', 'folds', 'output']:
        sub.add_argument('--'+field, required=True)
    sub.add_argument('--method', choices=['hpsr', 'raw', 'centered', 'scalar', 'spline'], default='hpsr')
    sub = commands.add_parser('predict')
    for field in ['features', 'readout', 'output']:
        sub.add_argument('--'+field, required=True)
    sub = commands.add_parser('subset')
    for field in ['features', 'labels', 'output']:
        sub.add_argument('--'+field, required=True)
    sub = commands.add_parser('select-records')
    for field in ['inputs', 'labels', 'output']:
        sub.add_argument('--'+field, required=True)
    sub = commands.add_parser('score')
    for field in ['predictions', 'labels', 'output']:
        sub.add_argument('--'+field, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Refusing to overwrite output')
    if args.command == 'select-records':
        from .io import records
        wanted = read_json(args.labels)
        rows = [row for row in records(args.inputs) if row['record_id'] in wanted]
        labels(args.labels, [row['record_id'] for row in rows])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    elif args.command == 'subset':
        cache = features(args.features)
        wanted = read_json(args.labels)
        mask = np.array([rid in wanted for rid in cache['record_ids']])
        labels(args.labels, cache['record_ids'][mask].tolist())
        output = {key: value[mask] if key in ['h', 'u', 'record_ids'] else value for key, value in cache.items()}
        if Path(args.output).suffix != '.npz':
            raise ValueError('Feature output must end in .npz')
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open('xb') as stream:
            np.savez_compressed(stream, **output)
    elif args.command in ['adapt', 'features']:
        from .runner import adapt, extract_features
        (adapt if args.command == 'adapt' else extract_features)(args)
    elif args.command == 'fit':
        from .selection import fit_selected
        train, dev = features(args.train_features), features(args.dev_features)
        y, dy = labels(args.train_labels, train['record_ids'].tolist()), labels(args.dev_labels, dev['record_ids'].tolist())
        foldmap = read_json(args.folds)
        if 'fold_of' in foldmap:
            foldmap = foldmap['fold_of']
        folds = np.array([foldmap[rid] for rid in dev['record_ids']], dtype=int)
        bundle, trace = fit_selected(train, dev, y, dy, folds, args.method,
                                     progress=lambda value: print(json.dumps(value), flush=True))
        bundle['label_order'] = LABELS
        bundle['fitted_train_ids'] = train['record_ids'].tolist()
        bundle['selection_dev_ids'] = dev['record_ids'].tolist()
        bundle['inputs_sha256'] = {field: file_hash(getattr(args, field)) for field in ['train_features', 'dev_features', 'train_labels', 'dev_labels', 'folds']}
        bundle['search_trace'] = trace
        write_json(args.output, bundle)
    elif args.command == 'predict':
        from .selection import predict_bundle, restore_bundle
        cache = features(args.features)
        bundle = restore_bundle(read_json(args.readout))
        if bundle['label_order'] != LABELS:
            raise ValueError('Readout label order mismatch')
        p = predict_bundle(bundle, cache)
        logq = cache['u'].astype(float) - logsumexp(cache['u'].astype(float), axis=1, keepdims=True)
        score = logq + np.log(p.clip(1e-12, 1))
        feasible = p >= .5
        gate = np.where(feasible, score, -np.inf).argmax(1)
        empty = ~feasible.any(1)
        gate[empty] = p[empty].argmax(1)
        output = dict(format='hpsr-sfp-predictions-v1', label_order=LABELS,
                      record_ids=cache['record_ids'].tolist(), acceptability=p,
                      readout_sha256=file_hash(args.readout), feature_sha256=file_hash(args.features),
                      selection={'Usage-Only': logq.argmax(1), 'Rule-PoE': score.argmax(1), 'Rule-BayesGate': gate},
                      bayes_gate_fallback=empty)
        write_json(args.output, output)
    else:
        from .readouts import scores
        result = read_json(args.predictions)
        if result['label_order'] != LABELS:
            raise ValueError('Prediction label order mismatch')
        identifiers = result['record_ids']
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('Duplicate prediction IDs')
        wanted = read_json(args.labels)
        selected = np.array([index for index, rid in enumerate(identifiers) if rid in wanted], dtype=int)
        y = labels(args.labels, [identifiers[index] for index in selected])
        p = np.array(result['acceptability'], float)[selected]
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid probabilities')
        report = scores(p, y)
        report['selection'] = {}
        for name, indices in result['selection'].items():
            values = y[np.arange(len(y)), np.array(indices, dtype=int)[selected]]
            report['selection'][name] = {label: int(np.sum(values == value)) for label, value in [('A', 1), ('N', 0), ('U', -1)]}
        usage = np.array(result['selection']['Usage-Only'])[selected]
        for name in ['Rule-PoE', 'Rule-BayesGate']:
            indices = np.array(result['selection'][name])[selected]
            report['selection'][name]['N_minus_Usage-Only'] = report['selection'][name]['N']-report['selection']['Usage-Only']['N']
            report['selection'][name]['changed_count'] = int(np.sum(indices != usage))
            report['selection'][name]['changed_percent'] = float(100*np.mean(indices != usage))
        report['selection']['Rule-BayesGate']['fallbacks'] = int(np.sum(np.array(result['bayes_gate_fallback'])[selected]))
        write_json(args.output, report)
    print('Saved '+args.output, flush=True)
