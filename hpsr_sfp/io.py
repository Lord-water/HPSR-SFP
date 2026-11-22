"""Explicit schemas, label alignment, and portable JSON artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import LABELS


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def portable(value):
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and np.isinf(value):
        return 'infinity' if value > 0 else '-infinity'
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(portable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def records(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    else:
        rows = read_json(path)
        if isinstance(rows, dict):
            rows = rows['records']
    identifiers = [row['record_id'] for row in rows]
    if not rows or len(set(identifiers)) != len(identifiers):
        raise ValueError('Empty input or duplicate record IDs')
    for row in rows:
        if not isinstance(row['history'], list) or not row['target_body']:
            raise ValueError('Expected dialogue history and nonempty target body')
    return rows


def labels(path, identifiers):
    mapping = read_json(path)
    if set(mapping) != set(identifiers):
        raise ValueError('Annotation IDs must exactly match this feature partition')
    values = {'A': 1., 'N': 0., 'U': -1.}
    for row in mapping.values():
        if set(row) != set(LABELS) or any(value not in values for value in row.values()):
            raise ValueError('Expected all twelve explicit A/N/U judgments')
    return np.array([[values[mapping[rid][label]] for label in LABELS] for rid in identifiers])


def features(path):
    with np.load(path, allow_pickle=False) as cache:
        output = {key: cache[key] for key in cache.files}
    h, u = output['h'], output['u']
    identifiers = output['record_ids'].tolist()
    if output['label_order'].tolist() != LABELS:
        raise ValueError('Candidate order mismatch')
    if h.ndim != 3 or h.shape[:2] != u.shape or u.shape != (len(identifiers), 12):
        raise ValueError('Expected h[N,12,D] and u[N,12]')
    if len(set(identifiers)) != len(identifiers) or not len(identifiers):
        raise ValueError('Empty cache or duplicate IDs')
    if not np.isfinite(h).all() or not np.isfinite(u).all():
        raise ValueError('Non-finite frozen features')
    if 'encoder_id' not in output:
        raise ValueError('Missing encoder identity')
    return output
