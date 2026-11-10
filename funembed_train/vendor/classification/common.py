"""Shared protocol, data and metrics; no model or CUDA initialization."""
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

CODE = Path(__file__).resolve().parent
BASE = Path(os.environ['FUNEMBED_CLASSIFICATION_WORK'])
ROOT = BASE.parent
SPLITS = ('internal_train', 'val', 'internal_test', 'external_test')
EXPECTED = dict(zip(SPLITS, (3784, 456, 1038, 984)))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path=None):
    return read(path or BASE / 'configs/common.json')


def rows_for(split):
    return [json.loads(line) for line in (BASE / 'data/manifests' / (split + '.jsonl')).open(encoding='utf-8')]


def paths_for(row, modality):
    return row['images'][modality]


def prepare(cfg):
    """Reuse the audited manifest; preserve old LP's common complete-case cohort."""
    source = ROOT / cfg['manifest']
    classes = read(ROOT / cfg['class_map'])
    assert [c['id'] for c in classes] == list(range(22))
    buckets = {s: [] for s in SPLITS}
    patients = {s: set() for s in SPLITS}
    for line in source.open(encoding='utf-8'):
        row = json.loads(line)
        if not row['prior_eval_eligible']:
            continue
        split = row['split']
        assert row['availability']['cfp_oct_center6'] and not row['image_header_errors']
        assert row['labels']['combined']['state'] in ('normal', 'disease')
        if row['center'] == 'shandong':
            assert split == 'external_test'
        paths = row['images']['center6_paths']
        assert [int(Path(p).stem[-3:]) for p in paths] == list(range(7, 13))
        cfp = [row['images']['cfp_path']]
        assert all(Path(p).is_file() for p in cfp + paths)
        y = [int(i in row['labels']['combined']['ids']) for i in range(22)]
        assert not (y[21] and sum(y) > 1)
        buckets[split].append(dict(record_id=row['record_id'], case_id=row['case_id'],
            patient_anon=row['patient_anon'], center=row['center'], split=split,
            images={'C': cfp, 'O': paths, 'B': cfp + paths}, labels=y))
        patients[split].add(row['patient_key'])
    assert {s: len(v) for s, v in buckets.items()} == EXPECTED
    for i, s in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            assert not patients[s] & patients[other], 'patient split overlap'
    directory = BASE / 'data/manifests'
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in buckets.items():
        path = directory / (split + '.jsonl')
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        if path.exists() and path.read_text(encoding='utf-8') != text:
            raise RuntimeError('Existing cohort differs; archive it before changing protocol: ' + str(path))
        path.write_text(text)
        hashes[split] = digest(path)
    write(directory / 'summary.json', dict(source=str(source), source_sha256=digest(source),
          label_source='combined', counts=EXPECTED, patient_counts={s: len(p) for s, p in patients.items()},
          class_map=classes, manifest_sha256=hashes, image_paths_checked=True))
    return EXPECTED


def protocol(cfg, size, modality):
    directory = Path(cfg['models'][size])
    weights = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(directory.glob('*.safetensors'))]
    if not weights:
        raise FileNotFoundError(directory)
    return dict(protocol=cfg['protocol'], model=str(directory), weights_stat=weights,
        implementation_sha256={name: digest(CODE/name) for name in ('common.py','model.py')},
        packages={name: importlib.metadata.version(name) for name in ('torch','transformers','qwen-vl-utils')},
        processor_sha256={name: digest(directory/name) for name in
                          ('preprocessor_config.json','tokenizer_config.json','special_tokens_map.json')},
        model_config_sha256=digest(directory / 'config.json'),
        official_script_sha256=digest(directory / 'scripts/qwen3_vl_embedding.py'),
        tokenizer_sha256=digest(directory / 'tokenizer.json'),
        template_sha256=digest(directory / 'chat_template.jinja'),
        modality=modality, instruction=cfg['instructions'][modality],
        preprocess={k: cfg[k] for k in ('min_pixels', 'max_pixels', 'max_length', 'attention', 'patch_embed_fp32')},
        manifests=read(BASE / 'data/manifests/summary.json')['manifest_sha256'])


def check_manifests():
    summary = read(BASE / 'data/manifests/summary.json')
    for split, value in summary['manifest_sha256'].items():
        assert digest(BASE / 'data/manifests' / (split + '.jsonl')) == value, split


def select_threshold(y, probability):
    """Exact micro-F1 search over distinct probabilities; tie -> higher threshold."""
    y, p = np.asarray(y).ravel(), np.asarray(probability).ravel()
    order = np.argsort(-p, kind='stable')
    ps, ys = p[order], y[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    tp = np.cumsum(ys)[ends]
    scores = 2 * tp / np.maximum(ends + 1 + y.sum(), 1)
    if not y.sum():
        return 1.0
    return float(ps[ends[np.argmax(scores)]])


def metrics(y, probability, threshold=None):
    y, p = np.asarray(y), np.asarray(probability)
    assert y.shape == p.shape and y.shape[1] == 22 and np.isfinite(p).all()
    per_class, aps, aucs = [], [], []
    for c in range(22):
        positive = int(y[:, c].sum())
        ap = float(average_precision_score(y[:, c], p[:, c])) if positive else None
        auc = float(roc_auc_score(y[:, c], p[:, c])) if 0 < positive < len(y) else None
        if ap is not None: aps.append(ap)
        if auc is not None: aucs.append(auc)
        per_class.append(dict(id=c, positives=positive, AP=ap, AUC=auc))
    result = dict(n=len(y), mAP=float(np.mean(aps)) if aps else None,
                  macro_auc=float(np.mean(aucs)) if aucs else None,
                  valid_AP_classes=len(aps), valid_AUC_classes=len(aucs), per_class=per_class)
    if threshold is not None:
        pred = p >= threshold
        result.update(threshold=float(threshold), micro_f1=float(f1_score(y, pred, average='micro', zero_division=0)),
                      macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)))
    return result
