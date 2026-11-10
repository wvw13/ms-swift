"""Independently check every exported relationship and global DDP step block."""
import argparse
import json
import os
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from common import ROOT, load_records, labels, read_json, read_jsonl, sha, image_input, assemble_pair, dump

def check_swift_loader(directory):
    """Check official preprocessing preserves schedule order and all branch contents."""
    os.environ['MODELSCOPE_CACHE'] = str(ROOT / 'ms-swift-discriminative/cache/modelscope')
    os.environ['HF_DATASETS_CACHE'] = str(ROOT / 'ms-swift-discriminative/cache/datasets')
    os.environ['HF_HUB_OFFLINE'] = '1'
    from swift.dataset import load_dataset
    dataset, val = load_dataset(str((Path(directory) / 'pairs.jsonl').resolve()),
                                split_dataset_ratio=0, shuffle=False, strict=True, num_proc=1)
    raw = read_jsonl(Path(directory) / 'pairs.jsonl')
    if len(dataset) != len(raw) or val is not None:
        raise ValueError('official preprocessing changed dataset length/split')
    def paths(x):
        if isinstance(x, list):
            return [paths(i) for i in x]
        return x['path'] if isinstance(x, dict) else x
    for i, row in enumerate(raw):
        loaded = dataset[i]
        for field in ('messages', 'positive_messages', 'negative_messages'):
            if loaded[field] != row[field]:
                raise ValueError(f'preprocessor changed row/branch: {i}, {field}')
        for field in ('images', 'positive_images', 'negative_images'):
            if paths(loaded[field]) != row[field]:
                raise ValueError(f'preprocessor changed image order: {i}, {field}')
    return {'status': 'passed', 'rows': len(raw), 'scope': 'Official dataset loader/order only; no tokenizer/model loaded'}

def verify(directory, check_paths=True):
    directory = Path(directory)
    cfg = read_json(directory / 'config.json')
    for name, digest in cfg['artifact_sha256'].items():
        if sha(directory / name) != digest:
            raise ValueError(f'artifact changed: {name}')
    for name, digest in cfg['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError(f'source changed: {name}')
    records = load_records(cfg['base'])
    lookup = {r['input_id']: r for rs in records.values() for r in rs}
    texts = read_json(directory / 'text_snapshot.json')
    schedule = read_jsonl(directory / 'schedule.jsonl')
    nstep = cfg['world_size'] * cfg['micro_batch'] * cfg['grad_accum']
    if len(schedule) != cfg['steps'] or len(schedule) * nstep != cfg['rows']:
        raise ValueError('schedule length mismatch')
    counts, image_paths = Counter(), set()
    with (directory / 'pairs.jsonl').open(encoding='utf-8') as data, (directory / 'pair_index.jsonl').open(encoding='utf-8') as idx:
        for row, (line, metadata) in enumerate(zip_longest(data, idx)):
            if line is None or metadata is None:
                raise ValueError('data/index length mismatch')
            payload, meta = json.loads(line), json.loads(metadata)
            step = row // nstep
            s = schedule[step]
            if s['step'] != step + 1 or s['row_start'] != step * nstep or s['row_count'] != nstep:
                raise ValueError('invalid optimizer step block')
            if (meta['row'], meta['step'], meta['task'], meta['direction']) != (row, step + 1, s['task'], s['direction']):
                raise ValueError('row schedule mismatch')
            q = lookup[meta['query_input']]
            ls = labels(q)
            if not ls or sorted(ls) != meta['query_labels'] or q['patient_anon'] != meta['query_patient']:
                raise ValueError('invalid query supervision')
            task, direction = meta['task'], meta['direction']
            if meta['query_state'] != ('normal' if ls == {21} else 'disease'):
                raise ValueError('normal/disease mismatch')
            images = [q]
            if task == 'T1':
                positive, negatives = meta['positive_class'], meta['negative_classes']
                if direction != q['input_type'] or positive not in ls or ls.intersection(negatives):
                    raise ValueError('T1 false negative or wrong positive/modality')
                if len(negatives) != cfg['negatives'] or len(set(negatives)) != len(negatives):
                    raise ValueError('T1 negative count/uniqueness')
                pos = texts[direction][str(positive)]
                neg = [texts[direction][str(n)] for n in negatives]
            else:
                positive = lookup[meta['positive_input']]
                negatives = [lookup[x] for x in meta['negative_inputs']]
                src, dest = direction.split('-')
                images += [positive] + negatives
                if q['input_type'] != src or any(x['input_type'] != dest for x in images[1:]):
                    raise ValueError('wrong case modality')
                if task not in ('T2', 'T3') or ((src == dest) != (task == 'T2')):
                    raise ValueError('wrong task/direction')
                if labels(positive) != ls:
                    raise ValueError('positive must have identical nonempty labels')
                if any(not labels(n) or not ls.isdisjoint(labels(n)) for n in negatives):
                    raise ValueError('partial overlap/unknown/positive used as negative')
                if len(negatives) != cfg['negatives'] or len({x['patient_key'] for x in images}) != len(images):
                    raise ValueError('candidate patient repeated or wrong negative count')
                if meta['positive_patient'] != positive['patient_anon'] or meta['negative_patients'] != [n['patient_anon'] for n in negatives]:
                    raise ValueError('candidate patient provenance mismatch')
                pos, neg = image_input(positive), [image_input(n) for n in negatives]
            expected = assemble_pair(anchor=image_input(q), positive=pos, negatives=neg)
            if payload != expected:
                raise ValueError('payload differs from source branches (metadata/label leakage or placeholder/order error)')
            for r in images:
                image_paths.update(r['image_paths'])
            counts[f'{task}:{direction}'] += 1
    if sum(counts.values()) != cfg['rows']:
        raise ValueError('wrong row count')
    if check_paths:
        missing = [p for p in image_paths if not Path(p).is_file()]
        if missing:
            raise ValueError(f'{len(missing)} required images missing')
    return {'status': 'passed', 'rows': cfg['rows'], 'steps': cfg['steps'], 'counts': counts,
            'unique_image_paths_checked': len(image_paths) if check_paths else 0,
            'scope': 'CPU relationship/field/order/path audit; not tokenizer/model/DDP execution'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--swift-loader', action='store_true')
    args = parser.parse_args()
    result = verify(args.directory)
    dump(args.directory / 'verification.json', result)
    if args.swift_loader:
        dump(args.directory / 'dataset_loader_check.json', check_swift_loader(args.directory))
    print(json.dumps(result, ensure_ascii=False))
