"""Patient-first v1 sampling, materialized in global optimizer-step order."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import (BASE, ROOT, GROUP, TASKS, EXPERIMENTS, TextBank, assemble_pair,
                    dump, image_input, labels, load_records, sha)

def by_patient(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['patient_key']].append(r)
    return dict(groups)

def choose_patient(groups, rng, exclude=()):
    keys = sorted(set(groups) - set(exclude))
    if not keys:
        raise ValueError('no candidate patient')
    key = rng.choice(keys)
    return rng.choice(groups[key])

class Sampler:
    def __init__(self, records, bank, seed, k, normal_fraction, text_kind='definition'):
        self.records, self.bank = records, bank
        self.rng, self.k = random.Random(seed), k
        self.normal_fraction, self.text_kind = normal_fraction, text_kind
        if k < 1 or not 0 <= normal_fraction <= 1:
            raise ValueError('require K >= 1 and normal fraction in [0,1]')
        self.valid = {m: [r for r in rs if labels(r)] for m, rs in records.items()}
        self.index = {}
        for mod, rs in self.valid.items():
            grouped = defaultdict(list)
            for r in rs:
                grouped[labels(r)].append(r)
            self.index[mod] = dict(grouped)
        self.pools, self.candidates, self.coverage = {}, {}, {}
        self.positive_cycle = {}
        self.exposures = {x: Counter() for x in ('patient', 'input', 'positive_class', 'query_class',
                                               'candidate_patient', 'candidate_input', 'negative_class')}

    def candidates_for(self, dest, label_set):
        key = (dest, label_set)
        if key not in self.candidates:
            positive = by_patient(self.index[dest].get(label_set, []))
            negative = by_patient([r for ls, rs in self.index[dest].items()
                                   if ls.isdisjoint(label_set) for r in rs])
            self.candidates[key] = positive, negative
        return self.candidates[key]

    def pool(self, task, direction):
        key = task, direction
        if key in self.pools:
            return self.pools[key]
        src, dest = (direction, None) if task == 'T1' else direction.split('-')
        eligible, stats = [], Counter()
        for q in self.valid[src]:
            state = 'normal' if labels(q) == {21} else 'disease'
            stats[f'{state}_known'] += 1
            if task == 'T1':
                ok = 22 - len(labels(q)) >= self.k
                has_positive = True
            else:
                pos, neg = self.candidates_for(dest, labels(q))
                # Some patients can offer both label states at different visits.
                # Choose positives only when K distinct negative patients remain.
                neg_keys = set(neg) - {q['patient_key']}
                pos_keys = {p for p in pos if p != q['patient_key']}
                has_positive = bool(pos_keys)
                ok = any(len(neg_keys - {p}) >= self.k for p in pos_keys)
            stats[f'{state}_strict_positive'] += int(has_positive)
            if ok:
                eligible.append(q)
                stats[f'{state}_eligible'] += 1
            else:
                stats[f'{state}_excluded_no_positive_or_K_patients'] += 1
        pools = {state: by_patient([r for r in eligible if (labels(r) == {21}) == (state == 'normal')])
                 for state in ('normal', 'disease')}
        for state, rate in [('normal', self.normal_fraction), ('disease', 1 - self.normal_fraction)]:
            if rate > 0 and not pools[state]:
                raise ValueError(f'{key}: no {state} queries; do not silently substitute directions')
        self.coverage[f'{task}:{direction}'] = dict(stats)
        self.pools[key] = pools
        return pools

    def sample(self, task, direction):
        pools = self.pool(task, direction)
        state = 'normal' if self.rng.random() < self.normal_fraction else 'disease'
        q = choose_patient(pools[state], self.rng)
        qlabels = labels(q)
        anchor = image_input(q)
        meta = {'task': task, 'direction': direction, 'query_input': q['input_id'],
                'query_patient': q['patient_anon'], 'query_labels': sorted(qlabels), 'query_state': state}
        self.exposures['patient'][q['patient_anon']] += 1
        self.exposures['input'][q['input_id']] += 1
        self.exposures['query_class'].update(map(str, sorted(qlabels)))
        if task == 'T1':
            queue = self.positive_cycle.setdefault(q['input_id'], [])
            if not queue:
                queue.extend(sorted(qlabels))
                self.rng.shuffle(queue)
            pos = queue.pop()
            neg = self.rng.sample(sorted(set(range(22)) - qlabels), self.k)
            positive = self.bank.branch(direction, pos, self.text_kind)
            negatives = [self.bank.branch(direction, n, self.text_kind) for n in neg]
            meta.update(positive_class=pos, negative_classes=neg)
            self.exposures['positive_class'][f'{direction}:{pos}'] += 1
            self.exposures['negative_class'].update(f'{direction}:{n}' for n in neg)
        else:
            _, dest = direction.split('-')
            pos_pool, neg_pool = self.candidates_for(dest, qlabels)
            neg_keys = set(neg_pool) - {q['patient_key']}
            feasible = {p: rs for p, rs in pos_pool.items()
                        if p != q['patient_key'] and len(neg_keys - {p}) >= self.k}
            pos = choose_patient(feasible, self.rng)
            neg_patients = self.rng.sample(sorted(neg_keys - {pos['patient_key']}), self.k)
            neg = [self.rng.choice(neg_pool[p]) for p in neg_patients]
            positive, negatives = image_input(pos), [image_input(n) for n in neg]
            meta.update(positive_input=pos['input_id'], negative_inputs=[n['input_id'] for n in neg],
                        positive_patient=pos['patient_anon'], negative_patients=[n['patient_anon'] for n in neg])
            for r in [pos] + neg:
                self.exposures['candidate_patient'][r['patient_anon']] += 1
                self.exposures['candidate_input'][r['input_id']] += 1
        return assemble_pair(anchor=anchor, positive=positive, negatives=negatives), meta

def schedule(steps, weights, seed, task_directions=None):
    """Shuffled exact-ratio cycles; independent direction cycles per task."""
    rng = random.Random(seed)
    task_cycle, directions = [], {t: [] for t in weights}
    for step in range(steps):
        if not task_cycle:
            task_cycle = [t for t, weight in weights.items() for _ in range(weight)]
            rng.shuffle(task_cycle)
        task = task_cycle.pop()
        if not directions[task]:
            directions[task] = (task_directions or TASKS)[task].copy()
            rng.shuffle(directions[task])
        yield step, task, directions[task].pop()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    p.add_argument('--cfp-definitions', type=Path, required=True)
    p.add_argument('--oct-definitions', type=Path, required=True)
    p.add_argument('--variant', type=int)
    p.add_argument('--directions-json', help='Task to selected direction list')
    p.add_argument('--text-kind', choices=['definition', 'name', 'name_definition'], default='definition')
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--world-size', type=int, required=True)
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--grad-accum', type=int, required=True)
    p.add_argument('--normal-fraction', type=float, required=True,
                   help='Explicit normal-query mixture; uniform patients within each stratum. Research decision.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--negatives', type=int, default=2)
    p.add_argument('--task-weights', help='Optional JSON integer weights, same task keys as experiment')
    p.add_argument('--t1-modalities', nargs='+', choices=['C', 'O', 'B'], help='Restrict T1 input modalities; default C O B')
    args = p.parse_args()
    task_directions = {k: list(v) for k, v in TASKS.items()}
    if args.directions_json:
        selected=json.loads(args.directions_json)
        for task,ds in selected.items():
            if task not in TASKS or not ds or len(set(ds))!=len(ds) or not set(ds)<=set(TASKS[task]):p.error('invalid directions')
            task_directions[task]=ds
    if args.t1_modalities:
        if len(set(args.t1_modalities)) != len(args.t1_modalities): p.error('duplicate T1 modalities')
        task_directions['T1'] = args.t1_modalities
    if min(args.steps, args.world_size, args.micro_batch, args.grad_accum) < 1:
        p.error('steps/world-size/micro-batch/grad-accum must be positive')
    weights = EXPERIMENTS[args.experiment] if not args.task_weights else json.loads(args.task_weights)
    if (args.experiment != 'CUSTOM' and set(weights) != set(EXPERIMENTS[args.experiment])) or not weights or not set(weights)<=set(TASKS) or any(type(v) is not int or v < 1 for v in weights.values()):
        p.error('weights must be positive integers for the experiment tasks')
    records = load_records(args.base)
    # Retain source fingerprints and exact texts, including any as-yet unreviewed content.
    bank = TextBank(args.cfp_definitions, args.oct_definitions, args.base, args.variant)
    sampler = Sampler(records, bank, args.seed, args.negatives, args.normal_fraction, args.text_kind)
    for task in weights:
        for direction in task_directions[task]:
            sampler.pool(task, direction)
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / 'text_snapshot.json', bank.snapshot(args.text_kind))
    sources = bank.files + [args.base / 'internal_train' / f'{m}.input_records.jsonl' for m in GROUP]
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(schema_version=1, task_weights=weights, instruction='I1', protocol='discriminative-v1',
                  source_sha256={str(x.resolve()): sha(x) for x in sources},
                  text_quality_status='source_preserved_not_clinically_validated')
    counts = Counter()
    batch_rows = args.world_size * args.micro_batch * args.grad_accum
    with (args.output / 'pairs.jsonl').open('w', encoding='utf-8') as data, (args.output / 'pair_index.jsonl').open('w', encoding='utf-8') as index, \
            (args.output / 'schedule.jsonl').open('w', encoding='utf-8') as sched:
        row = 0
        for step, task, direction in schedule(args.steps, weights, args.seed, task_directions):
            sched.write(json.dumps({'step': step + 1, 'task': task, 'direction': direction,
                                    'row_start': row, 'row_count': batch_rows}) + '\n')
            for _ in range(batch_rows):
                payload, meta = sampler.sample(task, direction)
                meta.update(row=row, step=step + 1)
                data.write(json.dumps(payload, ensure_ascii=False) + '\n')
                index.write(json.dumps(meta, ensure_ascii=False) + '\n')
                counts[f'{task}:{direction}:{meta["query_state"]}'] += 1
                row += 1
    config['rows'] = row
    config['artifact_sha256'] = {name: sha(args.output / name) for name in
                                ['pairs.jsonl', 'pair_index.jsonl', 'schedule.jsonl', 'text_snapshot.json']}
    dump(args.output / 'config.json', config)
    dump(args.output / 'sampling_report.json', {'rows': row, 'counts': counts, 'coverage': sampler.coverage,
                                               'exposures': sampler.exposures,
                                               'note': 'Planned exposures only, not completed training updates.'})
    print(json.dumps({'output': str(args.output), 'rows': row, 'counts': counts}, ensure_ascii=False))

if __name__ == '__main__':
    main()
