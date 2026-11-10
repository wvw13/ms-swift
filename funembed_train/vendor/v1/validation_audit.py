"""Opt-in bounded-run audit of actual inputs, loss scaling and parameter state.

Enabled only by FUNEMBED_E2E_AUDIT=1. Research metadata is removed before
collation; no changes to the representation, sampling or objective.
"""
import copy
import hashlib
import json
from collections import deque
from pathlib import Path
import torch
from common import read_jsonl, dump


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digest(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': tensor_hash(value)}
    if isinstance(value, dict):
        return {str(k): state_digest(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [state_digest(v) for v in value]
    return value


FIELDS = ('messages', 'images', 'positive_messages', 'positive_images', 'negative_messages', 'negative_images')


def row_key(row):
    def normalize(x):
        if isinstance(x, list):
            return [normalize(v) for v in x]
        if isinstance(x, dict):
            if 'path' in x:
                return x['path']
            return {k: normalize(v) for k, v in x.items()}
        return x
    return json.dumps({k: normalize(row.get(k, [])) for k in FIELDS}, sort_keys=True, ensure_ascii=False)


class ValidationAudit:
    def __init__(self, trainer, args, state, model, dataloader, pair_dir):
        self.trainer, self.args, self.model = trainer, args, model
        self.out = Path(args.output_dir)
        self.rank, self.world = args.process_index, args.world_size
        self.start_step = state.global_step
        self.meta = read_jsonl(pair_dir / 'pair_index.jsonl')
        self.raw = read_jsonl(pair_dir / 'pairs.jsonl')
        self.trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.initial_lora = {n: tensor_hash(p) for n, p in self.trainable.items()}
        self.frozen = {n: (p, p._version) for n, p in model.named_parameters() if not p.requires_grad}
        # Exact representative frozen tensors, plus version/gradient checks on all frozen tensors.
        selected = {}
        for group in ('visual', 'language_model', 'merger'):
            for n, (p, _) in self.frozen.items():
                if group in n:
                    selected[n] = tensor_hash(p)
                    break
        self.frozen_hashes = selected
        begin = {'global_step': state.global_step, 'optimizer': state_digest(trainer.optimizer.state_dict()),
                 'scheduler': state_digest(trainer.lr_scheduler.state_dict()),
                 'lora_sha256': self.initial_lora, 'frozen_representative_sha256': selected,
                 'frozen_tensor_count': len(self.frozen), 'trainable_numel': sum(p.numel() for p in self.trainable.values())}
        dump(self.out / f'validation_begin_step{state.global_step}_rank{self.rank}.json', begin)
        dataset = dataloader.dataset
        if not hasattr(dataset, 'encode_func') or not dataset.strict or args.dataloader_num_workers:
            raise RuntimeError('Validation audit requires strict lazy dataset and zero loader workers')
        keys = {row_key(dataset.dataset[i]): i for i in range(len(dataset))}
        if len(keys) != len(dataset):
            raise RuntimeError('Bounded validation requires distinct materialized rows')
        original_encode = dataset.encode_func
        def encode(row, **kwargs):
            index = keys[row_key(row)]
            # Dataset loader representation must equal the source model-facing branch.
            if row_key(row) != row_key(self.raw[index]):
                raise RuntimeError('Dataset preprocessing changed a branch')
            result = original_encode(row, **kwargs)
            result['_funembed_row'] = index
            return result
        dataset.encode_func = encode
        self.pending = deque()
        original_collate = dataloader.collate_fn
        def collate(features):
            features = copy.deepcopy(features)
            rows = [f.pop('_funembed_row') for f in features]
            branches = []
            for f in features:
                branches.extend([f['anchor_input_ids'], f['positive_input_ids']] + f['negative_input_ids'])
            result = original_collate(features)
            observed = [ids[mask.bool()].tolist() for ids, mask in zip(result['input_ids'], result['attention_mask'])]
            if observed != branches:
                raise RuntimeError('Flattened anchor/positive/negative token mapping changed')
            k = len(self.raw[rows[0]]['negative_messages'])
            if result['labels'].tolist() != ([1.] + [0.] * k) * len(rows):
                raise RuntimeError('Flattened candidate labels changed')
            stats = {'input_shape': list(result['input_ids'].shape),
                     'nonpadding_tokens': int(result['attention_mask'].sum()),
                     'padded_tokens': int(result['input_ids'].numel()),
                     'branch_lengths': result['attention_mask'].sum(1).tolist(),
                     'image_count': int(result['image_grid_thw'].shape[0]) if 'image_grid_thw' in result else 0}
            self.pending.append((rows, stats))
            return result
        dataloader.collate_fn = collate
        # ms-swift rebuilds its loader when resuming after skipped batches.
        trainer.data_collator = collate
        original_loss = trainer.compute_loss_func
        self.micro_count = 0
        self.executed = []
        def loss(outputs, labs, **kwargs):
            value = original_loss(outputs, labs, **kwargs)
            if not model.training:
                return value
            rows, stats = self.pending.popleft()
            update = state.global_step + 1
            micro = self.micro_count % args.gradient_accumulation_steps
            first = ((update - 1) * args.gradient_accumulation_steps + micro) * self.world * args.per_device_train_batch_size + self.rank
            expected = [first + j * self.world for j in range(args.per_device_train_batch_size)]
            if rows != expected:
                raise RuntimeError(f'Actual rows {rows} differ from schedule {expected}')
            metas = [self.meta[i] for i in rows]
            if len({(m['task'], m['direction'], m['step']) for m in metas}) != 1 or metas[0]['step'] != update:
                raise RuntimeError('Mixed task/direction within update')
            vectors = outputs['last_hidden_state']
            norms = vectors.float().norm(dim=-1)
            if not torch.isfinite(vectors).all() or not torch.allclose(norms, torch.ones_like(norms), atol=.015):
                raise RuntimeError('Invalid normalized representation')
            import os
            k = len(self.raw[rows[0]]['negative_messages'])
            v = vectors.reshape(len(rows), k + 2, -1)
            logits = (v[:, :1] @ v[:, 1:].transpose(1, 2)).squeeze(1) / float(os.environ['INFONCE_TEMPERATURE'])
            raw_loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(rows), dtype=torch.long, device=v.device))
            divisor = trainer.current_gradient_accumulation_steps
            if not torch.isfinite(value) or not torch.allclose(value, raw_loss / divisor, atol=.0005, rtol=.005):
                raise RuntimeError('InfoNCE row-only formula or accumulation scaling differs')
            entry = {'step': update, 'micro': micro, 'rank': self.rank, 'rows': rows,
                     'task': metas[0]['task'], 'direction': metas[0]['direction'], **stats,
                     'loss': float(raw_loss.detach()), 'scaled_loss': float(value.detach()), 'accumulation_divisor': divisor}
            with (self.out / f'actual_microsteps_rank{self.rank}.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
            self.executed.extend(rows)
            self.micro_count += 1
            return value
        trainer.compute_loss_func = loss

    def check_gradients(self, state):
        gradients = [p.grad for p in self.trainable.values() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or nonfinite LoRA gradients')
        if any(p.grad is not None or p.requires_grad for p, _ in self.frozen.values()):
            raise RuntimeError('Frozen parameter received a gradient')
        with (self.out / f'gradient_audit_rank{self.rank}.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'step': state.global_step + 1, 'finite': True,
                                'nonzero_gradient_tensors': sum(bool(g.ne(0).any()) for g in gradients)}) + '\n')

    def step_end(self, state):
        if any(p._version != version for p, version in self.frozen.values()):
            raise RuntimeError('Frozen parameter version changed')

    def finish(self, state):
        changed = [n for n, p in self.trainable.items() if tensor_hash(p) != self.initial_lora[n]]
        unchanged_frozen = all(tensor_hash(self.frozen[n][0]) == digest for n, digest in self.frozen_hashes.items())
        if not changed or not unchanged_frozen or self.pending:
            raise RuntimeError('Parameter update/frozen state/pending microstep audit failed')
        dump(self.out / f'validation_end_step{state.global_step}_rank{self.rank}.json',
             {'status': 'passed', 'start_step': self.start_step, 'end_step': state.global_step,
              'actual_microsteps': self.micro_count, 'executed_rows': self.executed,
              'changed_lora_tensors': len(changed), 'frozen_versions_unchanged': True,
              'frozen_representative_hashes_unchanged': unchanged_frozen})
