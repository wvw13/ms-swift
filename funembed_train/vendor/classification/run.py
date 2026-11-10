"""CLI: prepare, extract, smoke, train, evaluate. torchrun supported for FT/extract."""
import argparse
import importlib.metadata
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from common import (BASE, SPLITS, check_manifests, digest, load_config, metrics, prepare,
                    protocol, read, rows_for, save_torch, seed_all, select_threshold, stable, write)
from model import Cases, Classifier, Collator, to_device


def runtime():
    world, rank, local = (int(os.environ.get(k, d)) for k, d in
                           [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run on the GPU server outside a restricted sandbox.')
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    return torch.device('cuda', local), rank, world


def barrier():
    if dist.is_initialized(): dist.barrier()


def same_protocol(left, right):
    """Compare JSON semantics: serialization turns Python tuples into lists."""
    return stable(left) == stable(right)


def gather_rows(parts, n):
    """Merge disjoint validation shards, requiring exactly one prediction per case."""
    indices = np.concatenate([p[0] for p in parts])
    values = np.concatenate([p[1] for p in parts])
    assert len(indices) == n and np.array_equal(np.sort(indices), np.arange(n)), 'missing/duplicate predictions'
    return values[np.argsort(indices)]


def image_loader(rows, cfg, size, modality, *, rank=0, world=1, train=False, seed=0):
    ds = Cases(rows, modality)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=seed) if train else None
    dataset = ds if train else Subset(ds, list(range(rank, len(ds), world)))
    workers = cfg['workers']
    return DataLoader(dataset, batch_size=cfg['ft']['micro_batch'] if train else cfg['extract_batch'],
        sampler=sampler, num_workers=workers, collate_fn=Collator(cfg, size, modality), pin_memory=True,
        multiprocessing_context='spawn' if workers else None, persistent_workers=bool(workers)), sampler


@torch.no_grad()
def image_predict(model, loader, device, n, *, embedding=False):
    model.eval()
    ids, values = [], []
    for batch in loader:
        x = to_device(batch, device)
        value = model.embedding(x) if embedding else model(x).sigmoid()
        ids.append(batch['indices'].numpy())
        values.append(value.float().cpu().numpy())
    width = model.dimension if embedding else 22
    part = (np.concatenate(ids) if ids else np.empty(0, dtype=int),
            np.concatenate(values) if values else np.empty((0, width), dtype=np.float32))
    if dist.is_initialized():
        parts = [None] * dist.get_world_size()
        dist.all_gather_object(parts, part)
    else:
        parts = [part]
    return gather_rows(parts, n)


def cache_dir(p):
    return BASE / 'baseline/cache' / stable(p)[:20]


def load_cache(p, split):
    directory = cache_dir(p)
    assert same_protocol(read(directory / 'protocol.json'), p), 'feature protocol mismatch'
    meta = read(directory / (split + '.json'))
    path = directory / (split + '.npz')
    assert digest(path) == meta['sha256'], 'feature cache changed'
    rows = rows_for(split)
    with np.load(path, allow_pickle=False) as data:
        assert data['record_ids'].tolist() == [r['record_id'] for r in rows]
        assert np.array_equal(data['y'], np.asarray([r['labels'] for r in rows]))
        return torch.from_numpy(data['x'].copy()), torch.from_numpy(data['y'].copy()).float()


def extract(args, cfg, device, rank, world, p):
    directory = cache_dir(p)
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'protocol.json').exists(): assert same_protocol(read(directory / 'protocol.json'), p)
        write(directory / 'protocol.json', p)
    barrier()
    model = None
    for split in SPLITS:
        if (directory / (split + '.json')).exists():
            load_cache(p, split)
            continue
        if model is None: model = Classifier(cfg, args.model, 'lp', device)
        rows = rows_for(split)
        loader, _ = image_loader(rows, cfg, args.model, args.modality, rank=rank, world=world)
        t = time.monotonic()
        x = image_predict(model, loader, device, len(rows), embedding=True)
        assert np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4)
        if rank == 0:
            path = directory / (split + '.npz')
            temp = directory / (split + '.tmp.npz')
            np.savez(temp, x=x, y=np.asarray([r['labels'] for r in rows], dtype=np.float32),
                     record_ids=np.asarray([r['record_id'] for r in rows]))
            os.replace(temp, path)
            write(directory / (split + '.json'), dict(sha256=digest(path), n=len(rows), seconds=time.monotonic()-t))
            print(f'cached {split}: {len(rows)} cases -> {path}', flush=True)
        del loader
        barrier()


def run_signature(args, cfg, p):
    return dict(protocol=p, method=args.method, seed=args.seed, optimizer=cfg[args.method],
                code_sha256={f.name: digest(f) for f in sorted(Path(__file__).parent.glob('*.py'))})


def init_run(path, signature, cfg, rank):
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)
        if (path/'signature.json').exists():
            assert same_protocol(read(path/'signature.json'), signature), 'Run configuration/code changed: use a new --run directory'
        else:
            if any(path.iterdir()): raise RuntimeError('Nonempty directory without run signature')
            write(path/'signature.json', signature)
            write(path/'config.json', cfg)
            write(path/'data_summary.json', read(BASE/'data/manifests/summary.json'))
        for name in ('logs', 'checkpoints', 'metrics', 'predictions'): (path/name).mkdir(exist_ok=True)
    barrier()


def checkpoint_state(model, method):
    return model.trainable_state() if method == 'ft' else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_model(model, state, method):
    if method == 'ft': model.load_trainable(state)
    else: model.load_state_dict(state)


@torch.no_grad()
def lp_predict(head, features, device):
    head.eval()
    return torch.cat([head(x.to(device)).sigmoid().cpu() for x in features.split(512)]).numpy()


def train(args, cfg, device, rank, world, p):
    if args.method == 'lp' and world != 1: raise ValueError('LP uses a cached-feature single-GPU head; parallelize independent runs')
    path = Path(args.run).resolve()
    signature = run_signature(args, cfg, p)
    init_run(path, signature, cfg, rank)
    if (path/'train_complete.json').exists():
        if rank == 0: print('Training already complete:', path, flush=True)
        return
    if (path/'checkpoints/last.pt').exists() and not args.resume: raise ValueError('Use --resume or a new run directory')
    if rank == 0:
        write(path/'status.json', dict(state='loading_model_or_features', pid=os.getpid()))
        print(f'Loading {args.model}/{args.modality}/{args.method}, world_size={world}', flush=True)
    seed_all(args.seed)
    options = cfg[args.method]
    if args.method == 'lp':
        x, y = load_cache(p, 'internal_train')
        vx, vy = load_cache(p, 'val')
        model = nn.Linear(x.shape[1], 22).to(device)
        loader = DataLoader(TensorDataset(x, y), batch_size=options['batch_size'], shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], weight_decay=options['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=options['lr_patience'])
        sampler, accumulation, val_loader = None, 1, None
    else:
        rows, val = rows_for('internal_train'), rows_for('val')
        loader, sampler = image_loader(rows, cfg, args.model, args.modality, rank=rank, world=world, train=True, seed=args.seed)
        val_loader, _ = image_loader(val, cfg, args.model, args.modality, rank=rank, world=world)
        vy = torch.tensor([r['labels'] for r in val])
        model = Classifier(cfg, args.model, 'ft', device)
        seed_all(args.seed)
        model.head.reset_parameters()
        adapters = [v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('head.')]
        optimizer = torch.optim.AdamW([{'params': adapters, 'lr': options['lr']},
                                      {'params': model.head.parameters(), 'lr': options['head_lr']}], weight_decay=options['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=1)
        divisor = options['micro_batch'] * world
        if options['effective_batch'] % divisor: raise ValueError('effective_batch must be divisible by micro_batch * world_size')
        accumulation = options['effective_batch'] // divisor
        if accumulation < 1: raise ValueError('effective_batch too small')
    training_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
    if rank == 0:
        write(path/'parameter_summary.json', dict(trainable=sum(v.numel() for v in model.parameters() if v.requires_grad),
              total=sum(v.numel() for v in model.parameters()), trainable_names=[n for n,v in model.named_parameters() if v.requires_grad],
              world_size=world, gradient_accumulation=accumulation,
              sampler_padding_per_epoch=(len(loader.sampler)*world-len(rows)) if sampler is not None else 0))
        write(path/'environment.json', dict(packages={name:importlib.metadata.version(name) for name in
              ('torch','transformers','peft','qwen-vl-utils','tensorboard','scikit-learn')},
              cuda_version=torch.version.cuda, world_size=world,
              device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu-synthetic-test'))
    start, best, bad, step = 0, -1., 0, 0
    if args.resume and (path/'checkpoints/last.pt').exists():
        ck = torch.load(path/'checkpoints/last.pt', map_location=device, weights_only=False)
        assert same_protocol(ck['signature'], signature)
        assert ck['world_size'] == world, 'Resume with the original world size'
        restore_model(model, ck['model'], args.method)
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        start, best, bad, step = ck['epoch']+1, ck['best'], ck['bad'], ck['step']
    writer = SummaryWriter(str(path/'logs/tensorboard'), purge_step=step+1) if rank == 0 else None
    epoch_writer = SummaryWriter(str(path/'logs/tensorboard_epoch'), purge_step=start+1) if rank == 0 else None
    began = time.monotonic()
    epoch = start - 1
    try:
        for epoch in range(start, options['epochs']):
            if bad >= options['patience']: break
            # Epoch-boundary resume is reproducible for the same world size/config.
            seed_all(args.seed + 10000 * epoch + rank)
            if sampler is not None: sampler.set_epoch(epoch)
            training_model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss, total_n, epoch_start = 0., 0, time.monotonic()
            for batch_index, batch in enumerate(loader):
                window_start = (batch_index // accumulation) * accumulation
                if args.method == 'ft':
                    micro = options['micro_batch']
                    window_samples = min(accumulation * micro, len(loader.sampler) - window_start * micro)
                    labels = batch['y'].to(device)
                else:
                    labels = batch[1].to(device); window_samples = len(labels)
                boundary = (batch_index+1) % accumulation == 0 or batch_index+1 == len(loader)
                sync = training_model.no_sync() if world > 1 and not boundary else nullcontext()
                with sync:
                    logits = training_model(to_device(batch, device)) if args.method == 'ft' else model(batch[0].to(device))
                    per_case = nn.functional.binary_cross_entropy_with_logits(logits.float(), labels, reduction='none').mean(1)
                    loss = per_case.sum() / window_samples
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite train loss')
                    loss.backward()
                total_loss += float(per_case.detach().sum()); total_n += len(labels)
                if rank == 0 and (batch_index == 0 or (batch_index+1) % 10 == 0):
                    write(path/'status.json', dict(state='training', epoch=epoch+1, step=step,
                          micro_batch=batch_index+1, total_micro_batches=len(loader), pid=os.getpid(),
                          elapsed_seconds=time.monotonic()-began,
                          peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
                    if args.method == 'ft':
                        print(f'progress epoch={epoch+1} micro_batch={batch_index+1}/{len(loader)} optimizer_step={step}', flush=True)
                if boundary:
                    norm = nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], options.get('grad_clip', 1.0), error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                    if rank == 0:
                        writer.add_scalar('train/loss_microbatch', float(per_case.detach().mean()), step)
                        writer.add_scalar('train/grad_norm', float(norm), step)
                        if step % 10 == 0:
                            write(path/'status.json', dict(state='training', epoch=epoch+1, step=step,
                                  batch=batch_index+1, batches=len(loader), pid=os.getpid(),
                                  elapsed_seconds=time.monotonic()-began,
                                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
                            print(f'epoch={epoch+1} step={step} batch={batch_index+1}/{len(loader)} loss={float(per_case.mean()):.5f}', flush=True)
            if rank == 0: write(path/'status.json', dict(state='validating', epoch=epoch+1, step=step, pid=os.getpid()))
            pred = image_predict(model, val_loader, device, len(vy)) if args.method == 'ft' else lp_predict(model, vx, device)
            score = metrics(vy.numpy(), pred)
            scheduler.step(score['mAP'])
            improved = score['mAP'] > best
            if improved: best, bad = score['mAP'], 0
            else: bad += 1
            totals = torch.tensor([total_loss, total_n], device=device, dtype=torch.float64)
            if world > 1: dist.all_reduce(totals)
            if rank == 0:
                state = checkpoint_state(model, args.method)
                checkpoint = dict(model=state, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                  epoch=epoch, best=best, bad=bad, step=step, signature=signature, world_size=world)
                save_torch(path/'checkpoints/last.pt', checkpoint)
                if improved:
                    threshold = select_threshold(vy.numpy(), pred)
                    save_torch(path/'checkpoints/best.pt', dict(model=state, epoch=epoch, signature=signature, threshold=threshold))
                    write(path/'metrics/val.json', metrics(vy.numpy(), pred, threshold))
                record = dict(epoch=epoch+1, step=step, train_loss=float(totals[0]/totals[1]),
                              val_mAP=score['mAP'], val_macro_auc=score['macro_auc'], best_mAP=best,
                              seconds=time.monotonic()-epoch_start,
                              peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                write(path/'logs'/f'epoch_{epoch+1:04d}.json', record)
                for k in ('val_mAP', 'val_macro_auc', 'train_loss'): epoch_writer.add_scalar('epoch/'+k, record[k], epoch+1)
                epoch_writer.flush()
                writer.flush(); print(json.dumps(record), flush=True)
            barrier()
        if rank == 0:
            write(path/'train_complete.json', dict(best_val_mAP=best, seconds_this_invocation=time.monotonic()-began,
                  stopped_early=bad>=options['patience'], signature=signature))
            write(path/'status.json', dict(state='training_complete', best_val_mAP=best))
    except BaseException as error:
        if rank == 0: write(path/'status.json', dict(state='interrupted_or_failed', error=repr(error), epoch=epoch+1, step=step))
        raise
    finally:
        if writer: writer.close()
        if epoch_writer: epoch_writer.close()


def evaluate(args, cfg, device, rank, world, p):
    path = Path(args.run).resolve()
    signature = read(path/'signature.json')
    assert same_protocol(signature, run_signature(args, cfg, p)), 'evaluation config/code differs from trained run'
    if not (path/'train_complete.json').exists(): raise RuntimeError('Finish training before final test evaluation')
    ck = torch.load(path/'checkpoints/best.pt', map_location=device, weights_only=False)
    assert same_protocol(ck['signature'], signature)
    if args.method == 'ft': model = Classifier(cfg, args.model, 'ft', device)
    else:
        x, _ = load_cache(p, 'val'); model = nn.Linear(x.shape[1], 22).to(device)
    restore_model(model, ck['model'], args.method)
    for split in ('internal_test', 'external_test'):
        rows = rows_for(split)
        if args.method == 'ft':
            loader, _ = image_loader(rows, cfg, args.model, args.modality, rank=rank, world=world)
            pred = image_predict(model, loader, device, len(rows))
            y = np.asarray([r['labels'] for r in rows])
        else:
            x, labels = load_cache(p, split); y = labels.numpy(); pred = lp_predict(model, x, device)
        if rank == 0:
            write(path/'metrics'/f'{split}.json', metrics(y, pred, ck['threshold']))
            np.savez_compressed(path/'predictions'/f'{split}.npz', probability=pred, y=y,
                                record_ids=np.asarray([r['record_id'] for r in rows]))
        barrier()
    if rank == 0: write(path/'complete.json', dict(signature=signature, best_epoch=ck['epoch']+1, threshold=ck['threshold']))


def smoke(args, cfg, device, rank, world, p):
    """Actual GPU check, only when explicitly launched; no persistent training run."""
    seed_all(123)
    rows = rows_for('internal_train')[:max(4, world)]
    loader, _ = image_loader(rows, cfg, args.model, args.modality, rank=rank, world=world, train=True)
    model = Classifier(cfg, args.model, 'ft', device)
    trained = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=cfg['ft']['lr'])
    wrapper = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    frozen_versions = {n: v._version for n,v in model.named_parameters() if not v.requires_grad}
    t = time.monotonic(); losses, lengths = [], []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(wrapper(to_device(batch, device)), batch['y'].to(device))
        assert torch.isfinite(loss)
        loss.backward()
        adapter_grads = [v.grad for n,v in model.named_parameters() if 'lora_' in n and v.grad is not None]
        assert adapter_grads and all(torch.isfinite(g).all() for g in adapter_grads)
        assert any(g.abs().sum() > 0 for g in adapter_grads), 'LoRA receives no gradient'
        assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0
        optimizer.step(); losses.append(float(loss)); lengths.extend(batch['lengths'].tolist())
        if len(losses) == 2: break
    assert all(v._version == frozen_versions[n] and v.grad is None for n,v in model.named_parameters() if not v.requires_grad)
    torch.cuda.synchronize()
    if rank == 0:
        output = dict(model=args.model, modality=args.modality, world_size=world, losses=losses,
                      lengths=lengths, seconds=time.monotonic()-t, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                      gradients_checked=True, frozen_parameters_unchanged=True, protocol=p)
        write(BASE/'finetuning/results'/f'smoke_{args.model}_{args.modality}_{world}gpu.json', output)
        print(json.dumps(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'extract', 'smoke', 'train', 'evaluate'])
    parser.add_argument('--config', default=str(BASE/'configs/common.json'))
    parser.add_argument('--model', choices=['2b', '8b'], default='2b')
    parser.add_argument('--modality', choices=list('COB'), default='B')
    parser.add_argument('--method', choices=['lp', 'ft'], default='ft')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.command == 'prepare': print(prepare(cfg)); return
    if args.command in ('train', 'evaluate') and not args.run: parser.error('--run is required')
    check_manifests()
    device, rank, world = runtime()
    p = protocol(cfg, args.model, args.modality)
    try:
        globals()[args.command](args, cfg, device, rank, world, p)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
