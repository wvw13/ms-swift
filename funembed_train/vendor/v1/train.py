"""Build a strict official ms-swift command; only --execute starts training."""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from common import ROOT, read_json, dump, sha
from verify import verify

def command(args, cfg):
    cmd = [sys.executable, '-m', 'swift.cli.sft']
    options = {
        'model': args.model, 'model_type': 'qwen3_vl_emb', 'task_type': 'embedding',
        'tuner_type': 'lora', 'target_regex': r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$',
        'freeze_vit': True, 'freeze_aligner': True, 'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout,
        'learning_rate': args.learning_rate, 'loss_type': 'infonce', 'torch_dtype': 'bfloat16',
        'attn_impl': args.attn_impl, 'gradient_checkpointing': True,
        'dataset': str(args.pairs / 'pairs.jsonl'), 'output_dir': str(args.output), 'add_version': False,
        'split_dataset_ratio': 0, 'dataset_shuffle': False, 'train_dataloader_shuffle': False,
        'group_by_length': False, 'packing': False, 'padding_free': False, 'strict': True,
        'lazy_tokenize': True, 'truncation_strategy': 'delete', 'max_length': args.max_length,
        'max_steps': cfg['steps'], 'num_train_epochs': 1,
        'per_device_train_batch_size': cfg['micro_batch'], 'gradient_accumulation_steps': cfg['grad_accum'],
        'seed': cfg['seed'], 'data_seed': cfg['seed'], 'eval_strategy': 'no',
        'save_strategy': 'steps', 'save_steps': args.save_steps, 'save_total_limit': args.save_total_limit,
        'logging_steps': 1, 'dataloader_num_workers': 0, 'dataloader_drop_last': True,
        'ddp_find_unused_parameters': False, 'report_to': 'tensorboard',
        'external_plugins': str(Path(__file__).with_name('runtime_plugin.py')),
        'callbacks': 'funembed_v1_audit', 'include_num_input_tokens_seen': True,
        'average_tokens_across_devices': False,
        'warmup_ratio': args.warmup_ratio, 'lr_scheduler_type': 'cosine', 'weight_decay': args.weight_decay,
    }
    for key, value in options.items():
        cmd += [f'--{key}', str(value).lower() if isinstance(value, bool) else str(value)]
    if args.resume:
        cmd += ['--resume_from_checkpoint', str(args.resume.resolve()), '--load_args', 'false',
                '--ignore_data_skip', 'false']
    # swift.cli.sft alone does not spawn workers; use torchrun's Python module explicitly.
    if cfg['world_size'] > 1:
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={cfg["world_size"]}', '--module', 'swift.cli.sft'] + cmd[3:]
    return cmd

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--image-max-tokens', type=int, required=True)
    p.add_argument('--max-length', type=int, required=True)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--attn-impl', choices=['sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--save-steps', type=int, default=50)
    p.add_argument('--save-total-limit', type=int, default=0, help='0 keeps all checkpoints for fixed-gallery validation')
    p.add_argument('--warmup-ratio', type=float, required=True)
    p.add_argument('--weight-decay', type=float, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', type=Path, help='Resume optimizer state and exact original schedule; keep original output/arguments')
    args = p.parse_args()
    args.pairs, args.output = args.pairs.resolve(), args.output.resolve()
    cfg = read_json(args.pairs / 'config.json')
    gpu_ids = args.gpus.split(',')
    if len(gpu_ids) != cfg['world_size'] or len(set(gpu_ids)) != len(gpu_ids):
        p.error('GPU count must match prepared world_size; regenerate data if changing world/batch/accumulation')
    if min(args.image_max_tokens, args.max_length, args.temperature, args.learning_rate, args.save_steps) <= 0:
        p.error('budgets/rates must be positive')
    if not 0 <= args.warmup_ratio <= 1 or args.weight_decay < 0:
        p.error('invalid warmup/weight decay')
    if args.lora_rank < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
        p.error('invalid LoRA parameters')
    verification = verify(args.pairs)
    cmd = command(args, cfg)
    env = os.environ.copy()
    env.pop('INFONCE_HARD_NEGATIVES', None)
    settings = {'CUDA_VISIBLE_DEVICES': args.gpus, 'INFONCE_USE_BATCH': 'False',
                'INFONCE_TEMPERATURE': str(args.temperature), 'INFONCE_INCLUDE_QQ': 'False',
                'INFONCE_INCLUDE_DD': 'False', 'INFONCE_MASK_FAKE_NEGATIVE': 'False',
                'IMAGE_MAX_TOKEN_NUM': str(args.image_max_tokens),
                'FUNEMBED_PAIR_DIR': str(args.pairs), 'HF_HUB_OFFLINE': '1', 'MODELSCOPE_OFFLINE': '1',
                'MODELSCOPE_CACHE': str(ROOT / 'ms-swift-discriminative/cache/modelscope'),
                'HF_DATASETS_CACHE': str(ROOT / 'ms-swift-discriminative/cache/datasets')}
    env.update(settings)
    print(json.dumps({'environment': settings, 'command': shlex.join(cmd), 'verification': verification}, indent=2))
    if args.execute:
        import swift
        if Path(swift.__file__).resolve() != (Path(os.environ['SWIFT_ROOT']) / 'swift/__init__.py').resolve():
            raise RuntimeError(f'unexpected swift installation: {swift.__file__}')
        if not Path(args.model).is_dir():
            raise ValueError('model must be an existing local directory')
        if args.resume:
            if not (args.resume / 'trainer_state.json').is_file() or args.resume.resolve().parent != args.output:
                raise ValueError('resume must be a complete checkpoint in the original output directory')
            previous = read_json(args.output / 'launch.json')
            base_cmd = cmd[:cmd.index('--resume_from_checkpoint')]
            if previous['command'] != base_cmd or previous['environment'] != settings or previous['sampling'] != cfg:
                raise ValueError('resume configuration differs from original launch')
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            framework_files = ['swift/dataloader/shard.py', 'swift/trainers/mixin.py',
                               'swift/loss/embedding.py', 'swift/template/templates/qwen.py',
                               'swift/model/models/qwen.py', 'swift/model/patcher.py']
            dump(args.output / 'launch.json', {'command': cmd, 'environment': settings, 'sampling': cfg,
                 'framework_sha256': {f: sha(Path(os.environ['SWIFT_ROOT']) / f) for f in framework_files}})
        if args.resume and any(sha(Path(os.environ['SWIFT_ROOT']) / f) != h for f, h in previous['framework_sha256'].items()):
            raise ValueError('framework changed since original launch')
        subprocess.run(cmd, env=env, check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
