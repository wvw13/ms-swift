"""Project-only callback: fail on schedule/LoRA drift, trace completed updates."""
import json
import os
import time
from pathlib import Path
from transformers import TrainerCallback
from swift.callbacks.mapping import callbacks_map

def fix_infonce_accumulation(trainer):
    """Average local row-mean InfoNCE across HF accumulation microsteps.

    transformers 4.57 skips its default division for custom compute_loss_func.
    Local swift.Trainer divides only if model_accepts_loss_kwargs is true;
    Qwen3-VL declares that false. Keep the correction in this project plugin.
    """
    from swift.loss.embedding import InfonceLoss
    original = trainer.compute_loss_func
    needs_fix = isinstance(original, InfonceLoss) and not trainer.model_accepts_loss_kwargs
    if not needs_fix:
        return False
    if trainer.accelerator.gradient_accumulation_steps != 1:
        raise RuntimeError('Unexpected Accelerator scaling; refusing double accumulation division')
    def averaged(outputs, labels, **kwargs):
        loss = original(outputs, labels, **kwargs)
        if trainer.model.training:
            steps = getattr(trainer, 'current_gradient_accumulation_steps', trainer.args.gradient_accumulation_steps)
            loss = loss / steps
        return loss
    trainer.compute_loss_func = averaged
    return True

class V1AuditCallback(TrainerCallback):
    def __init__(self, args=None, trainer=None):
        # Official ms-swift callbacks_map instantiates callbacks with (args, trainer).
        self.trainer = trainer

    def on_train_begin(self, args, state, control, model=None, train_dataloader=None, **kwargs):
        root = Path(os.environ['FUNEMBED_PAIR_DIR'])
        self.cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        self.schedule = [json.loads(s) for s in (root / 'schedule.jsonl').read_text(encoding='utf-8').splitlines()]
        for name, expected in [('world_size', self.cfg['world_size']),
                               ('per_device_train_batch_size', self.cfg['micro_batch']),
                               ('gradient_accumulation_steps', self.cfg['grad_accum']),
                               ('max_steps', self.cfg['steps'])]:
            if getattr(args, name) != expected:
                raise RuntimeError(f'{name} differs from materialized schedule')
        if args.train_dataloader_shuffle or args.group_by_length:
            raise RuntimeError('v1 schedule must not be shuffled/regrouped')
        if len(train_dataloader.dataset) != self.cfg['rows']:
            raise RuntimeError('rows dropped/added by preprocessing')
        trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
        allowed = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
        if not trainable or any('lora_' not in n or 'language_model' not in n or
                                not any(f'.{m}.' in n for m in allowed) for n in trainable):
            raise RuntimeError('trainable parameters must be language-model projection LoRA only')
        if state.is_world_process_zero:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.output_dir) / 'trainable_parameters.json').write_text(json.dumps(trainable, indent=2))
        self.start = time.monotonic()
        fixed = fix_infonce_accumulation(self.trainer)
        if state.is_world_process_zero:
            (Path(args.output_dir) / 'loss_normalization.json').write_text(json.dumps({
                'trainer_class': type(self.trainer).__name__,
                'model_accepts_loss_kwargs': self.trainer.model_accepts_loss_kwargs,
                'accelerator_gradient_accumulation_steps': self.trainer.accelerator.gradient_accumulation_steps,
                'project_accumulation_average_fix': fixed}, indent=2))
        self.validation = None
        if os.environ.get('FUNEMBED_E2E_AUDIT') == '1':
            from validation_audit import ValidationAudit
            self.validation = ValidationAudit(self.trainer, args, state, model, train_dataloader, root)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.monotonic()
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        import torch
        rank_entry = {'global_step': state.global_step, 'rank': args.process_index}
        if torch.cuda.is_available():
            rank_entry.update(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                              peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        with (Path(args.output_dir) / f'completed_updates_rank{args.process_index}.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(rank_entry) + '\n')
        if state.is_world_process_zero:
            entry = dict(self.schedule[state.global_step - 1])
            entry.update(global_step=state.global_step, elapsed_seconds=time.monotonic() - self.start,
                         step_seconds=time.monotonic() - self.step_start,
                         num_input_tokens_seen=getattr(state, 'num_input_tokens_seen', None))
            import torch
            if torch.cuda.is_available():
                entry.update(rank0_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                             rank0_peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            with (Path(args.output_dir) / 'completed_updates.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        if self.validation:
            self.validation.step_end(state)
            stop = int(os.environ.get('FUNEMBED_VALIDATION_STOP_AFTER', '0'))
            if stop and state.global_step == stop:
                control.should_save = True
                control.should_training_stop = True

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.validation:
            self.validation.check_gradients(state)

    def on_train_end(self, args, state, control, **kwargs):
        if self.validation:
            self.validation.finish(state)

callbacks_map['funembed_v1_audit'] = V1AuditCallback
