"""Differentiable official Qwen embedding + a separate multilabel linear head."""
import importlib.util
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from common import paths_for


class Cases(Dataset):
    def __init__(self, rows, modality):
        self.rows, self.modality = rows, modality

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return index, paths_for(row, self.modality), row['labels']


class Collator:
    def __init__(self, cfg, size, modality):
        from transformers import Qwen3VLProcessor
        self.cfg, self.modality = cfg, modality
        self.processor = Qwen3VLProcessor.from_pretrained(cfg['models'][size], padding_side='right', local_files_only=True)

    def __call__(self, items):
        from qwen_vl_utils import process_vision_info
        conversations = []
        for index, paths, labels in items:
            content = [dict(type='image', image=Path(p).as_uri(),
                            min_pixels=self.cfg['min_pixels'], max_pixels=self.cfg['max_pixels']) for p in paths]
            conversations.append([
                dict(role='system', content=[dict(type='text', text=self.cfg['instructions'][self.modality])]),
                dict(role='user', content=content)])
        # Follow the shipped embedding script, without its NULL fallback/truncation.
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, videos = process_vision_info(conversations, image_patch_size=16)
        assert videos is None
        inputs = self.processor(text=text, images=images, padding=True, truncation=False,
                                do_resize=False, return_tensors='pt')
        lengths = inputs['attention_mask'].sum(1)
        if int(lengths.max()) > self.cfg['max_length']:
            raise ValueError(f"Input length {lengths.tolist()} exceeds max_length={self.cfg['max_length']}; no silent truncation")
        expected_images = sum(len(item[1]) for item in items)
        assert len(inputs['image_grid_thw']) == expected_images
        merge = self.processor.image_processor.merge_size
        expected_tokens = int((inputs['image_grid_thw'].prod(-1) // (merge * merge)).sum())
        image_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        assert int((inputs['input_ids'] == image_id).sum()) == expected_tokens
        return dict(inputs=dict(inputs), y=torch.tensor([x[2] for x in items], dtype=torch.float32),
                    indices=torch.tensor([x[0] for x in items]), lengths=lengths)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch['inputs'].items()}


class Classifier(nn.Module):
    def __init__(self, cfg, size, method, device):
        super().__init__()
        directory = Path(cfg['models'][size])
        name = 'funembed_official_' + size
        spec = importlib.util.spec_from_file_location(name, directory / 'scripts/qwen3_vl_embedding.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.encoder = module.Qwen3VLForEmbedding.from_pretrained(
            directory, dtype=torch.bfloat16, attn_implementation=cfg['attention'], local_files_only=True)
        self.encoder.config.use_cache = False
        self.encoder.requires_grad_(False)
        self.dimension = self.encoder.config.text_config.hidden_size
        if cfg['patch_embed_fp32']:
            patch = self.encoder.model.visual.patch_embed
            forward = patch.forward
            patch.float()
            def fp32_forward(x, *args, **kwargs):
                with torch.autocast(device_type='cuda', enabled=False):
                    return forward(x.float(), *args, **kwargs).to(torch.bfloat16)
            patch.forward = fp32_forward
        self.targets = []
        if method == 'ft':
            from peft import LoraConfig, get_peft_model
            pattern = r'.*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
            self.targets = [n for n, m in self.encoder.named_modules() if isinstance(m, nn.Linear) and re.fullmatch(pattern, n)]
            expected = self.encoder.config.text_config.num_hidden_layers * 7
            assert len(self.targets) == expected, (len(self.targets), expected)
            lc = cfg['ft']
            self.encoder = get_peft_model(self.encoder, LoraConfig(r=lc['rank'], lora_alpha=lc['alpha'],
                lora_dropout=lc['dropout'], target_modules=self.targets, bias='none'))
            if lc['gradient_checkpointing']:
                self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.head = nn.Linear(self.dimension, 22, dtype=torch.float32)
        self.to(device)
        self.method = method
        self.train(method == 'ft')
        for name, p in self.encoder.named_parameters():
            if p.requires_grad:
                assert 'lora_' in name and 'language_model' in name and 'visual' not in name, name

    def train(self, mode=True):
        super().train(mode)
        # No augmentation/dropout in frozen visual branch.
        if hasattr(self, 'encoder'):
            self.encoder.get_submodule('model.visual' if not hasattr(self.encoder, 'peft_config') else 'base_model.model.model.visual').eval()
        return self

    def embedding(self, inputs):
        output = self.encoder(**inputs, use_cache=False)
        mask = inputs['attention_mask']
        positions = mask.shape[1] - 1 - mask.flip(1).argmax(1)
        h = output.last_hidden_state[torch.arange(len(mask), device=mask.device), positions]
        return F.normalize(h.float(), dim=-1)

    def forward(self, inputs):
        return self.head(self.embedding(inputs))

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        assert set(state) == expected, 'checkpoint parameter mismatch'
        self.load_state_dict(state, strict=False)
