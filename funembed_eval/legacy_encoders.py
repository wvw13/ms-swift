"""Unified encoder adapters for CLIP / SigLIP / Qwen3-VL-Embedding.

All adapters expose the same interface:
    encode_images(paths)      -> np.float32 [n, d], L2-normalized (per image)
    encode_texts(texts)       -> np.float32 [n, d], L2-normalized (per text)
    encode_joint(image_lists) -> np.float32 [n, d], L2-normalized (per case;
                                  only Qwen supports multi-image joint input)

CLIP / SigLIP use their own processors and the pooled+projected retrieval
features (get_image_features / get_text_features); neither is assumed to be
normalized, so L2 normalization is always applied explicitly.

Qwen3-VL-Embedding adapters load the OFFICIAL inference implementation shipped
with the weights (``<model>/scripts/qwen3_vl_embedding.py``) via importlib:
system instruction "Represent the user's input.", one user message per item,
last-token pooling with attention-mask-based position lookup (padding-safe),
2B/8B share this single implementation. The joint multi-image mode reuses the
official preprocessing internals (format/pooling) with several image content
entries in one user message, CFP first then OCT 007..012.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _l2n(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


def _load_pil(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _dir_hash(path: str) -> str:
    """Cheap content fingerprint of a model directory (name+size+mtime)."""
    import hashlib

    h = hashlib.sha1()
    for root, _, files in sorted(os.walk(path)):
        for fn in sorted(files):
            if fn.endswith((".safetensors", ".bin", ".pt", ".msgpack", ".h5")):
                fp = os.path.join(root, fn)
                st = os.stat(fp)
                h.update(f"{os.path.relpath(fp, path)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


class BaseEncoder:
    supports_joint = False

    def encode_images(self, paths: list) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts: list) -> np.ndarray:
        raise NotImplementedError

    def encode_joint(self, image_lists: list) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no joint multi-image input")

    def info(self) -> dict:
        return dict(self.base_info)

    def release(self):
        pass


class HFClipLikeEncoder(BaseEncoder):
    """Shared implementation for CLIPModel and SiglipModel retrieval features."""

    def __init__(self, name: str, cfg: dict, model_cls, processor_cls, expected_dim: int):
        self.name = name
        path = cfg["path"]
        self.model_path = path
        self.device = cfg.get("device", "cpu")
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            cfg.get("dtype", "float32")
        ]
        self.model = model_cls.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.processor = processor_cls.from_pretrained(path)
        self.batch_size = int(cfg.get("batch_size", 32))
        self.expected_dim = expected_dim
        cfg_json = os.path.join(path, "config.json")
        import json

        with open(cfg_json) as f:
            mc = json.load(f)
        self._model_config = mc
        self.base_info = {
            "model": name,
            "model_path": path,
            "weights_hash": _dir_hash(path),
            "type": type(self.model).__name__,
            "device": str(self.device),
            "dtype": cfg.get("dtype", "float32"),
            "expected_dim": expected_dim,
            "n_params": int(sum(p.numel() for p in self.model.parameters())),
            "transformers_version": __import__("transformers").__version__,
        }

    @torch.no_grad()
    def encode_images(self, paths: list) -> np.ndarray:
        out = []
        for i in range(0, len(paths), self.batch_size):
            chunk = paths[i : i + self.batch_size]
            pixels = self.processor(images=[_load_pil(p) for p in chunk], return_tensors="pt").pixel_values
            feats = self.model.get_image_features(pixel_values=pixels.to(self.device))
            out.append(_l2n(feats).cpu())
        emb = torch.cat(out).numpy()
        assert emb.shape[1] == self.expected_dim, f"{self.name}: dim {emb.shape[1]} != {self.expected_dim}"
        return emb

    @torch.no_grad()
    def encode_texts(self, texts: list) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            inputs = self._tokenize(chunk)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_text_features(**inputs)
            out.append(_l2n(feats).cpu())
        emb = torch.cat(out).numpy()
        assert emb.shape[1] == self.expected_dim
        return emb

    def _tokenize(self, chunk):
        raise NotImplementedError

    def info(self) -> dict:
        d = dict(self.base_info)
        pp = self.processor.image_processor.to_dict() if hasattr(self.processor, "image_processor") else {}
        size = pp.get("crop_size") or pp.get("size") or {}
        d["image_input_size"] = size
        d["tokenizer_max_len"] = getattr(self.processor.tokenizer, "model_max_length", None)
        return d

    def release(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ClipEncoder(HFClipLikeEncoder):
    def __init__(self, name: str, cfg: dict):
        from transformers import CLIPModel, CLIPProcessor

        super().__init__(name, cfg, CLIPModel, CLIPProcessor, expected_dim=768)

    def _tokenize(self, chunk):
        # CLIP: pad to longest, truncate to context length (77)
        return self.processor.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")


class SiglipEncoder(HFClipLikeEncoder):
    def __init__(self, name: str, cfg: dict):
        from transformers import SiglipModel, SiglipProcessor

        super().__init__(name, cfg, SiglipModel, SiglipProcessor, expected_dim=1152)

    def _tokenize(self, chunk):
        # SigLIP official usage: pad to the trained max length (64), truncate
        return self.processor.tokenizer(chunk, padding="max_length", truncation=True, return_tensors="pt")


class QwenVLEncoder(BaseEncoder):
    supports_joint = True

    def __init__(self, name: str, cfg: dict):
        self.name = name
        path = cfg["path"]
        self.model_path = path
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            cfg.get("dtype", "bfloat16")
        ]
        # official script shipped with the weights
        script = os.path.join(path, "scripts", "qwen3_vl_embedding.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"official embedding script not found: {script}")
        spec = importlib.util.spec_from_file_location(f"qwen3_vl_embedding_{name}", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        max_pixels = int(cfg.get("max_pixels", 1024 * 1024))   # target budget 1024x1024 per image
        min_pixels = int(cfg.get("min_pixels", 4 * 32 * 32))
        self.instruction = cfg.get("instruction", "Represent the user's input.")
        kwargs = dict(
            max_length=int(cfg.get("max_length", 8192)),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            default_instruction=self.instruction,
            dtype=dtype,
        )
        if str(self.device).startswith("cuda"):
            # The official constructor places the model on "cuda" (= default
            # device cuda:0); pin the default device to the target GPU so no
            # shadow allocation stays cached on cuda:0 in multi-GPU runs.
            torch.cuda.set_device(self.device)
        if self.device == "cpu":
            # the official constructor hardcodes cuda-if-available; force CPU
            orig = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
            try:
                self.embedder = mod.Qwen3VLEmbedder(path, **kwargs)
            finally:
                torch.cuda.is_available = orig
        else:
            self.embedder = mod.Qwen3VLEmbedder(path, **kwargs)
            self.embedder.model.to(self.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # release transient default-device memory
        self.embedder.model.eval()
        # ---- environment-specific performance fix (numerics-preserving) -------
        # On this machine (RTX 5880 Ada, torch 2.9.1+cu128, driver 580) the bf16
        # convolution inside the vision patch embedding takes ~21 s per image
        # (kernel fallback), while the same conv in float32 takes 0.3 ms with
        # cosine similarity 0.999998 to the bf16 result. We therefore compute
        # ONLY the patch-embedding conv in fp32 (strictly higher precision) and
        # cast its output back to the model dtype; every other layer follows the
        # official bf16 path. Disable via config ``patch_embed_fp32: false``.
        self.patch_embed_fp32 = bool(cfg.get("patch_embed_fp32", True))
        if self.patch_embed_fp32:
            visual = self.embedder.model.model.visual
            pe = visual.patch_embed
            model_dtype = next(pe.parameters()).dtype
            if model_dtype == torch.bfloat16:
                orig_forward = pe.forward

                def forward_fp32(x, *a, **k):
                    return orig_forward(x.to(torch.float32), *a, **k).to(model_dtype)

                pe.forward = forward_fp32
                pe.float()
        self.batch_size = int(cfg.get("batch_size", 4))
        self.joint_batch_size = int(cfg.get("joint_batch_size", 1))
        self.max_length = int(cfg.get("max_length", 8192))
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.expected_dim = int(cfg["embed_dim"])
        # diagnostics of the last encode calls
        self.last_seq_lens: list = []
        self.last_pooled_positions: list = []
        self._truncated = False

        import json

        with open(os.path.join(path, "config.json")) as f:
            self._model_config = json.load(f)
        self.base_info = {
            "model": name,
            "model_path": path,
            "weights_hash": _dir_hash(path),
            "type": "Qwen3VLForEmbedding (official script)",
            "device": str(self.device),
            "dtype": cfg.get("dtype", "bfloat16"),
            "expected_dim": self.expected_dim,
            "n_params": int(sum(p.numel() for p in self.embedder.model.parameters())),
            "instruction": self.instruction,
            "max_pixels_per_image": max_pixels,
            "min_pixels_per_image": min_pixels,
            "max_length": self.max_length,
            "transformers_version": __import__("transformers").__version__,
        }

    # ---------------- low-level helpers (mirror official process()) ---------
    @torch.no_grad()
    def _run_conversations(self, conversations: list) -> np.ndarray:
        inputs = self.embedder._preprocess_inputs(conversations)
        seq_len = inputs["input_ids"].shape[1]
        self.last_seq_lens.append(int(seq_len))
        if seq_len >= self.max_length:
            self._truncated = True
        inputs = {k: v.to(self.embedder.model.device) for k, v in inputs.items()}
        outputs = self.embedder.forward(inputs)
        emb = self.embedder._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
        # remember pooled position of the first item to prove it is the true end
        am = inputs["attention_mask"]
        self.last_pooled_positions.append(int(am.shape[1] - am.flip(dims=[1]).argmax(dim=1)[0].item() - 1))
        return _l2n(emb).cpu().numpy()

    def encode_images(self, paths: list) -> np.ndarray:
        outs = []
        for i in range(0, len(paths), self.batch_size):
            chunk = paths[i : i + self.batch_size]
            convs = [self.embedder.format_model_input(image=p) for p in chunk]
            outs.append(self._run_conversations(convs))
        emb = np.concatenate(outs)
        assert emb.shape[1] == self.expected_dim, f"{self.name}: dim {emb.shape[1]} != {self.expected_dim}"
        return emb

    def encode_texts(self, texts: list) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            convs = [self.embedder.format_model_input(text=t) for t in chunk]
            outs.append(self._run_conversations(convs))
        emb = np.concatenate(outs)
        assert emb.shape[1] == self.expected_dim
        return emb

    def _joint_conversation(self, image_paths: list) -> list:
        """One user message containing several images (CFP first, then OCTs)."""
        content = []
        for p in image_paths:
            content.append(
                {
                    "type": "image",
                    "image": p if p.startswith(("http", "file://")) else "file://" + p,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
        return [
            {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
            {"role": "user", "content": content},
        ]

    def encode_joint(self, image_lists: list) -> np.ndarray:
        outs = []
        for i in range(0, len(image_lists), self.joint_batch_size):
            chunk = image_lists[i : i + self.joint_batch_size]
            convs = [self._joint_conversation(imgs) for imgs in chunk]
            outs.append(self._run_conversations(convs))
        emb = np.concatenate(outs)
        assert emb.shape[1] == self.expected_dim
        return emb

    def info(self) -> dict:
        d = dict(self.base_info)
        d["last_seq_len_max"] = max(self.last_seq_lens) if self.last_seq_lens else None
        d["truncated"] = self._truncated
        return d

    def release(self):
        del self.embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_encoder(name: str, cfg_all: dict) -> BaseEncoder:
    models = cfg_all["models"]
    if name not in models:
        raise KeyError(f"unknown model '{name}'; configured: {list(models)}")
    cfg = dict(models[name])
    cfg.setdefault("path", "")
    kind = cfg.get("type")
    if kind == "clip":
        return ClipEncoder(name, cfg)
    if kind == "siglip":
        return SiglipEncoder(name, cfg)
    if kind == "qwen":
        return QwenVLEncoder(name, cfg)
    raise ValueError(f"unknown model type '{kind}' for '{name}'")
