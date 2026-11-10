"""Versioned v1 inputs; CPU-only imports. No changes to audited base records."""
import hashlib
import json
import sys
from pathlib import Path

import os
ROOT = Path(os.environ['FUNEMBED_WORK'])
BASE = ROOT / 'inputs/phase2'
from swift_inputs import image_branch, text_branch, assemble_pair
CLASSES = json.loads(Path(os.environ['FUNEMBED_CLASSES']).read_text(encoding='utf-8'))

GROUP = {'C': 'cfp', 'O': 'oct', 'B': 'combined'}
TASKS = {'T1': ['C', 'O', 'B'], 'T2': ['C-C', 'O-O', 'B-B'], 'T3': ['O-C', 'C-O']}
EXPERIMENTS = {'E1': {'T1': 1}, 'E2': {'T2': 1}, 'E3': {'T2': 1, 'T3': 1},
               'E4': {'T1': 1, 'T2': 1, 'T3': 1}, 'T3': {'T3':1}, 'CUSTOM': {}}
IMAGE_INSTRUCTIONS = {
 'C': 'Represent this color fundus photograph for ophthalmic visual-feature and case retrieval. Focus on visible surface lesions, vessels, optic disc appearance, and color or pigment changes. Ignore watermarks, patient identifiers, device names, and non-anatomical text overlays. Encode only visible evidence; do not generate text.',
 'O': 'Represent these six ordered OCT slices jointly for ophthalmic visual-feature and case retrieval. Focus on retinal layer architecture, reflectivity, fluid, and foveal contour across slices. Ignore watermarks, patient identifiers, device names, and non-anatomical text overlays. Encode only visible evidence; do not generate text.',
 'B': 'Represent these ophthalmic images jointly for visual-feature and case retrieval. The first image is a color fundus photograph; the next six are ordered OCT slices. Preserve visible fundus lesions, vessels, optic disc and pigment features, and OCT layer structure, reflectivity, fluid and foveal contour. Ignore watermarks, identifiers, device names, and non-anatomical text overlays. Do not generate text.'}
# Preserve the PDF wording, including its combined-image phrase, for reproducibility.
IMAGE_TYPES = {'C': 'color fundus photographs', 'O': 'OCT image sequences',
               'B': 'combined color fundus and OCT images sequences'}
DEFAULT_INSTRUCTION = "Represent the user's input."

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def labels(record, protocol='modal'):
    group = GROUP[record['input_type']] if protocol == 'modal' else 'combined'
    label = record['labels'][group]
    if label['state'] not in ('normal', 'disease') or label.get('unknown'):
        return frozenset()
    ids = frozenset(label['ids'])
    if not ids or (21 in ids and ids != {21}):
        return frozenset()
    return ids

def load_records(base=BASE, split='internal_train'):
    result = {}
    for mod in GROUP:
        records = read_jsonl(Path(base) / split / f'{mod}.input_records.jsonl')
        for r in records:
            if r['split'] != split or r['input_type'] != mod:
                raise ValueError('record split/modality mismatch')
            if split == 'internal_train' and (not r['use_in_training_allowed'] or r['center'] == 'shandong'):
                raise ValueError('forbidden training record')
            paths = r['image_paths']
            if len(paths) != {'C': 1, 'O': 6, 'B': 7}[mod]:
                raise ValueError('wrong image count')
            oct_paths = paths if mod == 'O' else paths[1:] if mod == 'B' else []
            if any(not p.endswith(f'_{i:03}.jpg') for p, i in zip(oct_paths, range(7, 13))):
                raise ValueError('wrong OCT slice order')
        if len({r['input_id'] for r in records}) != len(records):
            raise ValueError('duplicate input IDs')
        result[mod] = records
    return result

def image_input(record, instruction='I1'):
    return image_branch(record['image_paths'], instruction=(DEFAULT_INSTRUCTION if instruction == 'I0'
                        else IMAGE_INSTRUCTIONS[record['input_type']]))

class TextBank:
    def __init__(self, cfp, oct_file, base=BASE, variant=None):
        self.files = [Path(cfp), Path(oct_file), Path(base) / 'text_dictionary.json']
        self.names = {x['label_id']: x['english_text'] for x in read_json(self.files[2])['entries']}
        self.definitions = {}
        for mod, path in zip(('C', 'O'), self.files):
            def unique(pairs):
                d = {}
                for key, value in pairs:
                    if key in d:
                        raise ValueError(f'duplicate JSON key: {key}')
                    d[key] = value
                return d
            raw = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)
            if set(raw) != {c['label'] for c in CLASSES}:
                raise ValueError(f'{path}: dictionary must match canonical 22 keys')
            self.definitions[mod] = {}
            for entry in CLASSES:
                label_id, name = entry['id'], entry['label']
                value = raw[name]
                if isinstance(value, list):
                    if variant is None:
                        raise ValueError('v1 variants require an explicit --variant index; never concatenate silently')
                    if not 0 <= variant < len(value):
                        raise ValueError(f'variant unavailable: {name}')
                    value = value[variant]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'empty/non-string definition: {name}')
                self.definitions[mod][label_id] = value.strip()

    def content(self, mod, label_id, kind='definition'):
        if kind == 'name':
            return self.names[label_id]
        definition = (f"CFP features: {self.definitions['C'][label_id]} OCT features: {self.definitions['O'][label_id]}"
                      if mod == 'B' else self.definitions[mod][label_id])
        return definition if kind == 'definition' else f'{self.names[label_id]} {definition}'

    def branch(self, mod, label_id, kind='definition', instruction='I1'):
        if instruction == 'I0':
            prompt = DEFAULT_INSTRUCTION
        elif kind == 'name':
            prompt = f'Represent this ophthalmic disease name for retrieval by {IMAGE_TYPES[mod]}.'
        else:
            prompt = (f'Represent this ophthalmic visual-feature description for retrieval by {IMAGE_TYPES[mod]}. '
                      'Preserve lesion morphology, location, distribution, and structural details. '
                      'Do not generate additional findings.')
        return text_branch(self.content(mod, label_id, kind), instruction=prompt)

    def snapshot(self, kind='definition', instruction='I1'):
        return {m: {str(i): self.branch(m, i, kind, instruction) for i in range(22)} for m in GROUP}
