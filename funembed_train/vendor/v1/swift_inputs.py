"""Single-branch construction and shape-only pairing; no clinical sample selection.

Research metadata is deliberately rejected by these model-facing interfaces.
These functions neither load a tokenizer/model nor read image contents.
"""
from copy import deepcopy
from collections.abc import Mapping, Sequence


def _messages(content: str, instruction: str | None):
    if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
        raise ValueError('instruction must be a nonempty string or None (template default)')
    result = []
    if instruction is not None:
        result.append({'role': 'system', 'content': instruction})
    result.append({'role': 'user', 'content': content})
    return result


def image_branch(image_paths: Sequence[str], *, instruction: str | None):
    """Paths become images, never text; user content contains only placeholders."""
    if isinstance(image_paths, (str, bytes)) or not isinstance(image_paths, Sequence):
        raise TypeError('image_paths must be a sequence of strings')
    paths = list(image_paths)
    if not paths or any(not isinstance(p, str) or not p.strip() for p in paths):
        raise ValueError('at least one nonempty image path is required')
    branch = {'messages': _messages('<image>' * len(paths), instruction), 'images': paths}
    validate_branch(branch)
    return branch


def text_branch(text: str, *, instruction: str | None):
    """One deterministic text supplied by the caller; no diagnosis combination."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be a nonempty string')
    if any(tag in text for tag in ('<image>', '<video>', '<audio>')):
        raise ValueError('text branch must not contain media placeholders')
    branch = {'messages': _messages(text, instruction)}
    validate_branch(branch)
    return branch


def validate_branch(branch: Mapping):
    """Check our messages/images interface, without tokenizer or file I/O."""
    if not isinstance(branch, Mapping) or set(branch) - {'messages', 'images'}:
        raise ValueError('branch accepts only messages and optional images; keep metadata in a sidecar')
    messages = branch.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    for message in messages:
        if not isinstance(message, dict) or set(message) != {'role', 'content'}:
            raise ValueError('each message requires role and content only')
        if not isinstance(message['content'], str) or not isinstance(message['role'], str):
            raise ValueError('role/content must be strings')
    images = branch.get('images', [])
    if not isinstance(images, list) or any(not isinstance(x, str) or not x for x in images):
        raise ValueError('images must be a list of nonempty paths')
    placeholders = sum(m['content'].count('<image>') for m in messages)
    if placeholders != len(images):
        raise ValueError(f'image placeholder/path mismatch: {placeholders} vs {len(images)}')
    return True


def assemble_pair(*, anchor: Mapping, positive: Mapping, negatives: Sequence[Mapping]):
    """Assemble exactly the caller's single positive and ordered negatives.

    This function does not determine whether any branch is clinically positive
    or negative, infer labels, sample candidates, or carry research metadata.
    An empty negatives list is permitted as a shape; it is not a loss policy.
    """
    validate_branch(anchor)
    validate_branch(positive)
    if isinstance(negatives, (str, bytes)) or not isinstance(negatives, Sequence):
        raise TypeError('negatives must be an explicit sequence (possibly empty)')
    for negative in negatives:
        validate_branch(negative)
    result = deepcopy(dict(anchor))
    result['positive_messages'] = [deepcopy(positive['messages'])]
    result['positive_images'] = [deepcopy(positive.get('images', []))]
    if negatives:
        result['negative_messages'] = [deepcopy(v['messages']) for v in negatives]
        result['negative_images'] = [deepcopy(v.get('images', [])) for v in negatives]
    return result
