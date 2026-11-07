"""General, task-free mask proposals from instruction text and one shared catalogue.

No task/config/checker input is accepted. Catalogue geometry, placement priors and
role labels are ignored. This component does not make the legacy task-prior fitter
board-compliant; its output is masks and candidate categories, never exact SAM IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping
import unicodedata

import numpy as np

from .sam_masks import SamMask


@dataclass(frozen=True)
class GeneralMaskConfig:
    model: str = 'sam3'
    sam_threshold: float = 0.2
    v2b_confidence: float = 0.25
    dedup_iou: float = 0.5
    instruction_context_words: int = 3
    max_instruction_phrase_words: int = 6

    def __post_init__(self):
        if self.model not in ('sam3', 'sam3.1_text'):
            raise ValueError('explicit detector-only SAM model required')
        if not .05 <= self.sam_threshold <= 1 or not 0 <= self.v2b_confidence <= 1:
            raise ValueError('invalid uniform confidence threshold')
        if not 0 <= self.dedup_iou <= 1:
            raise ValueError('invalid uniform deduplication threshold')


# English grammatical boundaries only; no object/task-specific phrase inventory.
_BOUNDARY = frozenset('a an the this that these those each every all some any of in into on onto at to from with '
                      'by for under over above below between beside behind around through and or then than '
                      'which whose who where while after before until up down out off'.split())
_ORDINAL = frozenset('first second third fourth fifth sixth seventh eighth ninth tenth last next'.split())
_ARTICLE = frozenset('a an the this that these those each every all some'.split())


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.replace('_', ' ').lower()))


def catalogue_labels(index: Mapping) -> dict[str, tuple[str, ...]]:
    """Global category/caption labels only. No tasks.json, role, geometry or prior fields."""
    labels: dict[str, set[str]] = {}
    for key, row in sorted(index.items()):
        if not isinstance(key, str) or key.count('/') != 2 or not isinstance(row, Mapping):
            continue
        category = key.split('/')[1]
        for value in [category, *(row.get('names') or [])]:
            if not isinstance(value, str):
                continue
            label = normalize(value)
            if label:
                labels.setdefault(label, set()).add(category)
    return {label: tuple(sorted(cats)) for label, cats in sorted(labels.items())}


def instruction_phrases(instruction: str, labels: Mapping[str, tuple], cfg=GeneralMaskConfig()) -> list[str]:
    """Deterministic noun/object-phrase approximation, without another model or task inference.

    Extend catalogue nouns with nearby instruction modifiers; also retain short
    determiner-led chunks to cover nouns absent from the catalogue. No semantic
    aliases are supplied by code. Punctuation/English function words bound chunks.
    """
    out: set[str] = set()
    chunks = re.split(r'[,.;:!?]', str(instruction))
    for chunk in chunks:
        words = normalize(chunk).split()
        for i, word in enumerate(words):
            if word not in _ARTICLE:
                continue
            phrase = []
            for w in words[i + 1:]:
                if w in _BOUNDARY:
                    break
                if w not in _ORDINAL and not w.isdigit():
                    phrase.append(w)
            if 0 < len(phrase) <= cfg.max_instruction_phrase_words:
                out.add(' '.join(phrase))
        for label in labels:
            noun = label.split()
            variants = [noun, [*noun[:-1], noun[-1] + 's']]
            if noun[-1].endswith('y'):
                variants.append([*noun[:-1], noun[-1][:-1] + 'ies'])
            for v in variants:
                for i in range(len(words) - len(v) + 1):
                    if words[i:i + len(v)] != v:
                        continue
                    start = max(0, i - cfg.instruction_context_words)
                    prefix = words[start:i]
                    # The last grammatical boundary starts the noun phrase.
                    for j in range(len(prefix) - 1, -1, -1):
                        if prefix[j] in _BOUNDARY:
                            prefix = prefix[j + 1:]
                            break
                    prefix = [w for w in prefix if w not in _ORDINAL and not w.isdigit()]
                    phrase = [*prefix, *v]
                    if len(phrase) <= cfg.max_instruction_phrase_words:
                        out.add(' '.join(phrase))
    return sorted(out)


def prompts(instruction: str, index: Mapping, cfg=GeneralMaskConfig()) -> list[str]:
    labels = catalogue_labels(index)
    return sorted(set(labels) | set(instruction_phrases(instruction, labels, cfg)))


def categories_for(phrase: str, labels: Mapping[str, tuple]) -> tuple[str, ...]:
    if phrase in labels:
        return labels[phrase]
    words = set(normalize(phrase).split())
    return tuple(sorted({cat for label, cats in labels.items() if set(label.split()) <= words for cat in cats}))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def deduplicate(base: list, extra: list, threshold=.5) -> list:
    """Base-first union preserves existing asset identities; extras are score-sorted."""
    kept = []
    for d in [*base, *sorted(extra, key=lambda d: -d.score)]:
        if np.any(d.mask) and not any(mask_iou(d.mask, old.mask) > threshold for old in kept):
            kept.append(d)
    return kept


@dataclass
class GeneralProposal:
    mask: np.ndarray
    score: float
    categories: tuple[str, ...]
    prompt: str
    # No asset key: a category/text response is not an exact identity.


def propose(image: np.ndarray, instruction: str, index: Mapping, v2b, sam,
            cfg=GeneralMaskConfig()) -> tuple[list, dict]:
    """One API for every episode. Inputs contain no task identity or task candidate pool."""
    labels = catalogue_labels(index)
    bank = prompts(instruction, index, cfg)
    base = v2b.detect(image, conf=cfg.v2b_confidence)
    masks: list[SamMask] = sam.text(image, bank, thr=cfg.sam_threshold, model=cfg.model)
    extra = [GeneralProposal(m.mask, m.score, categories_for(m.prompt, labels), m.prompt)
             for m in masks if m.score >= cfg.sam_threshold and np.any(m.mask)]
    return deduplicate(base, extra, cfg.dedup_iou), {'prompts': bank, 'base': len(base), 'sam': len(extra)}
