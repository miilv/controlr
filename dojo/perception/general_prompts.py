"""Instruction-only grammatical phrase approximation plus three fixed generic words."""
from dataclasses import dataclass
import re

from .general_masks import normalize


@dataclass(frozen=True)
class PromptConfig:
    generic: tuple[str, ...] = ('object', 'container', 'tool')
    max_phrase_words: int = 6


# Grammar/action words, shared across every instruction. No object or task lists.
BOUNDARY = frozenset(('a an the this that these those each every all some any of in into on onto at to from with '
                      'by for under over above below between beside behind around through and or then than '
                      'which whose who where when while after before until up down out off it them its their '
                      'according using than again same as more needed').split())
VERBS = frozenset(('pick put place move take lift hold throw close open insert sort remember observe identify '
                   'pour reset push pull release grasp keep fill turn select match use is are be has have').split())
NUMBERS = frozenset(('one two three four five six seven eight nine ten first second third fourth fifth sixth '
                     'seventh eighth ninth tenth last next corresponding').split())


def instruction_prompts(instruction, cfg=PromptConfig()):
    phrases = set(cfg.generic)
    for sentence in re.split(r'[,.;:!?]', str(instruction)):
        chunks = []
        chunk = []
        for w in normalize(sentence).split():
            if w in BOUNDARY or w in VERBS:
                if chunk:
                    chunks.append(chunk)
                chunk = []
            elif w not in NUMBERS and not w.isdigit():
                chunk.append(w)
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            if 0 < len(chunk) <= cfg.max_phrase_words:
                phrases.add(' '.join(chunk))
                if len(chunk) > 1 and chunk[-1] in ('object', 'objects'):
                    phrases.add(' '.join(chunk[:-1]))
    return sorted(phrases)
