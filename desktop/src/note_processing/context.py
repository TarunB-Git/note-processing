"""Attach an optional reading guide and bounded neighboring text."""

import numpy as np

from note_processing.encoder import unit_vector


def attach_context(units, guide='', neighbor_words=0):
    if not isinstance(guide, str) or len(guide.split()) > 1000:
        raise ValueError('Context accepts up to 1,000 words')
    if not isinstance(neighbor_words, int) or neighbor_words not in {0, 500, 1000}:
        raise ValueError('Neighbor context must be 0, 500, or 1,000 words')
    groups = {}
    for unit in units:
        groups.setdefault(unit['source'], []).append(unit)
    result = {}
    for source, notes in groups.items():
        for index, note in enumerate(notes):
            neighbors = ''
            if neighbor_words:
                half = neighbor_words // 2
                before, after = [], []
                for previous in reversed(notes[:index]):
                    before[:0] = previous['text'].split()
                    if len(before) >= half:
                        break
                for following in notes[index + 1:]:
                    after.extend(following['text'].split())
                    if len(after) >= half:
                        break
                neighbors = 'Before: ' + ' '.join(before[-half:]) + '\nAfter: ' + ' '.join(after[:half])
            result[note['id']] = {**note, 'context': guide.strip(), 'neighbors': neighbors}
    return [result[note['id']] for note in units]


def contextual_vector(encoder, note, vector):
    context = '\n'.join(filter(None, [note.get('context', ''), note.get('neighbors', '')]))
    if not context:
        return vector
    words = context.split()
    chunks = [' '.join(words[offset:offset + 140]) for offset in range(0, len(words), 140)]
    context_vector = unit_vector(np.stack([item.vector for item in encoder.encode(chunks)]).mean(axis=0))
    return unit_vector(vector * 0.85 + context_vector * 0.15).astype(np.float32)
