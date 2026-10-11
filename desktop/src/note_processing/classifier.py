"""Sort notes using format cues, a trained intent head, and confirmed examples."""

import json
from pathlib import Path
from threading import RLock

import numpy as np

from note_processing.dates import extract_deadline
from note_processing.context import contextual_vector
from note_processing.encoder import Encoder
from note_processing.learning import classes_key, encoder_key, fit_head, matrices, softmax
from note_processing.paths import APP_HOME, PACKAGE_DIR
from note_processing.signals import features, normalized, rule

DEFAULT_CLASSES = PACKAGE_DIR / 'config/classes.json'
LOCAL_CLASSES = APP_HOME / 'classes.json'
BASE_DIR = PACKAGE_DIR


def validate_classes(classes):
    if not isinstance(classes, list) or not 2 <= len(classes) <= 32:
        raise ValueError('Provide between 2 and 32 categories')
    import re
    cleaned, names = [], set()
    for category in classes:
        if not isinstance(category, dict):
            raise ValueError('Each category needs a name, description, and examples')
        name, description, examples = category.get('name', ''), category.get('description', ''), category.get('examples', [])
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}', name):
            raise ValueError('Category IDs use lowercase letters, numbers, hyphens, and underscores')
        if name in names:
            raise ValueError(f'Duplicate category: {name}')
        if not isinstance(description, str) or not 1 <= len(description.strip()) <= 2000:
            raise ValueError(f'{name}: add a description of up to 2,000 characters')
        if not isinstance(examples, list) or len(examples) > 100 or any(not isinstance(s, str) or not 1 <= len(s.strip()) <= 2000 for s in examples):
            raise ValueError(f'{name}: use up to 100 non-empty examples')
        title = category.get('title', name)
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 60:
            raise ValueError('Category title must be 1–60 characters')
        names.add(name)
        cleaned.append({'name': name, 'title': title.strip(), 'description': description.strip(), 'examples': list(dict.fromkeys(s.strip() for s in examples))})
    return cleaned


def load_classes(path=None):
    path = Path(path) if path else LOCAL_CLASSES if LOCAL_CLASSES.exists() else DEFAULT_CLASSES
    return validate_classes(json.loads(path.read_text(encoding='utf8')))


class Classifier:
    def __init__(self, classes=None, model_dir=None, corrections=None, encoder=None, load_head=True):
        active = APP_HOME / 'active-encoder.json'
        if model_dir is None and active.exists():
            model_dir = json.loads(active.read_text())['directory']
        self.encoder = encoder or (Encoder(model_dir) if model_dir else Encoder())
        self.lock = RLock()
        self.set_classes(classes if classes is not None else load_classes())
        if load_head:
            self.load_head()
        self.set_corrections(corrections or [])

    def seed_records(self):
        return [{'text': text, 'label': c['name']} for c in self.classes for text in (c['examples'] or [c['description']])]

    def set_classes(self, classes):
        with self.lock:
            self.classes = validate_classes(classes)
            self.names = [c['name'] for c in self.classes]
            self.references, self.spans = [], []
            for category in self.classes:
                start = len(self.references)
                self.references.extend([category['description'], *category['examples']])
                self.spans.append((start, len(self.references)))
            self.reference_vectors = np.stack([item.vector for offset in range(0, len(self.references), 32)
                                               for item in self.encoder.encode(self.references[offset:offset + 32])])
            x, y = matrices(self.encoder, self.seed_records(), self.names)
            self.weights, self.bias = fit_head(x, y, self.names)
            self.head_kind = 'starter'

    def load_head(self):
        path = APP_HOME / 'training/head.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (saved['names'].tolist() == self.names and saved['encoder_key'].item() == encoder_key(self.encoder)
                        and 'classes_key' in saved and saved['classes_key'].item() == classes_key(self.classes)
                        and saved['weights'].shape == self.weights.shape):
                    self.install_head(saved['weights'], saved['bias'])

    def install_head(self, weights, bias):
        with self.lock:
            self.weights, self.bias = weights, bias
            self.head_kind = 'personal'

    def fresh(self, corrections=None, encoder=None):
        return Classifier(self.classes, encoder=encoder or self.encoder, corrections=corrections, load_head=False)

    def set_corrections(self, corrections):
        with self.lock:
            self.corrections = [r for r in corrections if r['label'] in self.names]
            self.exact = {normalized(r['text']): r['label'] for r in self.corrections}
            self.correction_vectors = np.stack([item.vector for offset in range(0, len(self.corrections), 32)
                for item in self.encoder.encode([r['text'] for r in self.corrections[offset:offset + 32]])]) if self.corrections else None

    def classify(self, units, anchor=None, progress=None):
        with self.lock:
            results = []
            for offset in range(0, len(units), 32):
                results.extend(self._batch(units[offset:offset + 32], anchor))
                if progress:
                    progress(min(offset + 32, len(units)), len(units))
            return results

    def _batch(self, units, anchor):
        encoded = self.encoder.encode([u['text'] for u in units])
        vectors = [contextual_vector(self.encoder, unit, item.vector) for unit, item in zip(units, encoded)]
        similarities = np.stack(vectors) @ self.reference_vectors.T
        semantic = np.stack([similarities[:, start:end].max(axis=1) for start, end in self.spans], axis=1)
        x = np.stack([features(unit, vector) for unit, vector in zip(units, vectors)])
        probabilities = softmax(x @ self.weights + self.bias)
        results = []
        for index, (unit, item) in enumerate(zip(units, encoded)):
            ranking = np.argsort(probabilities[index])[::-1]
            winner = int(ranking[0])
            label, source = self.names[winner], self.head_kind + '_head'
            reason = 'The trained classifier favors this intent from wording and context.'
            cue_label, cue_reason, strong = rule(unit, self.names)
            corrected = self.exact.get(normalized(unit['text']))
            closest = None
            if corrected:
                label, source, reason = corrected, 'correction', 'You previously assigned this note to this category.'
            elif self.correction_vectors is not None:
                distances = self.correction_vectors @ item.vector
                nearest = int(distances.argmax())
                if float(distances[nearest]) >= 0.86:
                    closest = self.corrections[nearest]
                    label, source, reason = closest['label'], 'similar_correction', 'Very similar to a note you classified.'
            has_context = bool(unit.get('context') or unit.get('neighbors'))
            if source not in {'correction', 'similar_correction'} and cue_label and (strong or self.head_kind == 'starter' and not has_context):
                label, source, reason = cue_label, 'format' if strong else 'intent', cue_reason
            margin = float(probabilities[index, ranking[0]] - probabilities[index, ranking[1]])
            review = []
            if source.endswith('_head') and margin < 0.12:
                review.append('The top two category scores are close.')
            if source.endswith('_head') and float(semantic[index].max()) < 0.25 and 'thoughts' in self.names:
                label, source, reason = 'thoughts', 'fallback', 'No clear action, question, proposal, link, or code; kept as a thought.'
                # A generic thought is a valid category, not an automatic review queue.
                review = []
            if item.truncated:
                review.append('Only the first 256 tokens were encoded; split this long note for a better match.')
            if has_context and source.endswith('_head'):
                reason += ' The optional context contributes 15% of the embedding.'
            label_index = self.names.index(label)
            score = float(probabilities[index, label_index])
            start, end = self.spans[label_index]
            reference = start + int(similarities[index, start:end].argmax())
            highlights = []
            if unit.get('kind') != 'code':
                candidates = [(float(v @ self.reference_vectors[reference]), left, right) for left, right, v in item.words
                              if right - left >= 4 and any(c.isalpha() for c in unit['text'][left:right])]
                highlights = sorted([{'start': left, 'end': right} for _, left, right in sorted(candidates, reverse=True)[:3]], key=lambda h: h['start'])
            dates = extract_deadline(unit['text'], anchor)
            results.append({**unit, 'label': label, 'label_source': source, 'reason': reason,
                            'score': round(score, 4), 'scores': {name: round(float(p), 4) for name, p in zip(self.names, probabilities[index])},
                            'score_kind': 'uncalibrated head probability', 'matched_example': closest['text'] if closest else self.references[reference],
                            'highlights': highlights, 'date': dates['parsed_date'] if label == 'todos' else None,
                            'needs_review': bool(review), 'review_reasons': review})
        return results

    def save_classes(self, classes, path=LOCAL_CLASSES):
        with self.lock:
            self.set_classes(classes)
            self.set_corrections(getattr(self, 'corrections', []))
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.classes, indent=2, ensure_ascii=False) + '\n')
            temporary.replace(path)
