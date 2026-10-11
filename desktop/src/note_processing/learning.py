"""Train and evaluate a small classifier on confirmed labels."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from note_processing.signals import features, normalized
from note_processing.context import contextual_vector


def encoder_key(encoder):
    path = encoder.directory / 'model.onnx'
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classes_key(classes):
    return hashlib.sha256(json.dumps(classes, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def matrices(encoder, records, names):
    vectors = []
    for offset in range(0, len(records), 32):
        batch = records[offset:offset + 32]
        vectors.extend(features(record, contextual_vector(encoder, record, item.vector)) for record, item in zip(batch, encoder.encode([r['text'] for r in batch])))
    return np.stack(vectors), np.asarray([names.index(r['label']) for r in records])


def softmax(logits):
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def fit_head(x, y, names, steps=350):
    weights = np.zeros((x.shape[1], len(names)), dtype=np.float32)
    bias = np.zeros(len(names), dtype=np.float32)
    targets = np.eye(len(names), dtype=np.float32)[y]
    counts = np.bincount(y, minlength=len(names))
    sample_weights = len(y) / (len(names) * np.maximum(counts[y], 1))
    rng = np.random.default_rng(42)
    for step in range(steps):
        indexes = rng.choice(len(y), min(len(y), 256), replace=False)
        batch = x[indexes]
        error = (softmax(batch @ weights + bias) - targets[indexes]) * sample_weights[indexes, None]
        rate = 0.3 / (1 + step / 200)
        weights -= rate * (batch.T @ error / len(indexes) + 0.003 * weights)
        bias -= rate * error.mean(axis=0)
    return weights, bias


def metrics(actual, predicted, names):
    matrix = [[sum(a == left and p == right for a, p in zip(actual, predicted)) for right in names] for left in names]
    per_class = []
    for index, name in enumerate(names):
        tp = matrix[index][index]
        precision = tp / max(sum(row[index] for row in matrix), 1)
        recall = tp / max(sum(matrix[index]), 1)
        per_class.append({'label': name, 'precision': round(precision, 3), 'recall': round(recall, 3),
                          'f1': round(2 * precision * recall / max(precision + recall, 1e-9), 3), 'support': sum(matrix[index])})
    represented = [row['f1'] for row in per_class if row['support']]
    return {'accuracy': round(sum(a == p for a, p in zip(actual, predicted)) / max(len(actual), 1), 3),
            'macro_f1': round(sum(represented) / max(len(represented), 1), 3), 'per_class': per_class, 'confusion': matrix}


def split_labels(records, names, assignment_path=None):
    unique = {}
    for record in records:
        if record['label'] in names:
            unique[normalized(record['text'])] = record
    groups = {name: [] for name in names}
    for record in unique.values():
        groups[record['label']].append(record)
    missing = [name for name, rows in groups.items() if len(rows) < 6]
    if missing:
        raise ValueError('Confirm at least 6 different notes per category before evaluated training. Still needed: ' + ', '.join(missing))
    train, test = [], []
    assignments = {}
    path = Path(assignment_path) if assignment_path else None
    if path and path.exists():
        assignments = json.loads(path.read_text())
    rng = np.random.default_rng(42)
    for name in names:
        rows = sorted(groups[name], key=lambda r: normalized(r['text']))
        if not path:
            indexes = rng.permutation(len(rows))
            held = max(1, int(len(rows) * 0.2))
            test.extend(rows[i] for i in indexes[:held])
            train.extend(rows[i] for i in indexes[held:])
            continue
        unknown = [r for r in rows if normalized(r['text']) not in assignments]
        if unknown:
            indexes = rng.permutation(len(unknown))
            held = max(1, int(len(unknown) * 0.2)) if not any(assignments.get(normalized(r['text'])) == 'test' for r in rows) else int(len(unknown) * 0.2)
            for rank, index in enumerate(indexes):
                assignments[normalized(unknown[index]['text'])] = 'test' if rank < held else 'train'
        class_test = [r for r in rows if assignments[normalized(r['text'])] == 'test']
        class_train = [r for r in rows if assignments[normalized(r['text'])] == 'train']
        if not class_test or not class_train:
            raise ValueError(f'{name}: confirm more new examples to keep both training and held-out notes')
        test.extend(class_test)
        train.extend(class_train)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(assignments, ensure_ascii=False))
        temporary.replace(path)
    return train, test


def train_classifier(classifier, corrections, directory, progress=lambda *args: None):
    names = [c['name'] for c in classifier.classes]
    train, test = split_labels(corrections, names, Path(directory) / 'split.json')
    progress(10, 'Encoding confirmed notes')
    seeds = classifier.seed_records()
    held_text = {normalized(r['text']) for r in test}
    seeds = [r for r in seeds if normalized(r['text']) not in held_text]
    evaluation_classes = [{**c, 'examples': [s for s in c['examples'] if normalized(s) not in held_text]}
                          for c in classifier.classes]
    x, y = matrices(classifier.encoder, seeds + train, names)
    weights, bias = fit_head(x, y, names)
    from note_processing.classifier import Classifier
    candidate = Classifier(evaluation_classes, encoder=classifier.encoder, corrections=train, load_head=False)
    candidate.install_head(weights, bias)
    predicted = [r['label'] for r in candidate.classify(test)]
    # The reference run excludes held-out corrections and never sees their labels.
    baseline = Classifier(evaluation_classes, encoder=classifier.encoder, corrections=train, load_head=False)
    baseline_predictions = [r['label'] for r in baseline.classify(test)]
    report = {'kind': 'head', 'created': datetime.now(timezone.utc).isoformat(), 'train_count': len(train),
              'baseline_kind': 'starter with training-split correction memory',
              'test_count': len(test), 'counts': dict(Counter(r['label'] for r in corrections)),
              'candidate': metrics([r['label'] for r in test], predicted, names),
              'baseline': metrics([r['label'] for r in test], baseline_predictions, names),
              'mistakes': [{'text': row['text'], 'expected': row['label'], 'predicted': label}
                           for row, label in zip(test, predicted) if row['label'] != label]}
    report['adopted'] = report['candidate']['macro_f1'] >= report['baseline']['macro_f1']
    if report['adopted']:
        progress(70, 'Saving the evaluated head; held-out labels stay out of training')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'head.tmp.npz'
        np.savez(temporary, weights=weights, bias=bias, names=np.array(names), encoder_key=np.array(encoder_key(classifier.encoder)),
                 classes_key=np.array(classes_key(classifier.classes)))
        temporary.replace(directory / 'head.npz')
        classifier.install_head(weights, bias)
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    progress(100, 'Training complete' if report['adopted'] else 'Baseline retained; candidate scored lower')
    return report
