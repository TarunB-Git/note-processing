"""Coordinate imports, corrections, classification jobs, and saved outputs."""

import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from note_processing.deeper import classify_deeper, local_models
from note_processing.context import attach_context
from note_processing.documents import MAX_FILES, MAX_UNITS, read_file, scan_folder, split_document
from note_processing.jobs import Jobs
from note_processing.learning import train_classifier
from note_processing.outputs import render_outputs, safe_target, write_outputs
from note_processing.paths import APP_HOME
from note_processing.store import Store


class Workspace:
    def __init__(self, classifier, root, output, store=None):
        self.classifier = classifier
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        if not self.root.is_dir():
            raise ValueError('The input folder does not exist')
        self.store = store or Store()
        self.classifier.set_corrections(self.store.corrections())
        self.jobs = Jobs()

    def selected_path(self, value):
        if not isinstance(value, str):
            raise ValueError('Folder path must be text')
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Choose a path inside the input folder')
        return path

    def snapshot(self):
        return {'notes': self.store.notes(), 'classes': self.classifier.classes, 'root': str(self.root),
                'output_root': str(self.output), 'targets': self.store.get('targets', {}),
                'context': self.store.get('context', ''), 'neighbor_words': self.store.get('neighbor_words', 0),
                'learning': self.learning_status(), 'saved': self.store.get('saved', {})}

    def learning_status(self):
        corrections = self.store.corrections()
        report_path = APP_HOME / 'training/report.json'
        return {'confirmed': len(corrections), 'counts': dict(Counter(r['label'] for r in corrections)),
                'head': self.classifier.head_kind, 'report': json.loads(report_path.read_text()) if report_path.exists() else None}

    def import_notes(self, data, progress):
        mode = data.get('mode', 'lines')
        anchor = date.fromisoformat(data['anchor']) if data.get('anchor') else date.today()
        warnings, protected = [], []
        if data.get('kind') == 'folder':
            path = self.selected_path(data.get('path', ''))
            if path.is_file():
                units = read_file(path, path.relative_to(self.root).as_posix(), mode)
                protected = [str(path)]
            else:
                scan = scan_folder(path, mode, recursive=data.get('recursive', True))
                units, warnings = scan['units'], scan['warnings']
                protected = [str(path / name) for name in scan['files']]
        else:
            documents = data.get('documents', [{'name': 'Pasted notes', 'text': data.get('text', '')}])
            if not isinstance(documents, list) or len(documents) > MAX_FILES:
                raise ValueError('Choose up to 250 files')
            units = []
            for document in documents:
                if not isinstance(document, dict) or not isinstance(document.get('name'), str):
                    raise ValueError('Each document needs a name and text')
                units.extend(split_document(document.get('text', ''), document['name'], mode))
        if len(units) > MAX_UNITS:
            raise ValueError('Import exceeds 10,000 notes')
        context = data.get('context', self.store.get('context', ''))
        neighbor_words = data.get('neighbor_words', self.store.get('neighbor_words', 0))
        units = attach_context(units, context, neighbor_words)
        self.store.set('context', context)
        self.store.set('neighbor_words', neighbor_words)
        if len(self.store.notes()) + len(units) > MAX_UNITS:
            existing = {n['id'] for n in self.store.notes()}
            if len(existing | {n['id'] for n in units}) > MAX_UNITS:
                raise ValueError('The workspace holds up to 10,000 notes')
        self.store.set('protected_sources', list(set(self.store.get('protected_sources', []) + protected)))
        for offset in range(0, len(units), 32):
            progress(offset / max(len(units), 1) * 100, f'Sorting {offset} / {len(units)} notes')
            self.store.put(self.classifier.classify(units[offset:offset + 32], anchor))
        return {'imported': len(units), 'warnings': warnings}

    def move(self, data):
        ids = self.validate_ids(data.get('ids'))
        label = data.get('label')
        if label not in self.classifier.names:
            raise ValueError('Choose an existing category')
        self.store.update(ids, {'label': label}, teach=True)
        from note_processing.dates import extract_deadline
        for note in self.store.notes():
            if note['id'] in ids:
                self.store.update([note['id']], {'date': extract_deadline(note['text'])['parsed_date'] if label == 'todos' else None})
        self.classifier.set_corrections(self.store.corrections())
        return {'learning': self.learning_status()}

    def group(self, data):
        import hashlib
        from note_processing.dates import extract_deadline
        ids = self.validate_ids(data.get('ids'))
        if len(ids) < 2:
            raise ValueError('Select at least two fragments to group')
        label = data.get('label')
        if label not in self.classifier.names:
            raise ValueError('Choose a category for the grouped note')
        notes = [n for n in self.store.notes() if n['id'] in ids]
        text = '\n'.join(n['text'] for n in notes)
        unit = {**notes[0], 'id': hashlib.sha256('\0'.join(n['id'] for n in notes).encode()).hexdigest()[:24],
                'text': text, 'raw': '\n'.join(n.get('raw', n['text']) for n in notes),
                'kind': 'text', 'grouped_ids': [identifier for n in notes for identifier in [n['id'], *n.get('grouped_ids', [])]],
                'source_lines': [{'source': n['source'], 'line': n['line']} for n in notes], 'checkbox': False}
        prediction = self.classifier.classify([unit])[0]
        prediction['label'] = label
        prediction['date'] = extract_deadline(text)['parsed_date'] if label == 'todos' else None
        self.store.group(ids, prediction)
        self.classifier.set_corrections(self.store.corrections())
        return {'id': unit['id']}

    def validate_ids(self, ids):
        if not isinstance(ids, list) or not ids or len(ids) > MAX_UNITS or any(not isinstance(s, str) for s in ids):
            raise ValueError('Select one or more notes')
        known = {n['id'] for n in self.store.notes()}
        if any(identifier not in known for identifier in ids):
            raise ValueError('A selected note no longer exists')
        return list(dict.fromkeys(ids))

    def update_note(self, data):
        ids = self.validate_ids(data.get('ids'))
        if not isinstance(data.get('done'), bool):
            raise ValueError('Task completion must be true or false')
        return self.store.update(ids, {'done': data['done']})

    def reclassify(self, data, progress):
        selected = data.get('ids')
        if selected is not None:
            selected = set(self.validate_ids(selected))
        all_notes = attach_context(self.store.notes(), data.get('context', self.store.get('context', '')),
                                   data.get('neighbor_words', self.store.get('neighbor_words', 0)))
        if 'context' in data:
            self.store.set('context', data['context'])
        if 'neighbor_words' in data:
            self.store.set('neighbor_words', data['neighbor_words'])
        notes = [n for n in all_notes if (selected is None or n['id'] in selected)
                 and (data.get('include_confirmed', False) or not n.get('manual'))]
        mode = data.get('engine', 'small')
        if mode not in {'small', 'local', 'thinking'}:
            raise ValueError('Choose small, local, or thinking classification')
        if mode != 'small':
            model = data.get('model', 'qwen3:1.7b')
            if model not in local_models():
                raise ValueError(f'Local model missing. Run: ollama pull {model}')
        corrections = self.classifier.corrections
        for offset, note in enumerate(notes):
            progress(offset / max(len(notes), 1) * 100, f'Reclassifying {offset} / {len(notes)} notes')
            if mode == 'small':
                prediction = self.classifier.classify([note])[0]
            else:
                vector = self.classifier.encoder.encode([note['text']])[0].vector
                references = []
                if self.classifier.correction_vectors is not None:
                    indexes = np.argsort(self.classifier.correction_vectors @ vector)[::-1][:6]
                    references = [corrections[i] for i in indexes]
                prediction = {**note, **classify_deeper(note, self.classifier.classes, references, model, mode == 'thinking')}
                from note_processing.dates import extract_deadline
                prediction['date'] = extract_deadline(note['text'])['parsed_date'] if prediction['label'] == 'todos' else None
            prediction.pop('manual', None)
            # Explicitly including confirmed notes removes the pin; correction memory remains.
            if data.get('include_confirmed') and note.get('manual'):
                self.store.update([note['id']], {'manual': False})
            self.store.put([prediction])
        return {'reclassified': len(notes)}

    def outputs(self):
        rows = render_outputs(self.classifier.classes, self.store.notes(), self.store.get('targets', {}))
        saved = self.store.get('saved', {})
        for row in rows:
            path = safe_target(self.output, row['path'])
            row['exists'] = path.exists()
            row['saved_content'] = path.read_text(encoding='utf8') if path.exists() and path.stat().st_size <= 4 * 1024 * 1024 else None
            row['last_saved'] = saved.get('time')
        return rows

    def configure_outputs(self, targets):
        if not isinstance(targets, dict) or any(k not in self.classifier.names for k in targets):
            raise ValueError('Provide filenames for the existing categories')
        paths = [safe_target(self.output, v) for v in targets.values()]
        if len(set(paths)) != len(paths):
            raise ValueError('Each category needs a different output filename')
        self.store.set('targets', targets)

    def save_outputs(self, overwrite=False):
        from datetime import datetime, timezone
        rows = self.outputs()
        hashes = write_outputs(self.output, rows, self.store.get('output_hashes', {}), overwrite,
                               self.store.get('protected_sources', []))
        self.store.set('output_hashes', hashes)
        saved = {'time': datetime.now(timezone.utc).isoformat(), 'files': list(hashes)}
        self.store.set('saved', saved)
        return saved

    def train(self, progress):
        with self.classifier.lock:
            return train_classifier(self.classifier, self.store.corrections(), APP_HOME / 'training', progress)
