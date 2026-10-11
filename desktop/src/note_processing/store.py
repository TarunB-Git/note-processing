"""Persist organized notes, corrections, and workspace settings in SQLite."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from note_processing.paths import APP_HOME
from note_processing.signals import normalized


class Store:
    def __init__(self, path=None):
        self.path = Path(path or APP_HOME / 'notes.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, data TEXT NOT NULL, position INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS corrections (text_key TEXT PRIMARY KEY, text TEXT NOT NULL, label TEXT NOT NULL, updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL);
            ''')
            columns = {row[1] for row in db.execute('PRAGMA table_info(corrections)')}
            if 'details' not in columns:
                db.execute("ALTER TABLE corrections ADD COLUMN details TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def connection(self):
        with self.lock, sqlite3.connect(self.path) as db:
            yield db

    def notes(self):
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT data FROM notes ORDER BY position, id')]

    def put(self, notes):
        with self.connection() as db:
            position = db.execute('SELECT COALESCE(MAX(position), -1)+1 FROM notes').fetchone()[0]
            grouped = {identifier for row in db.execute('SELECT data FROM notes')
                       for identifier in json.loads(row[0]).get('grouped_ids', [])}
            for note in notes:
                if note['id'] in grouped:
                    continue
                existing = db.execute('SELECT data FROM notes WHERE id=?', (note['id'],)).fetchone()
                if existing:
                    old = json.loads(existing[0])
                    if old.get('manual'):
                        for key in ('label', 'manual', 'reason', 'label_source', 'needs_review', 'review_reasons'):
                            note[key] = old.get(key)
                    note['done'] = old.get('done', note.get('done', False))
                    db.execute('UPDATE notes SET data=? WHERE id=?', (json.dumps(note, ensure_ascii=False), note['id']))
                else:
                    db.execute('INSERT INTO notes VALUES (?,?,?)', (note['id'], json.dumps(note, ensure_ascii=False), position))
                    position += 1

    def update(self, ids, changes, teach=False):
        changed = []
        with self.connection() as db:
            previous = []
            for note_id in ids:
                row = db.execute('SELECT data FROM notes WHERE id=?', (note_id,)).fetchone()
                if not row:
                    raise ValueError('Note no longer exists')
                note = json.loads(row[0])
                if teach:
                    correction = db.execute('SELECT text_key,text,label,updated,details FROM corrections WHERE text_key=?', (normalized(note['text']),)).fetchone()
                    previous.append({'note': dict(note), 'correction': correction})
                note.update(changes)
                if teach:
                    note.update(manual=True, label_source='manual', reason='You assigned this category.', needs_review=False, review_reasons=[])
                    details = {key: note[key] for key in ('context', 'neighbors', 'kind', 'checkbox') if key in note}
                    db.execute('INSERT INTO corrections VALUES (?,?,?,?,?) ON CONFLICT(text_key) DO UPDATE SET text=excluded.text,label=excluded.label,updated=excluded.updated,details=excluded.details',
                               (normalized(note['text']), note['text'], note['label'], datetime.now(timezone.utc).isoformat(), json.dumps(details)))
                db.execute('UPDATE notes SET data=? WHERE id=?', (json.dumps(note, ensure_ascii=False), note_id))
                changed.append(note)
            if teach:
                db.execute('INSERT INTO history(data) VALUES (?)', (json.dumps(previous),))
                db.execute('DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY id DESC LIMIT 30)')
        return changed

    def undo_move(self):
        with self.connection() as db:
            row = db.execute('SELECT id,data FROM history ORDER BY id DESC LIMIT 1').fetchone()
            if not row:
                raise ValueError('No move to undo')
            for previous in reversed(json.loads(row[1])):
                if previous.get('created'):
                    db.execute('DELETE FROM notes WHERE id=?', (previous['created'],))
                    db.execute('DELETE FROM corrections WHERE text_key=?', (previous['text_key'],))
                    if previous.get('correction'):
                        db.execute('INSERT INTO corrections VALUES (?,?,?,?,?)', previous['correction'])
                    continue
                note = previous['note']
                if 'position' in previous:
                    db.execute('INSERT OR REPLACE INTO notes VALUES (?,?,?)', (note['id'], json.dumps(note, ensure_ascii=False), previous['position']))
                else:
                    db.execute('UPDATE notes SET data=? WHERE id=?', (json.dumps(note, ensure_ascii=False), note['id']))
                key = normalized(note['text'])
                db.execute('DELETE FROM corrections WHERE text_key=?', (key,))
                if previous['correction']:
                    db.execute('INSERT INTO corrections VALUES (?,?,?,?,?)', previous['correction'])
            db.execute('DELETE FROM history WHERE id=?', (row[0],))

    def group(self, ids, note):
        with self.connection() as db:
            previous, positions = [], []
            for identifier in ids:
                row = db.execute('SELECT data,position FROM notes WHERE id=?', (identifier,)).fetchone()
                if not row:
                    raise ValueError('Note no longer exists')
                original = json.loads(row[0])
                correction = db.execute('SELECT text_key,text,label,updated,details FROM corrections WHERE text_key=?', (normalized(original['text']),)).fetchone()
                previous.append({'note': original, 'position': row[1], 'correction': correction})
                positions.append(row[1])
            key = normalized(note['text'])
            correction = db.execute('SELECT text_key,text,label,updated,details FROM corrections WHERE text_key=?', (key,)).fetchone()
            previous.append({'created': note['id'], 'text_key': key, 'correction': correction})
            note.update(manual=True, label_source='manual', reason='You grouped these fragments and assigned the category.', needs_review=False, review_reasons=[])
            for identifier in ids:
                db.execute('DELETE FROM notes WHERE id=?', (identifier,))
            db.execute('INSERT INTO notes VALUES (?,?,?)', (note['id'], json.dumps(note, ensure_ascii=False), min(positions)))
            details = {key: note[key] for key in ('context', 'neighbors', 'kind', 'checkbox') if key in note}
            db.execute('INSERT INTO corrections VALUES (?,?,?,?,?) ON CONFLICT(text_key) DO UPDATE SET text=excluded.text,label=excluded.label,updated=excluded.updated,details=excluded.details',
                       (key, note['text'], note['label'], datetime.now(timezone.utc).isoformat(), json.dumps(details)))
            db.execute('INSERT INTO history(data) VALUES (?)', (json.dumps(previous),))

    def corrections(self):
        with self.connection() as db:
            return [{**json.loads(details), 'text': text, 'label': label} for text, label, details in db.execute('SELECT text,label,details FROM corrections ORDER BY updated')]

    def get(self, key, default=None):
        with self.connection() as db:
            row = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.connection() as db:
            db.execute('INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, json.dumps(value)))
