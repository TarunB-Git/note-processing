import tempfile
import unittest
from pathlib import Path

from note_processing.documents import split_document
from note_processing.outputs import output_zip, write_outputs
from note_processing.store import Store


class NoteIntegrityTests(unittest.TestCase):
    def test_import_keeps_links_code_checkboxes_and_locations(self):
        units = split_document('# Notes\n\n- [x] Send report\n[Docs](https://example.org)\n```python\na = 1\n```\n', 'notes.md')
        self.assertEqual([u['line'] for u in units], [3, 4, 5])
        self.assertTrue(units[0]['done'])
        self.assertIn('https://example.org', units[1]['text'])
        self.assertEqual(units[2]['text'], '```python\na = 1\n```')
        self.assertEqual(units[2]['kind'], 'code')

    def test_export_checks_every_target_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / 'existing.md'
            existing.write_text('Keep this edit')
            rows = [{'path': 'new.md', 'content': 'New'}, {'path': 'existing.md', 'content': 'Replacement'}]
            with self.assertRaises(ValueError):
                write_outputs(root, rows)
            self.assertFalse((root / 'new.md').exists())
            self.assertEqual(existing.read_text(), 'Keep this edit')
            with self.assertRaises(ValueError):
                write_outputs(root, [{'path': '../outside.md', 'content': 'No'}])
            with self.assertRaises(ValueError):
                write_outputs(root, [{'path': 'existing.md', 'content': 'No'}], overwrite=True, protected=[existing])
            with self.assertRaises(ValueError):
                output_zip([{'path': '../outside.md', 'content': 'No'}])

    def test_move_reimport_and_undo_preserve_user_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'notes.sqlite3')
            note = {**split_document('Why this project?')[0], 'label': 'questions', 'needs_review': False}
            store.put([dict(note)])
            store.update([note['id']], {'label': 'todos'}, teach=True)
            store.put([dict(note)])
            self.assertEqual(store.notes()[0]['label'], 'todos')
            self.assertEqual(store.corrections()[0]['label'], 'todos')
            store.undo_move()
            self.assertEqual(store.notes()[0]['label'], 'questions')
            self.assertFalse(store.corrections())

    def test_group_and_undo_restore_fragments_without_duplicate_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'notes.sqlite3')
            notes = [{**n, 'label': 'thoughts'} for n in split_document('First fragment\nSecond fragment')]
            store.put(notes)
            grouped = {**notes[0], 'id': 'group', 'text': 'First fragment\nSecond fragment', 'grouped_ids': [n['id'] for n in notes]}
            store.group([n['id'] for n in notes], grouped)
            store.put(notes)
            self.assertEqual(len(store.notes()), 1)
            store.undo_move()
            self.assertEqual([n['text'] for n in store.notes()], ['First fragment', 'Second fragment'])
            self.assertFalse(store.corrections())


if __name__ == '__main__':
    unittest.main()
