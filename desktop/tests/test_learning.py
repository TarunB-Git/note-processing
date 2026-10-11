import unittest
import tempfile
from pathlib import Path

from note_processing.context import attach_context
from note_processing.learning import split_labels
from note_processing.signals import normalized


class LearningBoundaryTests(unittest.TestCase):
    def test_adding_examples_keeps_existing_holdout_assignments(self):
        records = [{'text': f'{label} example {i}', 'label': label} for label in ['ideas', 'todos'] for i in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.json'
            train, test = split_labels(records, ['ideas', 'todos'], path)
            added = records + [{'text': f'{label} new {i}', 'label': label} for label in ['ideas', 'todos'] for i in range(5)]
            new_train, new_test = split_labels(added, ['ideas', 'todos'], path)
            self.assertTrue({r['text'] for r in train}.issubset({r['text'] for r in new_train}))
            self.assertTrue({r['text'] for r in test}.issubset({r['text'] for r in new_test}))

    def test_duplicate_labels_never_cross_the_holdout_boundary(self):
        records = [{'text': f'{label} example {i}', 'label': label} for label in ['ideas', 'todos'] for i in range(10)]
        records.append({'text': ' IDEAS   example 0 ', 'label': 'ideas'})
        train, test = split_labels(records, ['ideas', 'todos'])
        self.assertEqual(len(train) + len(test), 20)
        self.assertFalse({normalized(r['text']) for r in train} & {normalized(r['text']) for r in test})
        self.assertEqual(len(test), 4)

    def test_context_is_bounded_and_stays_in_its_source(self):
        units = [{'id': 'before', 'source': 'a', 'text': 'before ' * 600},
                 {'id': 'middle', 'source': 'a', 'text': 'Target'},
                 {'id': 'after', 'source': 'a', 'text': 'after ' * 600},
                 {'id': 'other', 'source': 'b', 'text': 'Private other document'}]
        result = attach_context(units, 'These are story ideas.', 1000)
        self.assertLessEqual(len(result[1]['neighbors'].split()), 1002)
        self.assertNotIn('Private', result[1]['neighbors'])
        self.assertNotIn('Target', result[1]['neighbors'])
        with self.assertRaises(ValueError):
            attach_context(units, 'word ' * 1001)


if __name__ == '__main__':
    unittest.main()
