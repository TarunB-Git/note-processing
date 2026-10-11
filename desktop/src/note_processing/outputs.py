"""Render category files and protect existing files from accidental replacement."""

import hashlib
import io
import re
import tempfile
import zipfile
from pathlib import Path


def render_category(category, notes):
    lines = ['# ' + category.get('title', category['name']), '']
    for note in notes:
        text = note['text']
        if category['name'] == 'todos':
            parts = text.splitlines()
            due = f" · due {note['date']}" if note.get('date') else ''
            lines.append(f"- [{'x' if note.get('done') else ' '}] {parts[0]}{due}")
            lines.extend('  ' + part for part in parts[1:])
        elif category['name'] == 'junk' or note.get('kind') == 'code':
            if note.get('kind') == 'code' and re.match(r'^\s*(?:`{3,}|~{3,})', text):
                lines.extend([text, ''])
            else:
                fence = '`' * max(3, max([len(s) + 1 for s in re.findall(r'`+', text)] or [3]))
                lines.extend([fence, text, fence, ''])
        else:
            parts = text.splitlines()
            lines.append('- ' + parts[0])
            lines.extend('  ' + part for part in parts[1:])
    return '\n'.join(lines).rstrip() + '\n'


def render_outputs(classes, notes, targets=None):
    targets = targets or {}
    rows = []
    for category in classes:
        selected = [n for n in notes if n['label'] == category['name']]
        rows.append({'category': category['name'], 'title': category.get('title', category['name']),
                     'path': targets.get(category['name'], category['name'] + '.md'),
                     'count': len(selected), 'content': render_category(category, selected)})
    return rows


def safe_target(root, value):
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ValueError('Output filenames must be relative to the output folder')
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root or path.suffix.lower() not in {'.md', '.txt'}:
        raise ValueError('Choose a .md or .txt path inside the output folder')
    return path


def write_outputs(root, rows, previous=None, overwrite=False, protected=()):
    root = Path(root).resolve()
    previous = previous or {}
    protected = {str(Path(p).resolve()) for p in protected}
    planned, hashes = [], {}
    for row in rows:
        path = safe_target(root, row['path'])
        if path in [p for p, _ in planned]:
            raise ValueError('Each category needs a different output filename')
        if str(path) in protected:
            raise ValueError('An output path matches an imported source file')
        if path.exists():
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            if current != previous.get(str(path)) and not overwrite:
                raise ValueError(f'{row["path"]} already exists or was edited. Preview it and enable replacement to save.')
        planned.append((path, row['content'].encode('utf8')))
    for path, content in planned:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
    return hashes


def output_zip(rows):
    result = io.BytesIO()
    seen = set()
    with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED) as archive:
        for row in rows:
            name = row['path']
            safe_target(Path('/exports'), name)
            if name in seen:
                raise ValueError('Each category needs a different output filename')
            seen.add(name)
            archive.writestr(name, row['content'])
    return result.getvalue()
