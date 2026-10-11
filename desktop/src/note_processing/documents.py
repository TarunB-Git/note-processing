"""Keep note boundaries, links, code, and original source locations."""

import hashlib
import os
import re
from pathlib import Path

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_FILES = 250
MAX_UNITS = 10000
EXTENSIONS = {'.txt', '.md', '.markdown'}
SKIP_DIRECTORIES = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'models', 'organized', 'archive'}


def note_id(source, line, raw):
    return hashlib.sha256(f'{source}\0{line}\0{raw}'.encode()).hexdigest()[:24]


def split_document(text, source='Pasted notes', mode='lines'):
    if not isinstance(text, str):
        raise ValueError('Document text must be a string')
    if mode not in {'lines', 'paragraphs', 'sentences'}:
        raise ValueError('Choose lines, paragraphs, or sentences')
    if len(text.encode('utf8')) > MAX_FILE_BYTES:
        raise ValueError(f'{source}: file exceeds 4 MB')
    units, pending = [], []
    start, fence, code = 1, None, []
    markdown = Path(source).suffix.lower() in {'.md', '.markdown'}
    frontmatter = False

    def add(raw, line, kind='text'):
        if not raw.strip():
            return
        checkbox = re.match(r'^\s*[-*+]\s+\[([ xX])\]\s*', raw)
        content = raw.strip() if kind == 'code' else re.sub(r'^\s*(?:[-*+]\s+(?:\[[ xX]\]\s*)?|\d+[.)]\s+|>\s*)', '', raw).strip()
        chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content) if mode == 'sentences' and kind != 'code' and '://' not in content else [content]
        for offset, chunk in enumerate(chunks):
            units.append({'id': note_id(source, f'{line}:{offset}', raw), 'text': chunk,
                          'raw': raw, 'source': source, 'line': line, 'kind': kind,
                          'checkbox': bool(checkbox), 'done': bool(checkbox and checkbox[1].lower() == 'x')})
        if len(units) > MAX_UNITS:
            raise ValueError('Import exceeds 10,000 notes')

    def flush():
        if pending:
            add('\n'.join(pending), start)
            pending.clear()

    for number, line in enumerate(text.splitlines(), 1):
        if markdown and number == 1 and line.strip() == '---':
            frontmatter = True
            continue
        if frontmatter:
            if line.strip() == '---':
                frontmatter = False
            continue
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence:
            code.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                add('\n'.join(code), start, 'code')
                code, fence = [], None
            continue
        if marker:
            flush()
            start, fence, code = number, marker[1], [line]
            continue
        if markdown and (re.match(r'^\s*#{1,6}\s', line) or re.fullmatch(r'\s*[-*_]{3,}\s*', line)):
            flush()
            continue
        if not line.strip():
            flush()
            continue
        if mode == 'paragraphs':
            if re.match(r'^\s*(?:[-*+]\s|\d+[.)]\s)', line):
                flush()
            if not pending:
                start = number
            pending.append(line)
        else:
            add(line, number)
    flush()
    if code:
        add('\n'.join(code), start, 'code')
    return units


def read_file(path, source=None, mode='lines'):
    path = Path(path)
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError('Choose a .txt or .md file')
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f'{path.name}: file exceeds 4 MB')
    try:
        return split_document(path.read_text(encoding='utf-8-sig'), source or str(path), mode)
    except UnicodeDecodeError as error:
        raise ValueError(f'{path.name}: expected UTF-8 text') from error


def scan_folder(folder, mode='lines', recursive=True):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError('Folder does not exist')
    units, files, warnings, total = [], [], [], 0
    for current, directories, names in os.walk(folder, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in SKIP_DIRECTORIES and not d.startswith('.') and not (Path(current) / d).is_symlink())
        for name in sorted(names):
            path = Path(current) / name
            if name.startswith('.') or path.is_symlink() or path.suffix.lower() not in EXTENSIONS:
                continue
            if len(files) >= MAX_FILES:
                raise ValueError('Import exceeds 250 files')
            total += path.stat().st_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError('Import exceeds 32 MB')
            source = path.relative_to(folder).as_posix()
            files.append(source)
            try:
                units.extend(read_file(path, source, mode))
            except (OSError, ValueError) as error:
                warnings.append(str(error))
            if len(units) > MAX_UNITS:
                raise ValueError('Import exceeds 10,000 notes')
        if not recursive:
            break
    return {'units': units, 'files': files, 'warnings': warnings}
