"""Observable intent and format cues; shared by inference and training."""

import hashlib
import re

URL = re.compile(r'https?://[^\s<>]+|www\.[^\s<>]+', re.I)
QUESTION = re.compile(r'^(?:who|what(?!\s+if\b)|when|where|why|how|which)\b|^(?:is|are|was|were|do|does|did|can|could|will|would|should|has|have)\s+(?:i|you|he|she|it|we|they|there|this|that|the|a|an)\b', re.I)
IDEA = re.compile(r'\b(?:what if|we could|maybe (?:we|i|try|build|add|start)|idea(?: for|:)|could (?:build|make|add|create)|it would be (?:nice|cool|useful)|how about)\b', re.I)
INTENT = re.compile(r'^(?:todo\b|to.do\b|remind(?:er| me)\b|remember to\b|(?:i|we) (?:need|have|want) to\b|(?:please\s+)?(?:buy|send|email|call|book|schedule|finish|fix|pay|submit|write|read|review|update|clean|check|make|prepare|apply|order|pick up|follow up|download|install|upload|cancel|renew|contact|ask|get|collect|learn|watch|take|do)\b)', re.I)
CODE = re.compile(r'^\s*(?:def \w+\(|class \w+[:(]|(?:from \S+ )?import \S+|(?:const|let|var) \w+\s*=|function \w*\(|(?:SELECT|INSERT|CREATE TABLE)\b|Traceback \(|\$ (?:pip|npm|git)|[\w.]+\s*=\s*[^=]|[{}\[\];]+\s*$)|=>|</?\w+[^>]*>', re.I | re.M)
ATTACHMENT = re.compile(r'\b(?:file attached|image omitted|media omitted|IMG-\d+|VID-\d+)\b|^\s*(?:asdf|qwerty|test test|\.{3,})\s*$', re.I)


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def cues(unit):
    text = unit['text'].strip()
    return [bool(unit.get('kind') == 'code' or CODE.search(text)), bool(ATTACHMENT.search(text)),
            bool(URL.search(text)), bool(unit.get('checkbox')), bool(IDEA.search(text)),
            bool(QUESTION.search(text) or text.endswith('?')), bool(INTENT.search(text)),
            bool(re.search(r'\b(?:tomorrow|today|by monday|deadline|due)\b', text, re.I)),
            min(len(text) / 500, 1), min(len(text.split()) / 60, 1), text.isupper(),
            bool(re.search(r'\b(?:think|feel|realized|noticed|wonder|remember when)\b', text, re.I))]


def rule(unit, names):
    code, junk, url, checkbox, idea, question, task, *_ = cues(unit)
    text = unit['text'].strip()
    if 'junk' in names and code:
        return 'junk', 'Code or markup syntax is present.', True
    if 'junk' in names and (junk or not re.search(r'\w', text)):
        return 'junk', 'Attachment placeholder or non-content text.', True
    if 'todos' in names and checkbox:
        return 'todos', 'The source contains a task checkbox.', True
    if 'ideas' in names and idea:
        return 'ideas', 'The wording proposes a possibility or change.', False
    if 'questions' in names and question:
        return 'questions', 'Question wording or a question mark.', False
    if 'todos' in names and task:
        return 'todos', 'An action or explicit intention to do something.', False
    if 'links' in names and url:
        return 'links', 'Contains a saved URL without an explicit action or question.', True
    return None, None, False


def features(unit, vector):
    import numpy as np
    # Hashed words retain shorthand that sentence embeddings can smooth out.
    buckets = np.zeros(128, dtype=np.float32)
    text = normalized(unit['text'])[:2000]
    for word in re.findall(r'\b\w+\b', text):
        index = int.from_bytes(hashlib.blake2s(word.encode(), digest_size=4).digest(), 'little') % len(buckets)
        buckets[index] += 1
    buckets /= max(float(np.linalg.norm(buckets)), 1)
    return np.concatenate([vector * 2, np.asarray(cues(unit), dtype=np.float32), buckets])
