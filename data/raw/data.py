#!/usr/bin/env python3
"""
Extract TIMEX3 texts and sentences containing deadline cues from TimeML (.tml/.xml) files.

Produces:
 - timeml_timex_texts.txt         : one TIMEX3 text per line (deduplicated)
 - timeml_deadline_sentences.txt  : one candidate deadline sentence per line (deduplicated)

Usage: run the script in the directory that contains your .tml/.xml files,
or pass a directory path as the first argument.
"""
import glob, os, re, sys
from lxml import etree

# Changeable parameters
INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_TIMEX_TEXTS = "timeml_timex_texts.txt"
OUT_DEADLINE_SENTS = "timeml_deadline_sentences.txt"

# Deadline cue list -- extend if you want
DEADLINE_CUES = re.compile(
    r"\b(by|before|due|until|deadline|no later than|on or before|as soon as possible|ASAP|due by)\b",
    re.I
)

# Simple sentence splitter based on punctuation. Good enough for news text.
SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def find_sentence_containing(fulltext, snippet):
    if not snippet:
        return ""
    # Try splitting and locating the snippet in a sentence
    sents = SENT_SPLIT_RE.split(fulltext)
    for s in sents:
        if snippet in s:
            return s.strip()
    # fallback to window if exact match not found (still useful)
    idx = fulltext.find(snippet)
    if idx == -1:
        return ""
    start = max(0, idx - 150)
    end = min(len(fulltext), idx + len(snippet) + 150)
    return fulltext[start:end].strip()

timex_texts = []
deadline_sentences = []

# Find files
paths = glob.glob(os.path.join(INPUT_DIR, "*.tml")) + glob.glob(os.path.join(INPUT_DIR, "*.xml"))
if not paths:
    print("No .tml or .xml files found in", INPUT_DIR)
    sys.exit(1)

for p in paths:
    try:
        raw = open(p, "r", encoding="utf8", errors="ignore").read()
    except Exception as e:
        print("Skipping", p, "read error:", e)
        continue

    # parse XML; wrap if necessary
    try:
        root = etree.fromstring(raw.encode("utf8"))
    except Exception:
        try:
            root = etree.fromstring(f"<root>{raw}</root>".encode("utf8"))
        except Exception as e:
            print("Failed to parse", p, ":", e)
            continue

    # collect TIMEX3 nodes
    for timex in root.findall(".//TIMEX3"):
        txt = (timex.text or "").strip()
        if txt:
            timex_texts.append(txt)

            # get sentence context from raw file to preserve original punctuation
            sentence = find_sentence_containing(raw, txt)
            if sentence and DEADLINE_CUES.search(sentence):
                # include filename for traceability
                deadline_sentences.append(f"{os.path.basename(p)}\t{sentence}")

# deduplicate preserving order
def dedupe_keep_order(seq):
    seen = set()
    out = []
    for s in seq:
        key = s.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out

timex_texts = dedupe_keep_order(timex_texts)
deadline_sentences = dedupe_keep_order(deadline_sentences)

# write outputs
with open(OUT_TIMEX_TEXTS, "w", encoding="utf8") as f:
    for t in timex_texts:
        f.write(t + "\n")

with open(OUT_DEADLINE_SENTS, "w", encoding="utf8") as f:
    for s in deadline_sentences:
        f.write(s + "\n")

print(f"Wrote {len(timex_texts)} unique TIMEX3 texts to {OUT_TIMEX_TEXTS}")
print(f"Wrote {len(deadline_sentences)} candidate deadline sentences to {OUT_DEADLINE_SENTS}")

