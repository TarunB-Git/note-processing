# src/extract_deadlines.py
import re
import csv
import os

def load_cues(path):
    cues = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            # keep multi-word cues intact, ignore obvious xml fragments lines
            if len(s) > 1 and not s.startswith('<'):
                cues.append(s.lower())
    # sort by length desc to match longer cues first
    cues = sorted(set(cues), key=lambda x: -len(x))
    return cues

def read_timeml_lines(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            yield line.rstrip('\n')

# helper: find TIMEX3 tags (TimeML)
timex_re = re.compile(r'<TIMEX3\b[^>]*>(.*?)</TIMEX3>', re.IGNORECASE | re.DOTALL)

# helper: simple sentence splitter for lines (TimeML file may have line-per-sentence)
sent_split_re = re.compile(r'(?<=[.!?])\s+')

def contains_cue(sentence, cues):
    s = sentence.lower()
    for cue in cues:
        if cue in s:
            return True, cue
    return False, None

def timex_nearby(sentence, window_chars=30):
    # if there's a TIMEX3 in sentence: return True
    if timex_re.search(sentence):
        return True
    # also check for common date/time expressions (basic)
    date_like = re.search(r'\b(\d{1,2}[:.]\d{2}\s*(am|pm)?|\b\d{1,2}(st|nd|rd|th)?\b|\b(january|february|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|\btomorrow\b|\btoday\b|\bnext\b)', sentence, re.IGNORECASE)
    return bool(date_like)

def extract_from_timeml(timeml_path, cues, out_csv_path):
    matches = []
    for line in read_timeml_lines(timeml_path):
        # TimeML extract might already be one sentence per line. If not, split.
        parts = sent_split_re.split(line)
        for sent in parts:
            if len(sent.strip()) < 5:
                continue
            has_cue, which = contains_cue(sent, cues)
            has_timex = timex_nearby(sent)
            # Heuristic: require either (a) explicit cue OR (b) TIMEX + weak cue-like signal (e.g., 'by' or 'before')
            if has_cue or (has_timex and any(k in sent.lower() for k in ['by ', 'before ', 'until ', 'due '])):
                matches.append({
                    'sentence': sent.strip(),
                    'has_cue': has_cue,
                    'matched_cue': which if has_cue else '',
                    'has_timex': bool(timex_re.search(sent))
                })
    # deduplicate by sentence
    seen = set()
    unique = []
    for m in matches:
        s = m['sentence']
        if s not in seen:
            seen.add(s)
            unique.append(m)
    # write CSV
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, 'w', encoding='utf-8', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['sentence','has_cue','matched_cue','has_timex'])
        writer.writeheader()
        for u in unique:
            writer.writerow(u)
    return unique

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeml', required=True, help='path to timeml_deadline_sentences.txt')
    parser.add_argument('--cues', required=True, help='path to deadline_cues.txt')
    parser.add_argument('--out', default='data/processed/deadlines_extracted.csv')
    args = parser.parse_args()

    cues = load_cues(args.cues)
    print(f"Loaded {len(cues)} cues (examples): {cues[:10]}")
    hits = extract_from_timeml(args.timeml, cues, args.out)
    print(f"Extracted {len(hits)} unique candidate deadline sentences -> {args.out}")

