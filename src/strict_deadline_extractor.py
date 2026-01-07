# src/strict_deadline_extractor.py
"""
Strict deadline extractor.

Usage:
  python src/strict_deadline_extractor.py --input data/raw --out data/processed/strict_deadlines.csv

It accepts:
 - a single file path (txt/csv/json/xlsx) OR
 - a directory path (it will recursively scan .txt/.csv/.json/.xlsx files)

Output CSV columns: sentence, matched_cue, has_date, source_path, label_source
label_source is 'strict_rule'
"""
import os
import re
import csv
import argparse
import pandas as pd
from pathlib import Path

# strong cues (must be used; single 'by' or 'before' are not considered strong)
STRONG_CUES = ['deadline', 'due', 'no later than']

# date/time regex (broad but practical)
DATE_RE = re.compile(
    r'\b(\d{1,2}[:.]\d{2}\s*(am|pm)?|'                # times like 13:30, 1:30pm
    r'\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|'  # 2nd Jan
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|'  # month names
    r'\btomorrow\b|\btoday\b|\btonight\b|\bnext\b|\beod\b|\bby\s+\d{1,2}(am|pm)?\b)', re.I)

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def list_files(input_path):
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    files = []
    for root, _, names in os.walk(str(p)):
        for n in names:
            if n.lower().endswith(('.txt', '.csv', '.json', '.xlsx')):
                files.append(os.path.join(root, n))
    return files

def load_texts_from_file(path):
    ext = path.lower().split('.')[-1]
    try:
        if ext == 'txt':
            text = Path(path).read_text(encoding='utf8', errors='ignore')
            return [text]
        if ext == 'csv':
            df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=1000000)
            # prefer common text columns
            for c in ['text','content','note','sentence','body']:
                if c in df.columns:
                    return df[c].dropna().astype(str).tolist()
            # fallback to first object column
            for c in df.columns:
                if df[c].dtype == object:
                    return df[c].dropna().astype(str).tolist()
            return []
        if ext == 'json':
            df = pd.read_json(path, lines=True)
            for c in ['text','content','note','sentence','body']:
                if c in df.columns:
                    return df[c].dropna().astype(str).tolist()
            # fallback: attempt to stringify rows
            return df.astype(str).apply(lambda r: ' '.join(r.values), axis=1).tolist()
        if ext in ('xlsx','xls'):
            df = pd.read_excel(path, dtype=str)
            for c in ['text','content','note','sentence','body']:
                if c in df.columns:
                    return df[c].dropna().astype(str).tolist()
            for c in df.columns:
                if df[c].dtype == object:
                    return df[c].dropna().astype(str).tolist()
            return []
    except Exception:
        return []
    return []

def split_sentences(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def sentence_is_strict_deadline(sent):
    low = sent.lower()
    has_date = bool(DATE_RE.search(sent))
    matched_cue = None
    # If any strong cue present -> accept (regardless of date)
    for sc in STRONG_CUES:
        if sc in low:
            matched_cue = sc
            return True, matched_cue, has_date
    # no strong cue -> reject (we want strict)
    return False, None, False

def extract(input_path, out_csv, max_files=0, min_len=10):
    files = list_files(input_path)
    if max_files and max_files > 0:
        files = files[:max_files]
    results = []
    for f in files:
        texts = load_texts_from_file(f)
        for t in texts:
            for sent in split_sentences(str(t)):
                if len(sent) < min_len:
                    continue
                matched, cue, has_date = sentence_is_strict_deadline(sent)
                if matched:
                    results.append({
                        'sentence': sent,
                        'matched_cue': cue or '',
                        'has_date': int(has_date),
                        'source_path': f,
                        'label_source': 'strict_rule'
                    })
    # dedupe while preserving order
    seen = set()
    unique = []
    for r in results:
        s = r['sentence']
        if s not in seen:
            seen.add(s)
            unique.append(r)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['sentence','matched_cue','has_date','source_path','label_source'])
        writer.writeheader()
        for u in unique:
            writer.writerow(u)
    print(f"Wrote {len(unique)} strict candidates -> {out_csv}")
    return unique

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='file or directory (txt/csv/json/xlsx)')
    ap.add_argument('--out', default='data/processed/strict_deadlines.csv')
    ap.add_argument('--max_files', type=int, default=0, help='limit files for quick test')
    ap.add_argument('--min_len', type=int, default=10, help='minimum sentence length (chars)')
    args = ap.parse_args()
    extract(args.input, args.out, max_files=args.max_files, min_len=args.min_len)

