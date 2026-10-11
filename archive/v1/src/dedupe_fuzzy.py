#!/usr/bin/env python3
# src/dedupe_fuzzy.py
"""
Fuzzy dedupe CSV by sentence similarity.
Usage:
  python src/dedupe_fuzzy.py --in data/processed/combined.csv --out data/processed/combined_fuzzy_dedup.csv --threshold 0.92
Notes:
  - This is O(n^2); for large n, consider blocking by first N chars or hashing.
"""
import argparse, csv, unicodedata, re
from difflib import SequenceMatcher
from collections import defaultdict
import os

def normalize(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'^[\W_]+|[\W_]+$', '', s)
    return s

def similar(a,b):
    return SequenceMatcher(None, a, b).ratio()

def fuzzy_dedupe(in_csv, out_csv, sentence_col='sentence', threshold=0.92):
    rows = []
    with open(in_csv, newline='', encoding='utf8') as inf:
        reader = csv.DictReader(inf)
        fieldnames = reader.fieldnames
        for row in reader:
            row['_norm'] = normalize(row.get(sentence_col,""))
            rows.append(row)
    keep = []
    dropped = 0
    for i, r in enumerate(rows):
        if i % 5000 == 0 and i>0:
            print(f"Scanning {i}/{len(rows)}")
        n = r['_norm']
        skip = False
        for k in keep:
            if similar(n, k['_norm']) >= threshold:
                skip = True
                dropped += 1
                break
        if not skip:
            keep.append(r)
    # write
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf8') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for r in keep:
            # remove helper
            r2 = {k:v for k,v in r.items() if k != '_norm'}
            writer.writerow(r2)
    print(f"Wrote {len(keep)} rows; dropped {dropped}")
    return len(keep)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_csv", required=True)
    p.add_argument("--out", dest="out_csv", required=True)
    p.add_argument("--threshold", type=float, default=0.92)
    args = p.parse_args()
    fuzzy_dedupe(args.in_csv, args.out_csv, threshold=args.threshold)

