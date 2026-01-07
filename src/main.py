# src/predict_note.py
"""
Predict note class (task / question / deadline).

Usage examples:
# single sentence
python src/predict_note.py --model models/model_and_tfidf.joblib --input "Submit the report by Friday"

# batch from text file (one sentence per line)
python src/predict_note.py --model models/model_and_tfidf.joblib --input_file data/my_sentences.txt --out predictions.csv

# batch from CSV (column 'sentence')
python src/predict_note.py --model models/model_and_tfidf.joblib --input_csv data/test.csv --csv_col sentence --out predictions.csv
"""
import argparse
import re
import unicodedata
import joblib
import numpy as np
import csv
import os
from scipy.sparse import hstack

# -------------------------
# Local copy of BinaryFeatureExtractor (stateless)
# -------------------------
import re as _re
QUESTION_WORDS = set(['who','what','when','where','why','how','which','is','do','does','did','can','should','could'])
DEADLINE_CUES = set(['by','before','due','deadline','no later than','until','eod','cob','asap','tomorrow','today','next'])

class BinaryFeatureExtractor:
    """Minimal, stateless feature extractor that mirrors the training code."""
    def __init__(self):
        self.q_re = _re.compile(r'\?|\b(' + '|'.join(QUESTION_WORDS) + r')\b', _re.I)
        self.deadline_words = DEADLINE_CUES
        self.starts_re = _re.compile(r'^(submit|finish|complete|turn|send|email|deliver|make|do|create|update|fix|upload)\b', _re.I)

    def transform(self, X):
        out = []
        for s in X:
            s_low = s.lower()
            has_q = bool(self.q_re.search(s))
            has_deadline = any(w in s_low for w in self.deadline_words)
            starts = bool(self.starts_re.match(s_low))
            out.append([int(has_q), int(has_deadline), int(starts)])
        return np.array(out)

# -------------------------
# Utilities
# -------------------------
def normalize_text(s):
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'^[\W_]+|[\W_]+$', '', s)
    return s.lower()

def load_model(model_path):
    # model_and_tfidf.joblib expected to contain dict with 'model' and 'tfidf'
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and 'model' in bundle and 'tfidf' in bundle:
        return bundle['model'], bundle['tfidf']
    # older save format: try load directly
    if hasattr(bundle, 'predict'):
        # no tfidf present
        raise ValueError("Loaded model appears not to include 'tfidf'. Please supply file that contains both model and tfidf.")
    raise ValueError("Unexpected model file structure. Expected dict with keys 'model' and 'tfidf'.")

def score_to_confidence(decision_scores, label_index=None):
    """
    Accepts decision_scores that may be:
      - scalar-like (numpy scalar or array of length 1) -> treat as binary score
      - 1D array of length > 1 (multiclass raw scores) -> softmax and return prob for label_index or max prob
      - 2D array (n_samples x n_classes) -> use first row (for single-sample callers)
    Returns a float in [0,1] approximating confidence.
    """
    arr = np.asarray(decision_scores)
    # scalar or length-1
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        val = float(np.ravel(arr)[0])
        # logistic mapping
        conf = 1.0 / (1.0 + np.exp(-val))
        return float(conf)
    # 2D: assume shape (n_samples, n_classes) -> take first row
    if arr.ndim == 2:
        row = arr[0]
        arr1 = np.asarray(row).ravel()
        exps = np.exp(arr1 - np.max(arr1))
        probs = exps / exps.sum()
        if label_index is None:
            return float(np.max(probs))
        return float(probs[label_index])
    # 1D with length>1: multiclass raw scores
    if arr.ndim == 1 and arr.size > 1:
        arr1 = arr
        exps = np.exp(arr1 - np.max(arr1))
        probs = exps / exps.sum()
        if label_index is None:
            return float(np.max(probs))
        return float(probs[label_index])
    # fallback
    try:
        val = float(np.ravel(arr)[0])
        return float(1.0 / (1.0 + np.exp(-val)))
    except Exception:
        return None

# -------------------------
# Prediction helpers
# -------------------------
def predict_single(model, tfidf, sentence):
    s_norm = normalize_text(sentence)
    X_text = tfidf.transform([s_norm])
    bin_feats = BinaryFeatureExtractor().transform([s_norm])
    X_full = hstack([X_text, bin_feats])
    pred = model.predict(X_full)[0]
    # get decision scores if available
    try:
        df = model.decision_function(X_full)
    except Exception:
        df = None
    # compute confidence
    if df is None:
        conf = None
    else:
        arr = np.asarray(df)
        # prepare index of predicted class (if multiclass)
        pred_index = None
        if arr.ndim == 2:
            row = arr[0]
            pred_index = int(np.argmax(row))
            conf = score_to_confidence(row, pred_index)
        elif arr.ndim == 1:
            if arr.size == 1:
                conf = score_to_confidence(arr)
            else:
                pred_index = int(np.argmax(arr))
                conf = score_to_confidence(arr, pred_index)
        else:
            conf = score_to_confidence(arr)
    return pred, conf

def predict_batch(model, tfidf, lines):
    results = []
    normalized = [normalize_text(s) for s in lines]
    X_text = tfidf.transform(normalized)
    bin_feats = BinaryFeatureExtractor().transform(normalized)
    X_full = hstack([X_text, bin_feats])
    preds = model.predict(X_full)
    try:
        dfs = model.decision_function(X_full)
        dfs_arr = np.asarray(dfs)
    except Exception:
        dfs = None
        dfs_arr = None
    for i, s in enumerate(lines):
        pred = preds[i]
        conf = None
        if dfs_arr is not None:
            if dfs_arr.ndim == 2:
                row = dfs_arr[i]
                pred_idx = int(np.argmax(row))
                conf = score_to_confidence(row, pred_idx)
            elif dfs_arr.ndim == 1:
                # binary-like array; we take element i if length matches, else use the whole array
                if dfs_arr.size == len(lines):
                    val = dfs_arr[i]
                    conf = score_to_confidence(val)
                else:
                    pred_idx = int(np.argmax(dfs_arr))
                    conf = score_to_confidence(dfs_arr, pred_idx)
        results.append((s, pred, conf))
    return results

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to model_and_tfidf.joblib (dict with keys model and tfidf)')
    ap.add_argument('--input', help='Single sentence string to classify (shell-quoted)')
    ap.add_argument('--input_file', help='Path to text file with one sentence per line')
    ap.add_argument('--input_csv', help='Path to CSV file (use with --csv_col)')
    ap.add_argument('--csv_col', help='Column name in CSV with sentences (default: sentence)', default='sentence')
    ap.add_argument('--out', help='Output CSV path for batch mode (optional)', default=None)
    args = ap.parse_args()

    model, tfidf = load_model(args.model)

    # single
    if args.input:
        pred, conf = predict_single(model, tfidf, args.input)
        if conf is None:
            print(f"Predicted class: {pred}")
        else:
            print(f"Predicted class: {pred} (confidence ~ {conf:.3f})")
        return

    # text file mode
    lines = []
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf8', errors='ignore') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    elif args.input_csv:
        import pandas as pd
        df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
        if args.csv_col not in df.columns:
            raise ValueError(f"CSV column '{args.csv_col}' not found in {args.input_csv}")
        lines = df[args.csv_col].astype(str).tolist()
    else:
        raise ValueError("No input provided. Use --input or --input_file or --input_csv")

    results = predict_batch(model, tfidf, lines)

    # print or write out
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, 'w', encoding='utf8', newline="") as outf:
            w = csv.writer(outf)
            w.writerow(['sentence','predicted_label','confidence'])
            for s,p,c in results:
                w.writerow([s,p,(f"{c:.6f}" if c is not None else "")])
        print(f"Wrote {len(results)} predictions to {args.out}")
    else:
        for s,p,c in results:
            if c is None:
                print(f"[{p}] {s}")
            else:
                print(f"[{p} | conf {c:.3f}] {s}")

if __name__ == "__main__":
    main()

