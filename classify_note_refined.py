import sys
import re
from pathlib import Path
import json
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

MODELS_DIR = Path("models")
TF_WORD = MODELS_DIR / "tfidf_word_adj.joblib"
TF_CHAR = MODELS_DIR / "tfidf_char_adj.joblib"
CUE_VEC = MODELS_DIR / "cue_vec_adj.joblib"
META = MODELS_DIR / "feature_metadata_adj.json"

WEEKDAY_RE = re.compile(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', re.I)
RELATIVE_RE = re.compile(r'\b(tomorrow|today|tonight|this\s+(morning|afternoon|evening)|next week|next month|next year|next\b|in\s+\d+\s+(?:day|days|week|weeks|month|months|hour|hours|minute|minutes))\b', re.I)
MONTH_DAY_RE = re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+([1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\b', re.I)
ORDINAL_RE = re.compile(r'\b([1-9]|[12][0-9]|3[01])(st|nd|rd|th)\b', re.I)
CONTEXT_NUM_RE = re.compile(r'\b(?:on|by|due|before|until|next|this|in)\s+(?:the\s+)?([1-9]|[12][0-9]|3[01])(?:\b|[^0-9])', re.I)
SLASH_DATE_RE = re.compile(r'\b([0-3]?\d)[/-]([01]?\d)\b')
DEADLINE_CUES = re.compile(r'\b(due|deadline|no later than|eod|cob|by\b|before\b|until\b)\b', re.I)

def is_likely_deadline(text: str):
    s = (text or "").lower()
    if not s.strip():
        return False, None
    if DEADLINE_CUES.search(s):
        return True, "deadline_cue"
    if WEEKDAY_RE.search(s):
        return True, "weekday"
    if RELATIVE_RE.search(s):
        return True, "relative_phrase"
    if MONTH_DAY_RE.search(s):
        return True, "month_day"
    if ORDINAL_RE.search(s):
        return True, "ordinal_1_31"
    if CONTEXT_NUM_RE.search(s):
        return True, "context_number_1_31"
    if SLASH_DATE_RE.search(s):
        return True, "slash_date"
    return False, None

NUM_RE = re.compile(r'\d+')
QUESTION_RE = re.compile(r'\?|\b(who|what|when|where|why|how|which)\b', re.I)
IMPERATIVE_RE = re.compile(r'^(submit|finish|complete|turn|send|email|deliver|make|do|create|update|fix|upload|prepare|post|hand in|return)\b', re.I)

def engineered_features_one(s: str, parsed_date_flag: int = 0, has_date_ent: int = 0):
    sl = (s or "").lower()
    tokens = sl.split()
    tc = max(1, len(tokens))
    num_digits = 1 if NUM_RE.search(sl) else 0
    avg_token_len = float(sum(len(t) for t in tokens) / tc)
    unique_ratio = float(len(set(tokens)) / tc)
    numeric_ratio = float(sum(1 for t in tokens if NUM_RE.fullmatch(t)) / tc)
    has_weekday = 1 if WEEKDAY_RE.search(sl) else 0
    has_month = 1 if re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)', sl, re.I) else 0
    has_time = 1 if re.search(r'\b(?:\d{1,2}(:\d{2})?\s*(am|pm)|noon|midnight|eod|cob)\b', sl, re.I) else 0
    has_relative = 1 if RELATIVE_RE.search(sl) else 0
    has_duration = 1 if re.search(r'\b(hour|hours|minute|minutes|day|days|week|weeks|month|months|year|years)\b', sl, re.I) else 0
    has_question = 1 if QUESTION_RE.search(sl) else 0
    starts_imp = 1 if IMPERATIVE_RE.match(sl) else 0
    has_currency = 1 if re.search(r'[$£€¥]|usd|aud|gbp|cad', sl, re.I) else 0
    has_url_email = 1 if re.search(r'http[s]?://|www\.|@', sl, re.I) else 0
    q_marks = len(re.findall(r'\?', sl))
    return np.array([
        tc, avg_token_len, unique_ratio, numeric_ratio, num_digits,
        has_weekday, has_month, has_time, has_relative, has_duration,
        has_question, starts_imp, has_currency, has_url_email, q_marks,
        int(parsed_date_flag), int(has_date_ent)
    ], dtype=float).reshape(1, -1)

def load_artifacts():
    # prefer calibrated model if present
    candidates = [
        MODELS_DIR / "svc_calibrated_with_timex_spacy.joblib",
        MODELS_DIR / "svc_calibrated_adj.joblib",
        MODELS_DIR / "svc_calibrated_small.joblib",
        MODELS_DIR / "svc_with_timex_spacy.joblib",
        MODELS_DIR / "svc_adj.joblib",
        MODELS_DIR / "svc_small.joblib",
        MODELS_DIR / "sgd_partial_adj.joblib",
        MODELS_DIR / "sgd_partial.joblib"
    ]
    model = None
    model_path = None
    for p in candidates:
        if p.exists():
            model = joblib.load(p)
            model_path = p
            break
    if model is None:
        raise FileNotFoundError("No model found in models/. Expected one of: " + ", ".join(str(x) for x in candidates))
    if not TF_WORD.exists() or not CUE_VEC.exists() or not META.exists():
        raise FileNotFoundError("Missing vectorizers/metadata in models/. Expected tfidf_word_adj.joblib, cue_vec_adj.joblib, feature_metadata_adj.json.")
    tf_word = joblib.load(TF_WORD)
    tf_char = joblib.load(TF_CHAR) if TF_CHAR.exists() else None
    cue_vec = joblib.load(CUE_VEC)
    meta = json.load(open(META, 'r', encoding='utf8'))
    return model, model_path, tf_word, tf_char, cue_vec, meta

# --- inference helper ---
def predict_text(text: str):
    model, model_path, tf_word, tf_char, cue_vec, meta = load_artifacts()
    s = str(text or "")
    # lexical transforms
    Xw = tf_word.transform([s]).astype(np.float32)
    Xc = tf_char.transform([s]).astype(np.float32) if tf_char is not None else None
    Xcue = cue_vec.transform([s]).astype(np.float32)
    # engineered
    eng = engineered_features_one(s, parsed_date_flag=0, has_date_ent=0)
    # align engineered dims to model expectation
    expected_nfeat = getattr(model, "n_features_in_", None)
    if expected_nfeat is None:
        # try calibrated wrapper internal estimator
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            try:
                inner = model.calibrated_classifiers_[0].estimator
                expected_nfeat = getattr(inner, "n_features_in_", None)
            except Exception:
                expected_nfeat = None
        elif hasattr(model, "base_estimator"):
            expected_nfeat = getattr(model.base_estimator, "n_features_in_", None)
    if expected_nfeat is None:
        raise RuntimeError("Could not determine model.n_features_in_. Model may be incompatible.")
    word_dim = Xw.shape[1]
    char_dim = Xc.shape[1] if Xc is not None else 0
    cue_dim = Xcue.shape[1]
    sum_sparse = word_dim + char_dim + cue_dim
    expected_eng = expected_nfeat - sum_sparse
    if expected_eng < 0:
        raise RuntimeError(f"Model expects fewer sparse features ({expected_nfeat}) than sum(word+char+cue)={sum_sparse}. Vectorizer/model mismatch.")
    # trim/pad engineered
    eng_cols = eng.shape[1]
    if eng_cols > expected_eng:
        eng_al = eng[:, :expected_eng]
    elif eng_cols < expected_eng:
        pad = np.zeros((1, expected_eng - eng_cols), dtype=float)
        eng_al = np.hstack([eng, pad])
    else:
        eng_al = eng
    # assemble final X
    comps = [csr_matrix(Xw)]
    if Xc is not None:
        comps.append(csr_matrix(Xc))
    comps.append(csr_matrix(Xcue))
    comps.append(csr_matrix(eng_al))
    X_in = hstack(comps, format='csr')
    if X_in.shape[1] != expected_nfeat:
        raise RuntimeError(f"Feature alignment failed: got {X_in.shape[1]} features, expected {expected_nfeat}.")
    # predict
    if hasattr(model, "predict_proba"):
        probs_arr = model.predict_proba(X_in)[0]
        classes = list(model.classes_)
        probs = dict(zip(classes, [float(x) for x in probs_arr]))
        pred = max(probs.items(), key=lambda x: x[1])[0]
    else:
        pred = model.predict(X_in)[0]
        probs = None
    # deadline check rule: only flag if model predicted 'task' and heuristic detects a date
    is_date, reason = is_likely_deadline(s)
    possible_deadline_flag = False
    disclaimer = None
    if pred == 'task' and is_date:
        possible_deadline_flag = True
        disclaimer = f"Predicted 'task' by model, but text matches a date-like pattern ({reason}); likely a deadline. Review manually."
    # return structured result
    return {
        "text": s,
        "pred": pred,
        "probs": probs,
        "possible_deadline_flag": possible_deadline_flag,
        "deadline_reason": reason,
        "disclaimer": disclaimer,
        "model_path": str(model_path)
    }

# --- CLI entrypoint ---
def interactive():
    print("Simple note classifier CLI. Type a sentence and press Enter. Type 'quit' or empty line to exit.")
    while True:
        try:
            raw = input("note> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw or raw.lower() in ("quit", "exit"):
            print("Exiting.")
            break
        try:
            out = predict_text(raw)
        except Exception as e:
            print("Error during prediction:", e)
            continue
        print(f"PRED: {out['pred']}", end="")
        if out['possible_deadline_flag']:
            print("  [LIKELY DEADLINE FLAGGED]")
            print("NOTE:", out['disclaimer'])
        else:
            print()
        if out['probs'] is not None:
            print("PROBS:", out['probs'])
        print("-" * 60)

def one_shot(arg_text: str):
    try:
        out = predict_text(arg_text)
    except Exception as e:
        print("Error:", e)
        return
    print("Text:", out['text'])
    print("Predicted label:", out['pred'])
    if out['possible_deadline_flag']:
        print("Warning: likely deadline (reason:", out['deadline_reason'], ")")
        if out['disclaimer']:
            print("Disclaimer:", out['disclaimer'])
    if out['probs'] is not None:
        print("Probabilities:", out['probs'])
    print("Model artifact:", out['model_path'])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        one_shot(" ".join(sys.argv[1:]))
    else:
        interactive()

