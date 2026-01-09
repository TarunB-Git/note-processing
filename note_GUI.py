# app.py — Streamlit note classifier (no calendar). Saves as a single file.
import streamlit as st
from pathlib import Path
import joblib, json, re, hashlib
import numpy as np
from datetime import date, datetime, timedelta
from scipy.sparse import csr_matrix, hstack

# Attempt to import dateparser (optional but recommended)
try:
    import dateparser, dateparser.search as dp_search  # type: ignore
    _HAS_DATEPARSER = True
except Exception:
    _HAS_DATEPARSER = False

# --- model artifacts (adjust names if yours differ) ---
MODELS_DIR = Path("models")
TF_WORD = MODELS_DIR / "tfidf_word_adj.joblib"
TF_CHAR = MODELS_DIR / "tfidf_char_adj.joblib"
CUE_VEC = MODELS_DIR / "cue_vec_adj.joblib"
META = MODELS_DIR / "feature_metadata_adj.json"

# load artifacts (fail fast with helpful message)
def load_artifacts():
    if not MODELS_DIR.exists():
        raise FileNotFoundError("models/ directory not found — place model artifacts next to app.py")
    # prefer a calibrated model if present
    candidates = [
        MODELS_DIR / "svc_calibrated_adj.joblib",
        MODELS_DIR / "svc_calibrated_with_timex_spacy.joblib",
        MODELS_DIR / "svc_adj.joblib"
    ]
    model = None
    model_path = None
    for p in candidates:
        if p.exists():
            model = joblib.load(p)
            model_path = p
            break
    if model is None:
        raise FileNotFoundError("No supported model artifact found in models/. Expected one of: svc_calibrated_adj.joblib, svc_adj.joblib, etc.")
    if not TF_WORD.exists() or not CUE_VEC.exists():
        raise FileNotFoundError("Missing vectorizer artifacts in models/. Expected tfidf_word_adj.joblib and cue_vec_adj.joblib (and tfidf_char if used).")
    tf_word = joblib.load(TF_WORD)
    tf_char = joblib.load(TF_CHAR) if TF_CHAR.exists() else None
    cue_vec = joblib.load(CUE_VEC)
    meta = json.load(open(META, 'r', encoding='utf8')) if META.exists() else {}
    return model, model_path.name, tf_word, tf_char, cue_vec, meta

model, model_name, tf_word, tf_char, cue_vec, feature_meta = load_artifacts()

# --- simple engineered features (must align with how you trained the model) ---
NUM_RE = re.compile(r'\d+')
WEEKDAY_RE = re.compile(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', re.I)

def engineered_features_list(sent_list):
    rows = []
    for s in sent_list:
        sl = (s or "").lower()
        tokens = sl.split()
        tc = max(1, len(tokens))
        avg_token_len = float(sum(len(t) for t in tokens) / tc)
        unique_ratio = float(len(set(tokens)) / tc)
        numeric_ratio = float(sum(1 for t in tokens if NUM_RE.fullmatch(t)) / tc)
        has_weekday = 1 if WEEKDAY_RE.search(sl) else 0
        has_month = 1 if re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)', sl, re.I) else 0
        has_time = 1 if re.search(r'\b(?:\d{1,2}(:\d{2})?\s*(am|pm)|noon|midnight|eod|cob)\b', sl, re.I) else 0
        has_relative = 1 if re.search(r'\b(tomorrow|today|next|in\s+\d+\s+(?:day|days|week|weeks|month|months|hour|hours|minute|minutes))\b', sl, re.I) else 0
        has_question = 1 if re.search(r'\?|\b(who|what|when|where|why|how|which)\b', sl, re.I) else 0
        starts_imp = 1 if re.match(r'^(submit|finish|complete|turn|send|email|deliver|make|do|create|update|fix|upload|prepare|post|hand in|return)\b', sl, re.I) else 0
        q_marks = len(re.findall(r'\?', sl))
        parsed_date_flag = 0
        has_date_ent = 0
        rows.append([
            tc, avg_token_len, unique_ratio, numeric_ratio, int(bool(NUM_RE.search(sl))),
            has_weekday, has_month, has_time, has_relative, 0,
            has_question, starts_imp, 0, 0, q_marks,
            parsed_date_flag, has_date_ent
        ])
    return np.array(rows, dtype=float)

# --- date extraction helpers (deterministic next-occurrence logic) ---
WEEKDAY_NAME_TO_INT = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6
}
ORDINAL_RE = re.compile(r'\b([1-9]|[12][0-9]|3[01])(st|nd|rd|th)\b', re.I)
CONTEXT_NUM_RE = re.compile(r'\b(?:on|by|due|before|until|next|this|in|the)\s+(?:the\s+)?([1-9]|[12][0-9]|3[01])(?:\b|[^0-9])', re.I)
SLASH_DATE_RE = re.compile(r'\b([0-3]?\d)[/-]([01]?\d)(?:[/-](\d{2,4}))?\b', re.I)
MONTH_DAY_RE = re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+([1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\b', re.I)
WEEKDAY_RE_SIMPLE = re.compile(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.I)

def next_date_for_day_of_month(day: int, anchor: date = None) -> date:
    if anchor is None:
        anchor = date.today()
    year = anchor.year
    month = anchor.month
    import calendar as _cal
    # Try this month and subsequent months until a valid future date is found
    for add in range(0, 13):
        m = month + add
        y = year + ((m-1) // 12)
        m = ((m-1) % 12) + 1
        lastday = _cal.monthrange(y, m)[1]
        if day <= lastday:
            cand = date(y, m, day)
            if cand > anchor:
                return cand
    # fallback
    return anchor + timedelta(days=7)

def next_weekday_date(weekday_int: int, anchor: date = None) -> date:
    if anchor is None:
        anchor = date.today()
    days_ahead = (weekday_int - anchor.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return anchor + timedelta(days=days_ahead)
# REPLACE your extract_date_from_text(...) with this corrected version

import re
from datetime import date, datetime, timedelta

# optional: dateparser usage if installed (keeps your earlier import)
# _HAS_DATEPARSER boolean should already be defined in your app

# regexes (accept short month names like Jan, Feb)
MONTH_DAY_RE = re.compile(
    r'\b('
    r'jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?'
    r')\.?\s+([1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\b',
    re.I
)
ORDINAL_RE = re.compile(r'\b([1-9]|[12][0-9]|3[01])(st|nd|rd|th)\b', re.I)
CONTEXT_NUM_RE = re.compile(r'\b(?:on|by|due|before|until|next|this|in|the)\s+(?:the\s+)?([1-9]|[12][0-9]|3[01])(?:\b|[^0-9])', re.I)
SLASH_DATE_RE = re.compile(r'\b([0-3]?\d)[/-]([01]?\d)(?:[/-](\d{2,4}))?\b', re.I)
WEEKDAY_RE_SIMPLE = re.compile(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.I)

WEEKDAY_NAME_TO_INT = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6
}

def next_date_for_day_of_month(day: int, anchor: date = None) -> date:
    if anchor is None:
        anchor = date.today()
    year = anchor.year
    month = anchor.month
    import calendar as _cal
    # iterate months until we find a valid date strictly after anchor
    for add in range(0, 24):  # search up to 2 years ahead (safe)
        m = month + add
        y = year + ((m - 1) // 12)
        m = ((m - 1) % 12) + 1
        lastday = _cal.monthrange(y, m)[1]
        if day <= lastday:
            cand = date(y, m, day)
            if cand > anchor:
                return cand
    return anchor + timedelta(days=7)

def next_weekday_date(weekday_int: int, anchor: date = None) -> date:
    if anchor is None:
        anchor = date.today()
    days_ahead = (weekday_int - anchor.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    return anchor + timedelta(days=days_ahead)

def extract_date_from_text(s: str):
    """
    Return (date_obj, source_tag) if a date could be determined, else (None, None).
    """
    s0 = (s or "").strip()
    if not s0:
        return None, None

    today = date.today()

    # 1) Prefer dateparser search if available (and ask for future)
    if '_HAS_DATEPARSER' in globals() and globals()['_HAS_DATEPARSER']:
        try:
            res = dp_search.search_dates(s0, settings={'PREFER_DATES_FROM': 'future'})
            if res:
                dt = res[0][1]
                # ensure we return a date object
                return dt.date(), 'dateparser'
        except Exception:
            # fall through to regex heuristics
            pass

    # 2) Explicit month name + day, e.g. "Jan 24th" or "January 24"
    m = MONTH_DAY_RE.search(s0)
    if m:
        month_text = m.group(1)
        day = int(m.group(2))
        # normalize month (try abbreviated and full)
        try:
            # try abbreviated first
            mon_dt = datetime.strptime(month_text[:3], "%b")
            month_num = mon_dt.month
        except Exception:
            try:
                mon_dt = datetime.strptime(month_text, "%B")
                month_num = mon_dt.month
            except Exception:
                month_num = None
        if month_num is not None:
            # construct candidate in current year
            try:
                cand = date(today.year, month_num, day)
                if cand <= today:
                    # choose next year
                    cand = date(today.year + 1, month_num, day)
                return cand, 'month_day'
            except Exception:
                # invalid day for that month (shouldn't happen with regex), fallthrough
                pass

    # 3) ordinal like "24th" -> next 24th of a month
    m = ORDINAL_RE.search(s0)
    if m:
        day = int(m.group(1))
        return next_date_for_day_of_month(day), 'ordinal'

    # 4) contextual number: "on 24" or "by 5"
    m = CONTEXT_NUM_RE.search(s0)
    if m:
        day = int(m.group(1))
        return next_date_for_day_of_month(day), 'context_num'

    # 5) slash-style date: prefer mm/dd if plausible, otherwise dd/mm, pick next future occurrence
    m = SLASH_DATE_RE.search(s0)
    if m:
        d1 = int(m.group(1)); d2 = int(m.group(2))
        # try mm/dd
        try:
            cand = date(today.year, d1, d2)
            if cand > today:
                return cand, 'slash_mm_dd'
        except Exception:
            pass
        # try dd/mm
        try:
            cand = date(today.year, d2, d1)
            if cand > today:
                return cand, 'slash_dd_mm'
        except Exception:
            pass

    # 6) weekday name -> next weekday (not today)
    m = WEEKDAY_RE_SIMPLE.search(s0)
    if m:
        wd = m.group(1).lower()
        return next_weekday_date(WEEKDAY_NAME_TO_INT[wd]), 'weekday'

    # nothing found
    return None, None

# --- feature matrix builder and alignment --- 
def build_feature_matrix_for_sentences(sent_list):
    Xw = tf_word.transform(sent_list).astype(np.float32)
    Xc = tf_char.transform(sent_list).astype(np.float32) if tf_char is not None else None
    Xcue = cue_vec.transform(sent_list).astype(np.float32)
    Xeng = engineered_features_list(sent_list)
    expected_nfeat = getattr(model, "n_features_in_", None)
    if expected_nfeat is None:
        # try calibrated inner estimator
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_)>0:
            expected_nfeat = getattr(model.calibrated_classifiers_[0].estimator, "n_features_in_", None)
        elif hasattr(model, "base_estimator"):
            expected_nfeat = getattr(model.base_estimator, "n_features_in_", None)
    if expected_nfeat is None:
        raise RuntimeError("Cannot determine model input feature count.")
    word_dim = Xw.shape[1]
    char_dim = Xc.shape[1] if Xc is not None else 0
    cue_dim = Xcue.shape[1]
    sum_sparse = word_dim + char_dim + cue_dim
    expected_eng = expected_nfeat - sum_sparse
    if expected_eng < 0:
        raise RuntimeError("Vectorizer/model mismatch: model expects fewer sparse features than available.")
    # trim/pad engineered to expected_eng
    if Xeng.shape[1] > expected_eng:
        Xeng_al = Xeng[:, :expected_eng]
    elif Xeng.shape[1] < expected_eng:
        pad = np.zeros((len(sent_list), expected_eng - Xeng.shape[1]), dtype=float)
        Xeng_al = np.hstack([Xeng, pad])
    else:
        Xeng_al = Xeng
    parts = [csr_matrix(Xw)]
    if Xc is not None:
        parts.append(csr_matrix(Xc))
    parts.append(csr_matrix(Xcue))
    parts.append(csr_matrix(Xeng_al))
    X_all = hstack(parts, format='csr')
    return X_all

# --- UI and interactions ---
st.set_page_config("Note Classifier", layout="wide")
st.title("Note Classifier — Questions / Tasks / Deadlines")
st.markdown(
    "Instructions: paste or type notes (one sentence per line) in the editor on the left. "
    "Press **Classify** or edit the editor to trigger classification. "
    "Questions show with a Google link; tasks have persistent checkboxes; deadlines show the interpreted next date (or report 'Unable to resolve date')."
)
st.markdown("---")

left_col, right_col = st.columns([2,1])
with left_col:
    if "notes_editor" not in st.session_state:
        st.session_state["notes_editor"] = ""
    def _on_change_editor():
        st.session_state["do_classify"] = True
    notes_text = st.text_area("Notes (one per line)", key="notes_editor", height=420, on_change=_on_change_editor)
    if st.button("Classify"):
        st.session_state["do_classify"] = True

with right_col:
    st.subheader("Output")
    output_placeholder = st.empty()

if "task_checks" not in st.session_state:
    st.session_state["task_checks"] = {}

if st.session_state.get("do_classify", False):
    st.session_state["do_classify"] = False
    lines = [l.strip() for l in st.session_state["notes_editor"].splitlines() if l.strip()]
    if not lines:
        output_placeholder.info("No notes provided.")
    else:
        X = build_feature_matrix_for_sentences(lines)
        if hasattr(model, "predict_proba"):
            probs_all = model.predict_proba(X)
            preds = model.predict(X)
        else:
            preds = model.predict(X)
            probs_all = None

        questions = []
        tasks = []
        deadlines = []
        for i, s in enumerate(lines):
            pred = preds[i]
            probs = dict(zip(model.classes_, probs_all[i])) if probs_all is not None else None
            parsed_date, reason = extract_date_from_text(s)
            item = {"text": s, "pred": pred, "probs": probs, "parsed_date": parsed_date, "reason": reason}
            if pred == "question":
                questions.append(item)
            elif pred == "task":
                tasks.append(item)
                # if task contains date, also list as deadline for review
                if parsed_date is not None:
                    deadlines.append(item)
            elif pred == "deadline":
                deadlines.append(item)
            else:
                tasks.append(item)

        # render results
        with output_placeholder.container():
            qcol, tcol, dcol = st.columns([1,1,1])
            with qcol:
                st.markdown("### Questions")
                if not questions:
                    st.write("_No questions detected_")
                else:
                    for q in questions:
                        qtext = q["text"]
                        google_url = f"https://www.google.com/search?q={qtext.replace(' ', '+')}"
                        st.markdown(f"- {qtext}  [🔎]({google_url})")
            with tcol:
                st.markdown("### To-do (Tasks)")
                if not tasks:
                    st.write("_No tasks detected_")
                else:
                    for t in tasks:
                        tid = hashlib.md5(t["text"].encode("utf8")).hexdigest()
                        key = f"task_chk_{tid}"
                        prev = st.session_state["task_checks"].get(key, False)
                        checked = st.checkbox(t["text"], value=prev, key=key)
                        st.session_state["task_checks"][key] = checked
                        if checked:
                            st.markdown("<span style='text-decoration: line-through;color:gray'>{}</span>".format(t["text"]), unsafe_allow_html=True)
            with dcol:
                st.markdown("### Deadlines (interpreted)")
                if not deadlines:
                    st.write("_No deadlines detected_")
                else:
                    for d in deadlines:
                        dt = d["parsed_date"]
                        if dt is None:
                            date_str = "Unable to resolve date"
                            st.markdown("- {}  _→ {}_".format(d["text"], date_str))
                        else:
                            # format nicely
                            if isinstance(dt, datetime):
                                dt = dt.date()
                            date_str = dt.strftime("%A, %B %d, %Y")
                            st.markdown("- {}  _→ {}_".format(d["text"], date_str))
            st.success(f"Classified {len(lines)} notes — Q: {len(questions)}  T: {len(tasks)}  D: {len(deadlines)}")

# footer
st.markdown("---")
st.caption("If you see incorrect date interpretations, edit the note and re-run 'Classify'.")

