"""Feature definitions shared by training and inference."""

import re
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

FEATURE_NAMES = [
    "token_count", "avg_token_len", "unique_ratio", "numeric_ratio", "has_digit",
    "has_weekday", "has_month", "has_time", "has_relative", "has_duration",
    "has_question", "starts_imperative", "has_currency", "has_url_email", "q_marks",
]
CUE_VOCAB = [
    "deadline", "due", "no later than", "due by", "deadline for", "by", "before",
    "until", "on or before", "eod", "cob", "asap", "tomorrow", "today", "next",
    "in", "at", "before noon", "due on", "expected by", "no later",
]
NUM_RE = re.compile(r"\d+")
WEEKDAY_RE = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
MONTH_RE = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
TIME_RE = re.compile(r"\b(?:\d{1,2}(:\d{2})?\s*(am|pm)|am|pm|noon|midnight|eod|cob)\b", re.I)
RELATIVE_RE = re.compile(r"\b(tomorrow|today|next|in\s+\d+\s+(?:day|days|week|weeks|month|months|hour|hours|minute|minutes))\b", re.I)
QUESTION_RE = re.compile(r"\?|\b(who|what|when|where|why|how|which)\b", re.I)
IMPERATIVE_RE = re.compile(r"^(submit|finish|complete|turn|send|email|deliver|make|do|create|update|fix|upload|prepare|post|hand in|return)\b", re.I)
DURATION_RE = re.compile(r"\b(hour|hours|minute|minutes|day|days|week|weeks|month|months|year|years)\b", re.I)
CURRENCY_RE = re.compile(r"[$£€¥]|usd|aud|gbp|cad", re.I)
URL_EMAIL_RE = re.compile(r"http[s]?://|www\.|@", re.I)
SIGNOFF_RE = re.compile(r"(?:^\-+\s*[A-Za-z]{2,30}\s*$|\b(thanks|thank you|regards|best|cheers)\b[.,!]?)$", re.I)


def normalize_text(text):
    """Keep punctuation: '?' and currency markers are model inputs."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def normalize_legacy_text(text):
    """Match preprocessing used to fit the checked-in adjusted artifacts."""
    text = re.sub(r"\s+", " ", text).strip()
    text = SIGNOFF_RE.sub("", text).strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text).strip()
    return re.sub(r"\s+", " ", text).lower()


class EngineeredFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            text = text.lower()
            tokens = text.split()
            count = max(1, len(tokens))
            rows.append([
                count, sum(map(len, tokens)) / count, len(set(tokens)) / count,
                sum(bool(NUM_RE.fullmatch(t)) for t in tokens) / count,
                int(bool(NUM_RE.search(text))),
                *[int(bool(pattern.search(text))) for pattern in (
                    WEEKDAY_RE, MONTH_RE, TIME_RE, RELATIVE_RE, DURATION_RE,
                    QUESTION_RE, IMPERATIVE_RE, CURRENCY_RE, URL_EMAIL_RE,
                )],
                text.count("?"),
            ])
        return csr_matrix(np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)))


def create_features(max_features=50000):
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=max_features,
                                min_df=2, strip_accents="unicode", dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                max_features=30000, min_df=2, dtype=np.float32)),
        ("cue", CountVectorizer(vocabulary=CUE_VOCAB, ngram_range=(1, 3), binary=True)),
        ("engineered", EngineeredFeatures()),
    ])
