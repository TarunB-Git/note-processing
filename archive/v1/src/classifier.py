"""Shared classification service for the CLI, batch runner, and Streamlit app."""

import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

from src.deadline_parser import extract_deadline
from src.handcrafted_features import (
    EngineeredFeatures, FEATURE_NAMES, normalize_legacy_text, normalize_text,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
QUESTION_START_RE = re.compile(
    r"^(?:who|what|when|where|why|how|which)\b"
    r"|^(?:is|are|was|were|do|does|did|can|could|will|would|should|has|have)\s+"
    r"(?:i|you|he|she|it|we|they|there|this|that|these|those|the|a|an|my|our|your|his|her|their)\b", re.I,
)
ACTION_RE = re.compile(
    r"^(?:please\s+)?(?:submit|finish|complete|send|email|deliver|make|do|have|create|update|fix|"
    r"upload|prepare|post|return|buy|call|book|pay|read|remember|check|review|schedule|meet|pick up|hand in)\b", re.I,
)


class NoteClassifier:
    def __init__(self, models_dir=MODELS_DIR):
        directory = Path(models_dir).resolve()
        bundle_path = directory / "note_classifier_v1.joblib"
        self.features = None
        if bundle_path.exists():
            bundle = joblib.load(bundle_path)
            if bundle.get("schema_version") != 1:
                raise ValueError("Unsupported classifier bundle schema")
            self.model = bundle["model"]
            self.model_path = bundle_path
            self.normalize = normalize_text
        else:
            self.model_path = directory / "svc_calibrated_adj.joblib"
            if not self.model_path.exists():
                self.model_path = directory / "svc_adj.joblib"
            required = [self.model_path, directory / "tfidf_word_adj.joblib",
                        directory / "tfidf_char_adj.joblib", directory / "cue_vec_adj.joblib",
                        directory / "feature_metadata_adj.json"]
            missing = [p.name for p in required if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing artifacts in {directory}: {', '.join(missing)}")
            metadata = json.loads(required[-1].read_text(encoding="utf8"))
            if (metadata.get("engineered_feature_names") != FEATURE_NAMES
                    or metadata.get("engineered_feature_count") != len(FEATURE_NAMES)):
                raise ValueError("Engineered feature schema does not match the adjusted model")
            self.model = joblib.load(self.model_path)
            self.features = [joblib.load(p) for p in required[1:4]]
            self.normalize = normalize_legacy_text
            width = sum(len(v.get_feature_names_out()) for v in self.features) + len(FEATURE_NAMES)
            if width != self.model.n_features_in_:
                raise ValueError(f"Artifact mismatch: {width} features; model expects {self.model.n_features_in_}")
        if set(self.model.classes_) != {"task", "question", "deadline"}:
            raise ValueError("Expected model classes: task, question, deadline")

    def predict_batch(self, texts, anchor=None):
        texts = list(texts)
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Each note must be a non-empty string")
        normalized = [self.normalize(text) for text in texts]
        if any(not text for text in normalized):
            raise ValueError("A note contains no usable text after normalization")
        if self.features is None:
            inputs = normalized
        else:
            inputs = hstack([
                *[vectorizer.transform(normalized).astype(np.float32) for vectorizer in self.features],
                EngineeredFeatures().transform(normalized),
            ], format="csr")
        probabilities = self.model.predict_proba(inputs) if hasattr(self.model, "predict_proba") else None
        predictions = (self.model.classes_[probabilities.argmax(axis=1)] if probabilities is not None
                       else self.model.predict(inputs))
        results = []
        for index, (text, model_pred) in enumerate(zip(texts, predictions)):
            policy_text = normalize_text(text)
            temporal = extract_deadline(policy_text, anchor)
            pred = str(model_pred)
            source = "model"
            if "?" in policy_text or QUESTION_START_RE.match(policy_text):
                pred, source = "question", "question_syntax"
            elif ACTION_RE.match(policy_text):
                pred = "deadline" if temporal["has_deadline_signal"] else "task"
                source = "action_with_date" if temporal["has_deadline_signal"] else "action_syntax"
            elif pred == "task" and temporal["has_deadline_signal"]:
                pred, source = "deadline", "task_with_date"
            probs = (dict(zip(map(str, self.model.classes_), map(float, probabilities[index])))
                     if probabilities is not None else None)
            review_reasons = []
            if pred != model_pred:
                review_reasons.append(f"{source} changed model label {model_pred} to {pred}")
            if probs is not None and max(probs.values()) < 0.60:
                review_reasons.append("Model probability below 0.60")
            if pred == "deadline" and temporal["parsed_date"] is None:
                review_reasons.append(temporal["deadline_reason"] or "No supported calendar date found")
            results.append({
                "text": text, "pred": pred, "model_pred": str(model_pred), "probs": probs,
                "classification_source": source, **temporal,
                "needs_review": bool(review_reasons), "review_reasons": review_reasons,
                "possible_deadline_flag": pred == "deadline" and model_pred == "task",
                "disclaimer": "; ".join(review_reasons) or None,
                "model_path": str(self.model_path),
            })
        return results

    def predict_text(self, text, anchor=None):
        return self.predict_batch([text], anchor)[0]


@lru_cache(maxsize=4)
def get_classifier(models_dir=MODELS_DIR):
    return NoteClassifier(models_dir)


def predict_text(text, anchor=None, models_dir=MODELS_DIR):
    return get_classifier(models_dir).predict_text(text, anchor)
