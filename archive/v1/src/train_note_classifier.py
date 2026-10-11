"""Train, calibrate, and save a complete v1 classifier bundle."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.handcrafted_features import FEATURE_NAMES, create_features, normalize_text


def load_sentences(path):
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                            dtype=str, keep_default_na=False)
        for column in ("sentence_orig", "sentence", "text"):
            if column in frame:
                return frame[column].tolist()
        raise ValueError(f"{path} must have a sentence_orig, sentence, or text column")
    return path.read_text(encoding="utf8").splitlines()


def build_dataset_df(tasks, questions, deadlines, sample_questions=50000, random_seed=42):
    rows = [(normalize_text(text), label)
            for label, texts in (("task", tasks), ("question", questions), ("deadline", deadlines))
            for text in texts if text.strip()]
    frame = pd.DataFrame(rows, columns=["sentence", "label"])
    frame["dedupe_key"] = frame["sentence"].str.replace(r"^[\W_]+|[\W_]+$", "", regex=True)
    empty = frame["dedupe_key"].eq("")
    frame = frame[~empty].copy()
    conflicts = frame.groupby("dedupe_key")["label"].nunique()
    conflicts = conflicts[conflicts > 1].index
    conflicting_rows = frame["dedupe_key"].isin(conflicts)
    audit = {"empty_rows_removed": int(empty.sum()), "conflicting_rows_removed": int(conflicting_rows.sum()),
             "conflicting_texts_removed": len(conflicts)}
    frame = frame[~conflicting_rows]
    before = len(frame)
    frame = frame.drop_duplicates("dedupe_key")
    audit["duplicates_removed"] = before - len(frame)
    if sample_questions > 0:
        question_rows = frame[frame.label == "question"]
        question_rows = question_rows.sample(n=min(sample_questions, len(question_rows)), random_state=random_seed)
        frame = pd.concat([frame[frame.label != "question"], question_rows])
    frame = frame.drop(columns="dedupe_key").reset_index(drop=True)
    frame.attrs["audit"] = audit
    return frame


def train_and_evaluate(frame, out_dir, random_seed=42, tfidf_max_features=50000, cv_folds=3):
    if set(frame.label) != {"task", "question", "deadline"} or frame.label.value_counts().min() < 20:
        raise ValueError("Training requires at least 20 unique notes per class after cleaning")
    X_train, X_rest, y_train, y_rest = train_test_split(
        frame.sentence, frame.label, test_size=0.25, stratify=frame.label, random_state=random_seed)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, test_size=0.60, stratify=y_rest, random_state=random_seed)
    pipeline = Pipeline([
        ("features", create_features(tfidf_max_features)),
        ("svc", LinearSVC(class_weight="balanced", max_iter=20000, dual="auto", random_state=random_seed)),
    ])
    grid = GridSearchCV(pipeline, {"svc__C": [0.1, 1.0, 5.0]}, scoring="f1_macro",
                        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=random_seed),
                        n_jobs=1, verbose=1)
    grid.fit(X_train, y_train)
    model = CalibratedClassifierCV(grid.best_estimator_, method="sigmoid", cv="prefit")
    model.fit(X_calib, y_calib)
    model_preds = model.predict(X_test)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "note_classifier_v1.joblib"
    joblib.dump({"schema_version": 1, "model": model}, bundle_path, compress=3)

    from src.classifier import NoteClassifier
    service = NoteClassifier(output)
    final_preds = [item["pred"] for item in service.predict_batch(X_test.tolist())]
    labels = ["deadline", "question", "task"]
    report = {
        "schema_version": 1, "created": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__, "python_version": sys.version.split()[0],
        "seed": random_seed, "best_params": grid.best_params_, "cleaning": frame.attrs.get("audit", {}),
        "label_counts": {k: int(v) for k, v in frame.label.value_counts().items()},
        "split_counts": {"train": len(X_train), "calibration": len(X_calib), "test": len(X_test)},
        "engineered_feature_names": FEATURE_NAMES,
        "model_macro_f1": f1_score(y_test, model_preds, average="macro"),
        "service_macro_f1": f1_score(y_test, final_preds, average="macro"),
        "model_report": classification_report(y_test, model_preds, output_dict=True, zero_division=0),
        "service_report": classification_report(y_test, final_preds, output_dict=True, zero_division=0),
        "confusion_labels": labels,
        "model_confusion_matrix": confusion_matrix(y_test, model_preds, labels=labels).tolist(),
        "service_confusion_matrix": confusion_matrix(y_test, final_preds, labels=labels).tolist(),
        "policy_disagreements": [
            {"sentence": text, "label": actual, "model_pred": str(model_pred), "final_pred": final_pred}
            for text, actual, model_pred, final_pred in zip(X_test, y_test, model_preds, final_preds)
            if model_pred != final_pred
        ][:100],
    }
    (output / "evaluation_v1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print("Model-only classification:")
    print(classification_report(y_test, model_preds, digits=4, zero_division=0))
    print("Final classification policy:")
    print(classification_report(y_test, final_preds, digits=4, zero_division=0))
    print(f"Saved {bundle_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("tasks", "questions", "deadlines"):
        parser.add_argument(f"--{label}", required=True, type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("models/retrained"))
    parser.add_argument("--sample_questions", type=int, default=50000, help="0 uses all questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.sample_questions < 0:
            raise ValueError("sample_questions must be >= 0")
        frame = build_dataset_df(load_sentences(args.tasks), load_sentences(args.questions),
                                 load_sentences(args.deadlines), args.sample_questions, args.seed)
        print(frame.label.value_counts().to_string())
        print("Cleaning:", frame.attrs["audit"])
        train_and_evaluate(frame, args.out_dir, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
