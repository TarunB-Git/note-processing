# src/train_note_classifier.py
"""
Train a 3-way note classifier: task / question / deadline

Usage (example):
python src/train_note_classifier.py \
  --tasks data/raw/mslatte_tasks.txt \
  --questions data/raw/msmarcosmall.txt \
  --deadlines data/processed/synth_deadlines_rich.csv \
  --out_dir models/ \
  --sample_questions 50000 \
  --random_seed 42

Outputs:
 - models/vectorizer.joblib
 - models/model.joblib
 - models/eval_report.txt
 - models/confusion_matrix.png
"""
import argparse, os, re, unicodedata, csv, random
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, confusion_matrix
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Helpers: normalization + loaders

def normalize_text(s):
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    # strip trailing punctuation
    s = re.sub(r'^[\W_]+|[\W_]+$', '', s)
    return s.lower()

def load_plain_lines(path):
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        return [l.strip() for l in f if l.strip()]

def load_csv_sentences(path, sentence_col='sentence'):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if sentence_col in df.columns:
        return df[sentence_col].astype(str).tolist()
    # fallback: try first column
    return df.iloc[:,0].astype(str).tolist()

def load_and_label(tasks_path, questions_path, deadlines_path):
    # try to auto-detect file types (txt vs csv)
    def loader(p):
        if p.lower().endswith(('.csv', '.tsv')):
            return load_csv_sentences(p)
        else:
            return load_plain_lines(p)
    tasks = loader(tasks_path)
    questions = loader(questions_path)
    deadlines = loader(deadlines_path)
    return tasks, questions, deadlines

# Feature extractor transformers

QUESTION_WORDS = set(['who','what','when','where','why','how','which','is','do','does','did','can','should','could'])
DEADLINE_CUES = set(['by','before','due','deadline','no later than','until','eod','cob','asap','tomorrow','today','next'])

class BinaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract small binary features from raw text: question marker, deadline cue, starts with verb (heuristic)"""
    def __init__(self):
        # compile small regexes
        self.q_re = re.compile(r'\?|\b(' + '|'.join(QUESTION_WORDS) + r')\b', re.I)
        # deadline cues are checked as substrings for speed
        self.deadline_words = DEADLINE_CUES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for s in X:
            s_low = s.lower()
            has_q = bool(self.q_re.search(s))
            has_deadline = any(w in s_low for w in self.deadline_words)
            # starts with verb heuristic: first token is a verb-like (imperative) common action words
            starts = bool(re.match(r'^(submit|finish|complete|turn|send|email|deliver|make|do|create|update|fix|upload)\b', s_low))
            out.append([int(has_q), int(has_deadline), int(starts)])
        return np.array(out)

# Utility: dedupe & prepare DataFrame
def build_dataset_df(tasks, questions, deadlines, sample_questions=None, random_seed=42, dedupe_norm=True):
    rows = []
    for s in tasks:
        rows.append({'sentence': normalize_text(s), 'label': 'task', 'label_source': 'mslatte'})
    for s in deadlines:
        rows.append({'sentence': normalize_text(s), 'label': 'deadline', 'label_source': 'synthetic'})
    # questions may be huge: optionally subsample
    if sample_questions and sample_questions > 0:
        random.seed(random_seed)
        questions_sample = random.sample(questions, min(sample_questions, len(questions)))
    else:
        questions_sample = questions
    for s in questions_sample:
        rows.append({'sentence': normalize_text(s), 'label': 'question', 'label_source': 'msmarco'})
    df = pd.DataFrame(rows)
    if dedupe_norm:
        df = df.drop_duplicates(subset=['sentence'])
    return df

# Training pipeline
def train_and_evaluate(df, out_dir, random_seed=42, tfidf_max_features=50000, cv_folds=5):
    os.makedirs(out_dir, exist_ok=True)
    X = df['sentence'].tolist()
    y = df['label'].tolist()
    # train/test split stratified
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=random_seed)

    # Pipeline: TF-IDF + binary features (FeatureUnion)
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=tfidf_max_features, min_df=3, max_df=0.95)
    binary = Pipeline([('bin', BinaryFeatureExtractor())])
    # FeatureUnion requires numeric arrays; we will vectorize text then append binary features later manually
    # Simpler approach: vectorize text and then horizontally stack with binary features
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    bin_train = BinaryFeatureExtractor().fit_transform(X_train)
    bin_test = BinaryFeatureExtractor().transform(X_test)
    # combine
    from scipy.sparse import hstack
    X_train_full = hstack([X_train_tfidf, bin_train])
    X_test_full = hstack([X_test_tfidf, bin_test])

    # model & grid search
    svc = LinearSVC(class_weight='balanced', max_iter=20000)
    param_grid = {'C': [0.01, 0.1, 1.0, 5.0]}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    grid = GridSearchCV(svc, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train_full, y_train)

    best = grid.best_estimator_
    # evaluate on test
    y_pred = best.predict(X_test_full)
    report = classification_report(y_test, y_pred, digits=4)
    macro = f1_score(y_test, y_pred, average='macro')

    # save model and vectorizer (+ note: binary extractor is stateless, save class for rebuilding)
    joblib.dump({'model': best, 'tfidf': tfidf}, os.path.join(out_dir, 'model_and_tfidf.joblib'))

    # write evaluation report
    with open(os.path.join(out_dir, 'eval_report.txt'), 'w', encoding='utf8') as f:
        f.write("Best Params: " + str(grid.best_params_) + "\n\n")
        f.write("Macro-F1 on test: %.4f\n\n" % macro)
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nDataset label counts:\n")
        f.write(df['label'].value_counts().to_string())

    # confusion matrix plot
    labels = sorted(df['label'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('predicted'); plt.ylabel('true'); plt.title('Confusion matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

    print("Best params:", grid.best_params_)
    print("Macro-F1 on test:", macro)
    print(report)
    print("Saved model+tfidf and reports to", out_dir)
    return best

# CLI main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True, help='path to tasks file (txt or csv)')
    ap.add_argument('--questions', required=True, help='path to questions file (txt or csv)')
    ap.add_argument('--deadlines', required=True, help='path to deadlines file (txt or csv)')
    ap.add_argument('--out_dir', default='models', help='output directory to save model and reports')
    ap.add_argument('--sample_questions', type=int, default=50000, help='subsample size for questions (set 0 to use all)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tasks, questions, deadlines = load_and_label(args.tasks, args.questions, args.deadlines)
    print("Loaded: tasks=%d, questions=%d, deadlines=%d" % (len(tasks), len(questions), len(deadlines)))

    df = build_dataset_df(tasks, questions, deadlines, sample_questions=args.sample_questions, random_seed=args.seed)
    print("After normalization & dedupe: total rows =", len(df))
    print(df['label'].value_counts())

    model = train_and_evaluate(df, args.out_dir, random_seed=args.seed)

if __name__ == "__main__":
    main()

