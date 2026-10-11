# Note Processing v1

The original classifier sorts short English notes into questions, tasks, and deadlines. It includes a terminal interface, CSV/text batch processing, and a Streamlit app.

```text
Note → word/character TF-IDF + cue features → calibrated LinearSVC
     → question/action rules + date parser → label, date, review flag
```

`src/` contains classification, feature extraction, deadline parsing, and training. `models/` contains the adjusted model and its required vectorizers. The original experiments are recorded in `note.ipynb` and `NoteProcessing.pdf`; retained figures show evaluation and class distribution.

Model probabilities and the final rule-adjusted label are distinct outputs. The adjusted bundle is the default; retrained bundles use a versioned feature schema.

Python 3.10–3.12 is supported. Dependencies are listed in `requirements.txt`. Training datasets and superseded bundles are omitted from the public tree. This historical archive is excluded from the current app's Apache license grant.
