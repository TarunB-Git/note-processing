# Note Classification — SVC-based System

Project: Note Classification Using Support Vector Machines
Language: Python 3.8+

## Project summary
This project classifies free-form short notes into three classes: **question**, **task**, and **deadline** using a traditional machine learning pipeline. The classifier is a calibrated Linear SVM (Support Vector Classifier). Feature engineering combines TF–IDF (word and character n-grams), binary cue features, and small engineered lexical/punctuation features. A small Streamlit demo app shows classification and basic deadline parsing.

This repository contains:
- training and evaluation code
- inference helpers (CLI and small standalone module)
- a Streamlit demo app
- a Jupyter notebook with all preprocessing, training, diagnostics, and saved artifacts

## Quick setup (to Test)
1. Create and activate a virtual environment (Linux/macOS):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
2. Install dependencies:
    pip install --upgrade pip
    pip install -r requirements.txt
3. CLI classification: python classify_note_refined.py "finish the report by Monday"
4. streamlit run note_GUI.py

## To Rerun
1. Run:
    python src/train_note_classifier.py \
  --tasks data/processed/tasks_processed.csv \
  --questions data/processed/questions_processed.csv \
  --deadlines data/processed/deadlines_processed.csv \
  --out_dir models \
  --sample_questions 0 \
  --seed 42
  
 sample_questions 0 uses all question examples (no subsampling).
2. Or to run everything all again, use note.ipynb


