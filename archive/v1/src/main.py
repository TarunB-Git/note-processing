"""Batch classification using the same service as the interactive CLI and app."""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import MODELS_DIR, get_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="Single note")
    inputs.add_argument("--input_file", type=Path, help="Text file, one note per line")
    inputs.add_argument("--input_csv", type=Path, help="CSV file")
    parser.add_argument("--csv_col", default="sentence")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--anchor", type=date.fromisoformat)
    parser.add_argument("--out", type=Path, help="Write CSV results")
    args = parser.parse_args()
    try:
        if args.input is not None:
            lines = [args.input]
        elif args.input_file:
            lines = [line.strip() for line in args.input_file.read_text(encoding="utf8").splitlines() if line.strip()]
        else:
            with args.input_csv.open(encoding="utf8", newline="") as handle:
                reader = csv.DictReader(handle)
                if args.csv_col not in (reader.fieldnames or []):
                    raise ValueError(f"Missing CSV column: {args.csv_col}")
                lines = [row[args.csv_col] for row in reader]
        results = get_classifier(args.models_dir).predict_batch(lines, args.anchor)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", encoding="utf8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "sentence", "predicted_label", "model_label", "parsed_date", "needs_review", "review_reasons",
                ])
                writer.writeheader()
                for item in results:
                    writer.writerow({"sentence": item["text"], "predicted_label": item["pred"],
                                     "model_label": item["model_pred"], "parsed_date": item["parsed_date"],
                                     "needs_review": item["needs_review"], "review_reasons": item["disclaimer"]})
            print(f"Wrote {len(results)} predictions to {args.out}")
        else:
            print(json.dumps(results, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
