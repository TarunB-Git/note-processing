"""Classify a note, or start an interactive session when no text is supplied."""

import argparse
import json
from datetime import date

from src.classifier import MODELS_DIR, get_classifier, predict_text


def print_result(result):
    print(f"Predicted label: {result['pred']}")
    if result["parsed_date"]:
        print(f"Date: {result['parsed_date']}")
    if result["needs_review"]:
        print(f"Review: {result['disclaimer']}")
    if result["probs"]:
        print(f"Model probabilities ({result['model_pred']}): {result['probs']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="Note to classify; omit for interactive mode")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--anchor", type=date.fromisoformat, help="Reference date: YYYY-MM-DD")
    parser.add_argument("--json", action="store_true", help="Print the complete result as JSON")
    args = parser.parse_args()
    try:
        classifier = get_classifier(args.models_dir)
        if args.text:
            result = classifier.predict_text(" ".join(args.text), args.anchor)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_result(result)
            return
        print("Note classifier. Enter a note; blank input or 'quit' exits.")
        while True:
            try:
                text = input("note> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in {"quit", "exit"}:
                break
            try:
                result = classifier.predict_text(text, args.anchor)
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print_result(result)
            except ValueError as error:
                print(f"Error: {error}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
