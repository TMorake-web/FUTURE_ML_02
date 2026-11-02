"""Predict a ticket category and apply an explainable priority rule."""

import argparse
from pathlib import Path

import joblib

from train_model import clean_text
from priority_rules import predict_priority


def main() -> None:
    # Read the ticket text from the command line; if it is missing, ask for it.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="Ticket subject/description text")
    parser.add_argument("--models-dir", default="artifacts/models")
    args = parser.parse_args()

    ticket_text = " ".join(args.text).strip() or input("Enter a support ticket: ").strip()
    if not ticket_text:
        raise SystemExit("Please provide some ticket text.")

    model_path = Path(args.models_dir) / "category_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError("Category model is missing. Run train_model.py first.")

    # Load the category model created by train_model.py.
    category_model = joblib.load(model_path)
    # The category comes from machine learning; priority comes from the
    # separate, transparent keyword rules in priority_rules.py.
    category = category_model.predict([clean_text(ticket_text)])[0]
    priority = predict_priority(ticket_text)
    print(f"Predicted category: {category}")
    print(f"Rule-based priority: {priority}")


if __name__ == "__main__":
    main()
