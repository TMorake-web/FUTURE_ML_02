"""Train and evaluate the IT support ticket category classifier."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def clean_text(value: object) -> str:
    """Prepare raw ticket text so similar words have a consistent form."""
    # Unicode normalization makes visually similar characters more consistent.
    text = unicodedata.normalize("NFKC", str(value)).lower()
    # URLs rarely tell the classifier which support category a ticket belongs to.
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    # Keep letters, numbers, and spaces; replace punctuation with spaces.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Collapse repeated spaces left behind by the previous cleaning steps.
    return re.sub(r"\s+", " ", text).strip()


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Zenodo provides separate text and label files for train and test data.
    # Check all four first so a missing download gives a clear message.
    required = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {data_dir.resolve()}. "
            "Download the four Zenodo CSV files into the data folder."
        )

    def combine(feature_file: str, label_file: str) -> pd.DataFrame:
        # Load only the ticket ID/text and its category label.
        features = pd.read_csv(data_dir / feature_file, usecols=["id", "text"])
        labels = pd.read_csv(data_dir / label_file, usecols=["id", "category_truth"])
        # Match each ticket to its correct answer using its ID, not row position.
        joined = features.merge(labels, on="id", how="inner", validate="one_to_one")
        if len(joined) != len(features) or len(joined) != len(labels):
            raise ValueError(f"Ticket IDs did not fully match in {feature_file} and {label_file}.")
        # Clean text and remove rows missing either the ticket or its label.
        joined["text"] = joined["text"].fillna("").map(clean_text)
        joined["category_truth"] = joined["category_truth"].fillna("").astype(str).str.strip()
        joined = joined[(joined["text"] != "") & (joined["category_truth"] != "")]
        if joined.empty:
            raise ValueError(f"No usable labeled tickets found in {feature_file}.")
        return joined

    train = combine("X_train.csv", "y_train.csv")
    test = combine("X_test.csv", "y_test.csv")
    return train, test


def make_model() -> Pipeline:
    # TF-IDF turns words and short character patterns into numbers the model
    # can learn from. Word features capture terms/phrases; character features
    # can help with spelling differences and the dataset's multiple languages.
    text_features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=40000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        steps=[
            # A pipeline keeps the text conversion and classifier together.
            ("text_features", text_features),
            # Linear SVM is a common, efficient choice for text categories.
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )


def main() -> None:
    # Optional folder settings let the project run from the project root or
    # use another data/output location when needed.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Folder containing Zenodo CSVs")
    parser.add_argument("--output-dir", default="artifacts", help="Folder for model and reports")
    args = parser.parse_args()

    # Train only on the training split. Keep the separate test split unseen
    # until the final evaluation so the score reflects new tickets.
    train, test = load_splits(Path(args.data_dir))
    model = make_model()
    model.fit(train["text"], train["category_truth"])
    predictions = model.predict(test["text"])
    labels = sorted(set(test["category_truth"]))

    # Accuracy is the overall correct share. F1 combines precision and recall;
    # macro F1 gives each category equal weight, while weighted F1 reflects size.
    accuracy = float(accuracy_score(test["category_truth"], predictions))
    macro_f1 = float(f1_score(test["category_truth"], predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(test["category_truth"], predictions, average="weighted", zero_division=0))
    majority_baseline = float(test["category_truth"].value_counts(normalize=True).max())
    report = classification_report(
        test["category_truth"], predictions, labels=labels, output_dict=True, zero_division=0
    )

    # Save the reusable model separately from evaluation reports.
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "category_model.joblib")

    metrics = {
        "dataset": "Zenodo Classification of IT Support Tickets",
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "category_count": int(train["category_truth"].nunique()),
        "accuracy": accuracy,
        "majority_baseline_accuracy": majority_baseline,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "priority_method": "transparent keyword rules; no labeled priority column is available",
    }
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Build the plain-language overview from the metrics just written, so the
    # visual is reproducible whenever this training script is run.
    overview_script = Path(__file__).with_name("make_results_overview.py")
    overview_path = reports_dir / "results_overview.png"
    subprocess.run(
        [
            sys.executable,
            str(overview_script),
            "--metrics",
            str(metrics_path),
            "--output",
            str(overview_path),
        ],
        check=True,
    )

    # A confusion matrix shows which real categories were predicted as others.
    display = ConfusionMatrixDisplay.from_predictions(
        test["category_truth"], predictions, labels=labels, xticks_rotation=45, cmap="Blues"
    )
    display.ax_.set_title("IT support ticket category classification")
    plt.tight_layout()
    plt.savefig(reports_dir / "category_confusion_matrix.png", dpi=160)
    plt.close()

    # Print a compact summary in the terminal so the run is easy to check.
    print(f"Training tickets: {len(train)}")
    print(f"Test tickets: {len(test)}")
    print(f"Categories: {', '.join(labels)}")
    print(f"Accuracy: {accuracy:.3f} | majority baseline: {majority_baseline:.3f}")
    print(f"Macro F1: {macro_f1:.3f} | weighted F1: {weighted_f1:.3f}")
    print(classification_report(test["category_truth"], predictions, zero_division=0))
    print(f"Saved model and reports under: {output_dir.resolve()}")
    print(f"Plain-language results image: {overview_path.resolve()}")
    print("Priority levels are assigned by rules; this dataset has no priority labels to evaluate.")


if __name__ == "__main__":
    main()
