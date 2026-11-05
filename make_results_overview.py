"""Create a plain-language image summary from the saved metrics.json file."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default="artifacts/reports/metrics.json",
        help="Path to the metrics.json file created by train_model.py",
    )
    parser.add_argument(
        "--output",
        default="results_overview.png",
        help="Where to save the image (default: results_overview.png)",
    )
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    report = metrics["classification_report"]
    summary_rows = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    class_rows = [
        (name, values)
        for name, values in report.items()
        if name not in summary_rows
        and isinstance(values, dict)
        and "recall" in values
        and "support" in values
    ]
    class_rows.sort(key=lambda row: row[1]["recall"])

    model_accuracy = metrics["accuracy"]
    baseline_accuracy = metrics["majority_baseline_accuracy"]
    test_rows = metrics["test_rows"]
    correct_total = round(model_accuracy * test_rows)
    improvement_pp = (model_accuracy - baseline_accuracy) * 100

    fig = plt.figure(figsize=(15, 9), facecolor="white")
    fig.text(
        0.06,
        0.94,
        "Support ticket results, at a glance",
        fontsize=23,
        fontweight="bold",
        color="#14243a",
    )
    fig.text(
        0.06,
        0.895,
        f"The model sorted {correct_total} of {test_rows} held-out tickets into the right category.",
        fontsize=13,
        color="#334155",
    )

    ax_overall = fig.add_axes([0.08, 0.43, 0.34, 0.36])
    comparison_labels = ["Simple guess\n(most common type)", "Text model"]
    comparison_values = [baseline_accuracy, model_accuracy]
    bars = ax_overall.barh(
        comparison_labels,
        comparison_values,
        color=["#9aa8b8", "#2563a6"],
        height=0.52,
    )
    ax_overall.set_xlim(0, 1.08)
    ax_overall.xaxis.set_major_formatter(PercentFormatter(1))
    ax_overall.set_xticks(np.arange(0, 1.01, 0.25))
    ax_overall.set_title("Overall tickets sorted correctly", loc="left", fontsize=14, pad=15)
    ax_overall.set_xlabel("Share of test tickets", color="#475569")
    ax_overall.spines[["top", "right", "left"]].set_visible(False)
    ax_overall.spines["bottom"].set_color("#cbd5e1")
    ax_overall.tick_params(axis="y", length=0, labelsize=11, colors="#24364b")
    ax_overall.tick_params(axis="x", colors="#475569")
    ax_overall.grid(axis="x", color="#e7edf3", linewidth=0.8)
    ax_overall.set_axisbelow(True)
    for bar, value in zip(bars, comparison_values):
        ax_overall.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#14243a",
        )
    fig.text(
        0.08,
        0.24,
        f"+{improvement_pp:.1f} percentage points vs. simple guess",
        fontsize=11,
        color="#2563a6",
        fontweight="bold",
    )

    ax_classes = fig.add_axes([0.52, 0.22, 0.43, 0.60])
    names = [row[0] for row in class_rows]
    recalls = [row[1]["recall"] for row in class_rows]
    supports = [int(row[1]["support"]) for row in class_rows]
    correct_by_class = [round(row[1]["recall"] * row[1]["support"]) for row in class_rows]
    y_positions = np.arange(len(names))
    class_bars = ax_classes.barh(y_positions, recalls, color="#3d78ad", height=0.62)
    ax_classes.set_yticks(y_positions, labels=names)
    ax_classes.set_xlim(0, 1.28)
    ax_classes.xaxis.set_major_formatter(PercentFormatter(1))
    ax_classes.set_xticks(np.arange(0, 1.01, 0.25))
    ax_classes.set_title("How many tickets were sorted correctly in each type?", loc="left", fontsize=14, pad=15)
    ax_classes.set_xlabel("Share correct within that ticket type", color="#475569")
    ax_classes.spines[["top", "right", "left"]].set_visible(False)
    ax_classes.spines["bottom"].set_color("#cbd5e1")
    ax_classes.tick_params(axis="y", length=0, labelsize=10, colors="#24364b")
    ax_classes.tick_params(axis="x", colors="#475569")
    ax_classes.grid(axis="x", color="#e7edf3", linewidth=0.8)
    ax_classes.set_axisbelow(True)
    for bar, value, correct, total in zip(class_bars, recalls, correct_by_class, supports):
        ax_classes.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}  ({correct}/{total})",
            va="center",
            fontsize=10,
            color="#14243a",
        )

    fig.text(
        0.06,
        0.12,
        "Priority tags are assigned with keyword rules. This dataset has no labeled priorities, so priority accuracy was not measured.",
        fontsize=10.5,
        color="#24364b",
        wrap=True,
    )
    fig.text(
        0.06,
        0.075,
        "Evaluation uses a held-out set of 657 tickets; results on future tickets may differ.",
        fontsize=10,
        color="#526173",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved results overview to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
