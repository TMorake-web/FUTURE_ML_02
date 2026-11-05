# Support Ticket Classification

This project trains a text classifier for support ticket categories and uses simple, documented rules to assign High, Medium, or Low priority.

## Dataset

The [Future Interns Task 2 page](https://futureinterns.com/machine-learning-task-2-2026/) recommends the [Zenodo Classification of IT Support Tickets dataset](https://zenodo.org/records/7648117). It contains 2,229 manually classified support tickets in seven categories. Zenodo describes the tickets as manually classified by three IT support professionals and says personal information was removed.

Download these four files from the Zenodo record and place them in `data/`:

```text
data/
├── X_train.csv
├── y_train.csv
├── X_test.csv
└── y_test.csv
```

The feature CSVs contain `id` and `text`; label CSVs contain `id` and `category_truth`. The script joins features and labels by ticket ID and uses only the ticket text and category.

## Setup in VS Code on Windows

Use the existing `.venv` environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train and evaluate the category model:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The model uses word and character TF-IDF features with a linear SVM. It evaluates on Zenodo's provided test split and reports accuracy, macro F1, weighted F1, per-category precision and recall, and a confusion matrix. The majority-class baseline is shown for comparison. The training script also generates a plain-language results image from the saved metrics.

## Results at a glance

On the held-out test set, the model sorted **547 of 657 tickets correctly (83.3%)**. A simple guess that always chooses the most common category would score **33.6%** on this split. The 49.6 percentage-point difference compares the model with that simple baseline, not with an earlier trained model.

![Plain-language summary of ticket classification results](artifacts/reports/results_overview.png)

## Priority tagging

Zenodo's dataset has category labels but no priority labels. The project therefore assigns priority using explicit keyword rules in `priority_rules.py`:

- **High:** outage, security breach, compromised account, emergency/urgent/critical, and similar severe terms.
- **Medium:** errors, failed services, access problems, and other issues.
- **Low:** messages that do not match the High or Medium rules.

These priority labels are heuristic, not ground truth. Do not report priority accuracy or describe priority as a trained ML prediction unless you obtain independently labeled priority data.

## Try a ticket

After training:

```powershell
.\.venv\Scripts\python.exe predict_ticket.py "Our production system is down and many users are affected"
```

The output includes an ML-predicted category and a rule-based priority.

## Output files

```text
artifacts/
├── models/category_model.joblib
└── reports/
    ├── results_overview.png
    ├── category_confusion_matrix.png
    └── metrics.json
```

`results_overview.png` compares the model with the majority-class baseline and shows the share of tickets correctly sorted within each category. Its baseline comparison is not a comparison with an earlier trained model.

## Limitation

The category model is a decision-support prototype and can make mistakes, especially on multilingual or ambiguous tickets. Review its test metrics before describing performance, and do not use it as the sole basis for real support decisions.
