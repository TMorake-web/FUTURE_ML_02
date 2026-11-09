# Support Ticket Classification

## Project Overview

This project builds a support-ticket classification prototype. It predicts one of seven ticket categories from the ticket text and assigns a High, Medium, or Low priority using documented keyword rules.

The category is predicted by a machine-learning model. Priority is assigned by rules because the dataset does not include priority labels.

## Dataset

This project uses the [Classification of IT Support Tickets dataset on Zenodo](https://zenodo.org/records/7648117), recommended by the [Future Interns Task 2 brief](https://futureinterns.com/machine-learning-task-2-2026/).

The dataset contains 2,229 manually classified support tickets in seven categories. Zenodo states that three IT support professionals classified the tickets and that personal information was removed.

Download these four files from the Zenodo record and place them in the project's `data/` folder:

```text
data/
├── X_train.csv
├── y_train.csv
├── X_test.csv
└── y_test.csv
```

The feature files contain `id` and `text` columns. The label files contain `id` and `category_truth` columns. The training script joins features and labels by ticket ID and uses the ticket text and category.

The dataset files are not included in this repository. Download them from Zenodo before running the scripts.

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib

## How the Category Model Works

The model cleans ticket text and converts it into word and character TF-IDF features. A linear Support Vector Machine (SVM) uses those features to predict one of seven categories.

The model is evaluated using Zenodo's provided held-out test set. The script reports accuracy, macro F1, weighted F1, per-category precision and recall, and a confusion matrix. It also compares the results with a majority-class baseline.

## Results

On the held-out test set, the model classified **547 of 657 tickets correctly**, for an accuracy of **83.3%**. A baseline that always guesses the most common category would score **33.6%** on this test set. That is a difference of 49.6 percentage points compared with the baseline.

These results describe this test set; they do not guarantee the same performance on new tickets.

![Plain-language summary of ticket classification results](artifacts/reports/results_overview.png)

![Category classification confusion matrix](artifacts/reports/category_confusion_matrix.png)

## Priority Tagging

The dataset has category labels but no priority labels. The project therefore assigns priority using explicit keyword rules in `priority_rules.py`:

- **High:** outage, security breach, compromised account, emergency, urgent, critical, and similar severe terms.
- **Medium:** errors, failed services, access problems, and other issues.
- **Low:** messages that do not match the High or Medium rules.

These priority labels are heuristic estimates, not ground truth. The project does not report priority accuracy, and priority is not a trained machine-learning prediction.

## How this could help a support team

The predicted category could help staff send a ticket to the right support queue. The keyword-based priority label could flag urgent or security-related wording for earlier human review. A team member should check the suggestion before acting on it.

This prototype has not been integrated into a live help desk, and the project has not measured whether it reduces response time or staff workload.

## Setup in VS Code on Windows

Download the four dataset files from Zenodo and place them in the project's `data/` folder.

Open the project folder in VS Code. In the PowerShell terminal, create a virtual environment:

```powershell
python -m venv .venv
```

Install the required libraries:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train and evaluate the category model:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Try a ticket:

```powershell
.\.venv\Scripts\python.exe predict_ticket.py "Our production system is down and many users are affected"
```

The prediction output includes an ML-predicted category and a rule-based priority.

## Output Files

Running the training script creates:

```text
artifacts/
├── models/
│   └── category_model.joblib
└── reports/
    ├── results_overview.png
    ├── category_confusion_matrix.png
    └── metrics.json
```

The saved model is generated locally and is not included in the repository. The reports contain the evaluation results and visualizations.

## Limitations

The category model is a decision-support prototype. It can make mistakes, especially with multilingual or ambiguous tickets. Review the ticket and the model's results before taking action.

Priority depends on the listed keyword rules. Wording that the rules do not recognize may receive an unsuitable priority. The priority labels have not been evaluated against human-assigned priority labels.

## Project Information

- **Program:** Future Interns Machine Learning Internship
- **Track:** Machine Learning
- **Task:** Task 2 — Support Ticket Classification
- **Repository:** `FUTURE_ML_02`
