# AI-Based Multiclass Disease Classification Using Symptoms

An educational machine-learning project that explores multiclass disease classification from symptom indicators. It includes exploratory data analysis, preprocessing, model training and comparison, XGBoost tuning, evaluation reports, and a Streamlit application.

> **Medical disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, rule out, or treat any disease. Model predictions may be incorrect. Consult a qualified healthcare professional for medical concerns.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Models and Results](#models-and-results)
- [Streamlit Application](#streamlit-application)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Run the Notebooks](#run-the-notebooks)
- [Reports and Artifacts](#reports-and-artifacts)
- [Important Note About Data and Models](#important-note-about-data-and-models)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [GitHub Notes](#github-notes)
- [Dataset Attribution and License](#dataset-attribution-and-license)

## Project Overview

The project uses symptom indicators as input features and a disease label as the target. It explores data preparation, trains and compares machine-learning classifiers, evaluates their performance, and provides an interactive Streamlit interface for experimenting with model predictions.

The final app allows a user to select symptoms from the dataset's feature list and submit them to a saved XGBoost classifier. The app displays the category predicted by the model and, when available, other model-ranked categories with their raw model scores.

The output represents patterns learned from the dataset. It is not a clinical assessment or a reliable estimate of a person's actual disease risk.

## Objectives

- Explore a multiclass classification problem using symptom features.
- Inspect and preprocess a structured healthcare-related dataset.
- Encode target disease labels for model training.
- Compare Random Forest, XGBoost, and CatBoost.
- Tune XGBoost hyperparameters using a validation split.
- Evaluate the classifier with accuracy, precision, recall, F1-score, and a confusion matrix.
- Build a Streamlit interface for demonstrating the trained model.
- Understand the effect of class imbalance and rare disease labels.

## Dataset

**Dataset filename:** `CommunityHealthcare_MultiSymptomsDisease_Diagnostic_Dataset.csv`

The dataset inspected during development had these characteristics:

| Property | Description |
|---|---|
| Records | 100,000 |
| Total columns | 175 |
| Symptom features | 174 binary features |
| Target column | `disease` |
| Disease classes | 46 |
| Symptom values | Binary indicators (0/1) |

Each row contains symptom indicators and a target disease label. The 174 symptom columns are model inputs; `disease` is the target.

### Preprocessing

The workflow included:

1. Inspecting the dataset's dimensions, columns, target classes, and feature values.
2. Checking for missing values and exact duplicate rows.
3. Encoding disease labels with `LabelEncoder`.
4. Splitting the data into training and test sets.
5. Saving processed features, labels, and the label encoder.

The initial inspection reported no missing values and no exact duplicate rows. These findings apply to the inspected dataset copy and should be rechecked if the data is changed.

### Class imbalance

Some target classes had very few records. Classes observed with fewer than ten records included `Metabolic_Syndrome`, `Urinary_Tract_Infection`, `Typhoid_Fever`, and `Psoriasis`.

This imbalance can make model performance uneven: results on common classes may look stronger than results on rare classes. Review macro-averaged and per-class metrics as well as accuracy and weighted averages.

## Workflow

1. **Exploratory data analysis** — inspect feature distributions, target counts, and data quality.
2. **Preprocessing** — prepare symptom features and encode the target.
3. **Baseline training** — train a Random Forest classifier.
4. **Model experiments** — train XGBoost and CatBoost models.
5. **Comparison and error analysis** — compare metrics and inspect errors.
6. **Optimization and tuning** — test alternative XGBoost configurations.
7. **Final training** — train and save the selected XGBoost model.
8. **Evaluation** — produce summary metrics, a class-wise report, and a confusion matrix.
9. **Streamlit app** — load the saved model and accept user-selected symptoms.

## Models and Results

Recorded results from the initial model-comparison experiment:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Random Forest | 0.3523 | 0.2229 | 0.3567 |
| XGBoost (initial configuration) | 0.6671 | 0.3446 | 0.6577 |
| CatBoost | 0.5589 | 0.3136 | 0.5679 |

During XGBoost tuning, the best configuration by macro F1 achieved approximately **0.3940 macro F1** on the tuning validation split. A different configuration achieved the highest validation accuracy of approximately **0.6769**. These are results from separate experiments and should not be treated as guaranteed future results.

### Metrics

- **Accuracy:** fraction of predictions that are correct.
- **Precision:** fraction of predictions for a class that are correct.
- **Recall:** fraction of actual examples of a class that are identified.
- **F1-score:** harmonic mean of precision and recall.
- **Macro average:** weights every class equally, making rare-class performance visible.
- **Weighted average:** weights each class by its number of true examples.
- **Confusion matrix:** summarizes true versus predicted class counts.

The test set was used for model comparison during development. Consequently, evaluation on that same test set is exploratory rather than a fully independent estimate of generalization. A stronger study should use a fresh, untouched test set or a carefully designed validation strategy.

## Streamlit Application

The application is implemented in `app.py` and uses the saved XGBoost model.

### What the app shows

- A styled MediSense AI dashboard with a healthcare-inspired navy and teal theme.
- Project and model information.
- The number of symptom features and disease categories.
- A searchable multi-select control for symptoms.
- A summary of selected symptoms.
- A button to run the model prediction.
- The predicted disease category.
- Up to five model-ranked categories with raw model score outputs, when supported by the model.
- Medical disclaimers explaining the experimental nature of the predictions.

### How input is prepared

The app reads the feature-column order from `data/processed/X_train.csv`, creates a single row with all symptom indicators set to zero, and sets selected symptoms to one. Keeping the feature order consistent with training is essential.

Raw classifier scores should not be interpreted as calibrated probabilities that a person has a disease or as clinical risk estimates.

## Project Structure

The repository is designed to contain source code, notebooks, reports, and documentation. **The `data/` and `models/` directories are intentionally excluded from GitHub** to avoid publishing the dataset and large trained-model files.

```text
HealthCare/
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
├── data/                         # Local only; ignored by Git
│   ├── raw/
│   │   └── CommunityHealthcare_MultiSymptomsDisease_Diagnostic_Dataset.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── label_mapping.csv
│       └── label_encoder.joblib
├── models/                       # Local only; ignored by Git
│   ├── final_xgboost_model.joblib
│   ├── final_xgboost_params.joblib
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   └── catboost_model.joblib
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_xgboost_training.ipynb
│   ├── 06_catboost_training.ipynb
│   ├── 07_model_comparison.ipynb
│   ├── 08_xgboost_error_analysis.ipynb
│   ├── 09_model_optimization.ipynb
│   ├── 10_xgboost_tuning.ipynb
│   ├── 11_final_xgboost_model.ipynb
│   └── 12_final_model_evaluation.ipynb
└── reports/
    ├── catboost_metrics.csv
    ├── model_comparison.csv
    ├── random_forest_metrics.csv
    ├── xgboost_metrics.csv
    ├── xgboost_weighted_metrics.csv
    ├── final_xgboost_metrics.csv
    ├── final_xgboost_classification_report.csv
    └── final_xgboost_confusion_matrix.png
```

This is a representative layout. Additional reports may exist locally, and exact artifact names depend on which notebooks have been run.

## Requirements

- Python 3.10 or 3.11 is a reasonable starting point; use a version supported by the installed versions of XGBoost, pandas, scikit-learn, and Streamlit.
- `pip`
- VS Code or another Python IDE
- Local copies of the processed dataset files and model artifacts to run the existing app

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/VedBisne10/Multiclass_Disease_Classification.git
cd Multiclass_Disease_Classification
```

Or open your existing local project directory.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use your preferred terminal or follow your local execution-policy guidance.

### 3. Install dependencies

The app and notebooks use packages such as Streamlit, pandas, NumPy, scikit-learn, XGBoost, joblib, Matplotlib, CatBoost, and Jupyter.

Install the dependencies listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a package is missing, install it into the activated environment. For reproducibility, record and review package versions used for a working run.

## Run the Application

The existing Streamlit app expects these local files:

- `models/final_xgboost_model.joblib`
- `data/processed/label_encoder.joblib`
- `data/processed/X_train.csv`

Since `data/` and `models/` are not stored in GitHub, cloning the repository alone is **not enough to run the app**. Obtain the dataset and compatible artifacts from your own secure local backup or regenerate them by running the notebooks in sequence.

From the project root, run:

```bash
streamlit run app.py
```

Streamlit usually provides the local address `http://localhost:8501` in the terminal. Open it in a browser if it does not open automatically.

### Using the app

1. Wait for the model files to load.
2. Search for and select one or more symptoms.
3. Click **Run model prediction**.
4. Review the predicted category and any optional model-ranked categories.
5. Treat the output as an experimental result only, not as medical advice.

## Run the Notebooks

Open the project in VS Code, select the virtual environment's Python interpreter, and run the notebooks under `notebooks/` in order where applicable.

The notebooks cover data exploration, preprocessing, model training, evaluation, comparison, error analysis, optimization, XGBoost tuning, final training, and final evaluation.

Run the notebook that creates each required file before running a later notebook or app that depends on it. Since the dataset and models are ignored by Git, the required local files must be present or regenerated.

## Reports and Artifacts

The `reports/` directory is included in the repository for analysis and presentation. It may contain:

- Model comparison metrics.
- Per-model evaluation metrics.
- Classification reports.
- Confusion matrix visualizations.
- Feature-importance plots.
- XGBoost tuning results.
- Misclassified-record analysis.

Reports are experimental outputs and may not exactly match a new run if data, package versions, random seeds, or training settings change. Check whether a report contains any sensitive row-level data before publishing it.

## Important Note About Data and Models

The `.gitignore` file excludes `data/` and `models/` from GitHub to avoid uploading the dataset and large trained models. These files should remain in the local project directory for the current app to work.

To verify that key files are ignored, run:

```powershell
git check-ignore data/raw/CommunityHealthcare_MultiSymptomsDisease_Diagnostic_Dataset.csv
git check-ignore models/final_xgboost_model.joblib
```

Do not force-add these files unless you have a deliberate, safe, and appropriate reason to publish them. Confirm dataset redistribution rights and consider model file size before sharing artifacts elsewhere.

## Limitations

- **Not for medical use:** the application has not been validated for clinical decision-making.
- **Class imbalance:** rare classes have too few examples for reliable evaluation.
- **Generalization:** dataset patterns may not represent real-world patients or populations.
- **Model scores:** raw classifier outputs are not necessarily calibrated clinical probabilities.
- **Input limitations:** the app only accepts binary symptom features from the dataset and does not consider full medical history, physical examination, lab results, age, or other clinical context.
- **Evaluation limitations:** the test set was used during model comparison, so an untouched test set is recommended for a stronger final assessment.
- **Data rights:** verify the dataset's original source, license, terms, and privacy requirements before redistribution.

## Future Improvements

- Evaluate on a fresh, untouched test set.
- Improve evaluation for rare classes and clearly report when there is insufficient evidence.
- Add detailed class-wise visualizations and error analysis.
- Assess probability calibration before describing model scores as probabilities.
- Add automated tests for model loading, feature order, and input validation.
- Track dataset and model versions.
- Improve accessibility and usability of the symptom selector.
- Require rigorous external validation and qualified clinical review before considering any real-world health application.

## GitHub Notes

The intended repository contents are source code, notebooks, reports, and documentation. Dataset and model files are intentionally excluded.

To commit future source-code changes:

```bash
git status
git add README.md app.py requirements.txt .gitignore notebooks reports
git commit -m "Update project documentation and source"
git push
```

Check `git status` before committing and verify that `data/` and `models/` are not staged. `.gitignore` does not erase files that were included in earlier commits; this repository was prepared with a clean history to avoid pushing the oversized artifacts.

## Dataset Attribution and License

Add a software license if you intend to publish the code. Verify the dataset's original source and license and provide attribution as required. This README does not assert ownership or redistribution rights for the dataset.
