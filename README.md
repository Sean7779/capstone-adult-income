# Adult Income Prediction Assistant (Capstone Project)

This repository contains my end‑to‑end capstone project: an intelligent application that predicts whether a person’s income is likely to be above or below \$50K/year based on demographic and work information, with a natural language interface on top of a trained machine learning model.

The project uses the **Adult Income dataset** from the UCI Machine Learning Repository as the core data source and integrates data preprocessing, model training, experiment tracking with MLflow, a rule‑based natural language parser, and a Streamlit web app.

---

## 1. Project Description

The application is designed for exploring income patterns based on demographic and employment attributes. A user can describe a person in plain English (for example: age, education, work hours, and country), and the system:

1. Parses the text into structured features.
2. Runs those features through a trained Gradient Boosting model.
3. Returns a prediction (`<=50K` or `>50K`) and class probabilities.
4. Shows the parsed features and handles incomplete inputs by asking for clarifications instead of guessing.

This is a focused tool for the Adult Income prediction task, not a general chatbot.

---

## 2. Setup Instructions

### 2.1 Prerequisites

- Python 3.12
- Git
- Recommended: a virtual environment (venv or conda)

### 2.2 Install dependencies

From the project root:

```bash
python -m pip install -r requirements.txt
```

### 2.3 Data and Models

- Place the Adult dataset CSV at `data/adult.csv` (UCI Adult Income dataset).
- Place the trained model file at `models/best_model.joblib`.  
  (This is the best model chosen via MLflow experiments.)

These files are *not* committed to Git to avoid storing raw data and binary artifacts in the repository.

### 2.4 Environment variables

This project includes a `.env.example` file as a template.  
Currently, the application uses a **local rule‑based parser instead of a live LLM API**, so no external API key is required to run the app.

If a cloud LLM provider is used later, API keys should be added to a `.env` file (not committed) and loaded via environment variables.

---

## 3. Usage Instructions

### 3.1 Run tests

To verify preprocessing, model, and interface logic:

```bash
python -m pytest tests/ -v
```

You should see all 8 tests passing:
- 4 preprocessing tests
- 2 model tests
- 2 interface tests

### 3.2 Launch the Streamlit app

From the project root:

```bash
streamlit run src/app.py
```

Open the URL printed in the terminal, usually:

- Local: `http://localhost:8501`

### 3.3 Example interaction

In the text box, you can enter:

```text
I am a 45-year-old married male working in private industry as an executive manager. I have a masters degree, I work 50 hours per week, and I live in the United States.
```

The app will:

- Parse this into Adult Income features (age, workclass, education, hours per week, etc.).
- Run the features through the trained Gradient Boosting model.
- Show something like:

- Predicted income class: `>50K`
- Estimated probability of income >50K: ~0.85
- Estimated probability of income <=50K: ~0.15
- Parsed features used for the prediction.

If the description is incomplete (for example, missing workclass or hours per week), the app will list missing fields and ask for clarification instead of returning a prediction.

---

## 4. Architecture Overview

The repo follows a `src` layout with separate modules for preprocessing, training, evaluation, prediction, and the interface.

### 4.1 Key modules

- `src/preprocess.py`  
  Data cleaning and feature engineering for the Adult dataset.

- `src/train.py`  
  Trains several model configurations (logistic regression, random forest, gradient boosting), logs metrics and model artifacts to MLflow, and registers the best run.

- `src/evaluate.py`  
  Evaluation utilities for reporting metrics on a held‑out test set (accuracy, precision, recall, F1, etc.).

- `src/input_prep.py`  
  - `load_training_stats()` loads the Adult dataset and computes:
    - Median `fnlwgt`
    - Mapping from `education` to `education-num`
  - `prepare_model_input()` converts parsed features into a single row of model input, filling in reasonable defaults when necessary.

- `src/predict.py`  
  - Loads `models/best_model.joblib` with `joblib`.
  - Wraps the prepared input in a `pandas.DataFrame`.
  - Calls `model.predict()` and `model.predict_proba()`.
  - Returns a dictionary with `prediction` and `probabilities`.

- `src/llm_parser.py`  
  - A **rule‑based natural language parser** that:
    - Extracts features like age, workclass, education, marital status, occupation, relationship, sex, hours per week, and native country from free‑form text.
    - Marks missing fields, sets `clarification_needed`, and generates a clarification question.
  - Designed so a cloud LLM could replace or augment it in the future without changing the app’s structure.

- `src/app.py`  
  - Streamlit application:
    - Accepts user text.
    - Calls `parse_user_text()` from `llm_parser.py`.
    - Handles out‑of‑scope or incomplete queries.
    - Uses `prepare_model_input()` and `predict_income()` to run the model.
    - Displays prediction, probabilities, and parsed features.

- `src/compare_experiments.py`  
  - Uses MLflow’s Python API to:
    - Query runs for the `adult_income_capstone` experiment via `mlflow.search_runs()`.
    - Sort runs by F1‑score.
    - Print a table of FINISHED runs.
    - Identify and print the best FINISHED run’s metrics and model name.

### 4.2 MLflow experiment tracking

Training scripts log:

- Hyperparameters (model type, settings).
- Metrics (accuracy, precision, recall, F1).
- Model artifacts (saved models).
- Run status and start time.

`compare_experiments.py` programmatically identifies the best run based on F1‑score and its metrics, which is used to choose `models/best_model.joblib`.

---

## 5. Results Summary

Using the Adult Income dataset and several model configurations, the best performing model was:

- **Model:** Gradient Boosting
- **Best FINISHED run (example):**
  - Accuracy: ≈ 0.8699
  - Precision: ≈ 0.7959
  - Recall: ≈ 0.6138
  - F1‑score: ≈ 0.6931

These metrics are logged and viewable in MLflow, and `compare_experiments.py` prints the ranking of runs and highlights the best one.

---

## 6. Reflection

- **What I learned:**
  - How to design an end‑to‑end ML system: preprocessing, training, experiment tracking, interface, and testing.
  - How to use MLflow to compare experiments and pick the best model based on appropriate metrics.
  - How to build a natural language interface on top of a tabular model and handle incomplete user inputs gracefully.

- **Challenges:**
  - Integrating external LLM APIs while managing API keys and billing constraints.
  - Keeping imports and environments consistent between the app, training scripts, and tests.
  - Designing a parser that maps conversational text into the specific feature schema of the Adult dataset.

- **What I would improve with more time:**
  - Replace the rule‑based parser with a cloud LLM provider configured via environment variables, once credential and billing constraints are resolved.
  - Extend the interface to support more flexible phrasing and richer explanations of feature importance.
  - Containerize the application with Docker so it can be run with a single `docker build` / `docker run` command for easier deployment.

---

## 7. Repository Structure

```text
capstone-adult-income/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   └── config.yaml
├── data/
│   └── adult.csv           # (not committed; local file only)
├── models/
│   └── best_model.joblib   # (not committed; local file only)
├── notebooks/
│   └── exploration.ipynb   # optional EDA / experimentation
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── input_prep.py
│   ├── predict.py
│   ├── llm_parser.py
│   ├── app.py
│   └── compare_experiments.py
└── tests/
    ├── test_preprocess.py
    ├── test_model.py
    └── test_interface.py
```

Data and model artifacts are kept out of Git to avoid committing large or sensitive files; DVC or `.gitignore` can be used to manage them in a production workflow.




