Adult Income Prediction Assistant (Capstone Project)
This repo contains my capstone project: an app that predicts whether someone’s income is likely to be above or below $50K/year based on their demographic and work information. It uses a trained machine learning model and a simple natural language interface.

The project uses the Adult Income dataset from the UCI Machine Learning Repository. I built data preprocessing and model training code, use MLflow for experiment tracking, a rule‑based text parser, and a Streamlit web app to tie everything together.

1. Project description
The app is meant to explore income patterns based on demographic and work features. A user can describe a person in plain English (age, education, work hours, country, etc.), and the system:

Parses the text into structured features.

Sends those features to a trained Gradient Boosting model.

Returns a prediction (<=50K or >50K) and class probabilities.

Shows the parsed features and asks for more info when key fields are missing instead of guessing.

It’s focused specifically on the Adult Income task, not meant to be a general chatbot.

2. Setup
2.1 Prerequisites
Python 3.12

Git

(Recommended) a virtual environment like venv or conda

2.2 Install dependencies
From the project root:

bash
python -m pip install -r requirements.txt
2.3 Data and models
Put the Adult dataset CSV at data/adult.csv (UCI Adult Income dataset).

Put the trained model at models/best_model.joblib.
This file should match the best MLflow run.

These files are not committed to Git to avoid storing raw data and large binaries in the repo.

2.4 Environment variables
There is a .env.example file as a template.
Right now the app uses a local rule‑based parser instead of calling a live LLM API, so you don’t actually need an API key to run it.

If a cloud LLM is added later, its keys should go into a .env file (ignored by Git) and be loaded from environment variables.

3. How to run it
3.1 Run tests
To check preprocessing, model, and interface logic:

bash
python -m pytest tests/ -v
This should run 8 tests:

4 preprocessing tests

2 model tests

2 interface tests

3.2 Launch the Streamlit app
From the project root:

bash
streamlit run src/app.py
Then open the URL printed in the terminal, usually:

http://localhost:8501

3.3 Example input
In the text box, you can try something like:

text
I am a 45-year-old married male working in private industry as an executive manager. I have a masters degree, I work 50 hours per week, and I live in the United States.
The app will:

Parse this into Adult Income features (age, workclass, education, hours per week, etc.).

Run those features through the Gradient Boosting model.

Show, for example:

Predicted income class: >50K

Estimated probability of income >50K: ~0.85

Estimated probability of income <=50K: ~0.15

The parsed feature values used for the prediction

If the description is missing important fields (for example workclass or hours per week), the app lists the missing fields and asks for clarification instead of returning a prediction.

4. Code layout (architecture overview)
The repo uses a src directory with separate modules for preprocessing, training, evaluation, prediction, and the interface.

4.1 Main modules
src/preprocess.py
Cleans the Adult dataset and does feature engineering.

src/train.py
Trains several models (logistic regression, random forest, gradient boosting), logs metrics and models to MLflow, and identifies the best run.

src/evaluate.py
Helper functions to report metrics on a held‑out test set (accuracy, precision, recall, F1, etc.).

src/input_prep.py

load_training_stats() loads the Adult dataset and computes:

Median fnlwgt

A mapping from education to education-num

prepare_model_input() converts parsed features into a single model‑ready row and fills in reasonable defaults when needed.

src/predict.py

Loads models/best_model.joblib with joblib.

Wraps the input row in a pandas.DataFrame.

Calls model.predict() and model.predict_proba().

Returns a dict with prediction and probabilities.

src/llm_parser.py

Rule‑based natural language parser that:

Extracts features like age, workclass, education, marital status, occupation, relationship, sex, hours per week, and native country from free‑form text.

Tracks which fields are missing, sets clarification_needed, and builds a clarification question.

The idea is that later this could be replaced or extended with a real LLM without changing the rest of the app much.

src/app.py

Streamlit app that:

Accepts user text.

Calls parse_user_text() from llm_parser.py.

Handles out‑of‑scope or incomplete queries.

Uses prepare_model_input() and predict_income() to run the model.

Displays predictions, probabilities, and parsed features.

src/compare_experiments.py

Uses MLflow to:

Query runs for the adult_income_capstone experiment with mlflow.search_runs().

Sort runs by F1 score.

Print a table of FINISHED runs.

Show the best FINISHED run’s metrics and model name.

4.2 MLflow tracking
The training code logs:

Hyperparameters (model type and settings)

Metrics (accuracy, precision, recall, F1)

Model artifacts (saved models)

Run status and start time

compare_experiments.py uses these logs to find the best run by F1 score, which is what I use to select models/best_model.joblib.

5. Results (current best model)
On the Adult Income dataset, the best performing model so far is:

Model: Gradient Boosting

Best FINISHED run (example):

Accuracy: ≈ 0.8699

Precision: ≈ 0.7959

Recall: ≈ 0.6138

F1‑score: ≈ 0.6931

These metrics are logged in MLflow, and compare_experiments.py prints the runs and highlights the best one.

6. Reflection
What I learned

Building an end‑to‑end ML pipeline: preprocessing, training, experiment tracking, interface, and tests.

Using MLflow to compare experiments and choose a model based on the right metrics.

Connecting a natural language interface to a tabular model and handling incomplete user inputs.

Challenges

Dealing with external LLM APIs (keys, costs, and stability).

Keeping imports and environments in sync between the app, training scripts, and tests.

Designing a parser that maps conversational text into the Adult dataset schema.

What I’d improve next

Swap the rule‑based parser for a cloud LLM that’s wired in via environment variables, once I’m comfortable with the API and billing.

Support more flexible phrasing and give stronger explanations of why the model predicted a certain class.

Add Docker so the app can be run with a single docker build / docker run for easier deployment.

7. Repository structure
text
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
Data and model files stay out of Git on purpose, to avoid committing large or sensitive artifacts. In a production workflow I’d consider using a tool like DVC or just .gitignore rules to manage them.








