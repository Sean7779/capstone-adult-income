import joblib
import pandas as pd


MODEL_PATH = "models/best_model.joblib"


def load_model(model_path=MODEL_PATH):
    model = joblib.load(model_path)
    return model


def predict_income(input_data, model_path=MODEL_PATH):
    model = load_model(model_path)

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": prediction,
        "probabilities": probabilities
    }


if __name__ == "__main__":
    sample_input = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    result = predict_income(sample_input)

    print("Prediction:", result["prediction"])
    print("Probabilities:", result["probabilities"])



    