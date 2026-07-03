from src.input_prep import load_training_stats, prepare_model_input
from src.predict import predict_income


def test_predict_income_returns_expected_keys():
    stats = load_training_stats("data/adult.csv")

    raw_features = {
        "age": 45,
        "workclass": "Private",
        "education": "Masters",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }

    model_input = prepare_model_input(raw_features, stats)
    result = predict_income(model_input)

    assert "prediction" in result
    assert "probabilities" in result


def test_predict_income_probabilities_look_valid():
    stats = load_training_stats("data/adult.csv")

    raw_features = {
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    model_input = prepare_model_input(raw_features, stats)
    result = predict_income(model_input)

    probs = result["probabilities"]

    assert len(probs) == 2
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    assert abs((probs[0] + probs[1]) - 1.0) < 1e-6




