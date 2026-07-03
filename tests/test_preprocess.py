from src.input_prep import load_training_stats, prepare_model_input


def test_load_training_stats_returns_expected_keys():
    stats = load_training_stats("data/adult.csv")

    assert "fnlwgt_median" in stats
    assert "education_to_num" in stats
    assert stats["fnlwgt_median"] > 0
    assert isinstance(stats["education_to_num"], dict)


def test_prepare_model_input_fills_fnlwgt_from_stats():
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

    assert model_input["fnlwgt"] == stats["fnlwgt_median"]


def test_prepare_model_input_maps_education_to_education_num():
    stats = load_training_stats("data/adult.csv")

    raw_features = {
        "education": "Bachelors"
    }

    model_input = prepare_model_input(raw_features, stats)

    assert model_input["education"] == "Bachelors"
    assert model_input["education-num"] == stats["education_to_num"]["Bachelors"]


def test_prepare_model_input_applies_reasonable_defaults():
    stats = load_training_stats("data/adult.csv")

    raw_features = {
        "education": "Masters"
    }

    model_input = prepare_model_input(raw_features, stats)

    assert model_input["age"] == 40
    assert model_input["workclass"] == "Private"
    assert model_input["marital-status"] == "Never-married"
    assert model_input["occupation"] == "Unknown"
    assert model_input["relationship"] == "Not-in-family"
    assert model_input["race"] == "White"
    assert model_input["sex"] == "Male"
    assert model_input["capital-gain"] == 0
    assert model_input["capital-loss"] == 0
    assert model_input["hours-per-week"] == 40
    assert model_input["native-country"] == "United-States"

    