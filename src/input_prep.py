import pandas as pd


def load_training_stats(data_path="data/adult.csv"):
    df = pd.read_csv(data_path)

    fnlwgt_median = df["fnlwgt"].median()

    education_counts = df["education"].value_counts()
    education_to_num = {}

    for edu in education_counts.index:
        subset = df[df["education"] == edu]
        education_to_num[edu] = subset["education-num"].median()

    return {
        "fnlwgt_median": fnlwgt_median,
        "education_to_num": education_to_num
    }


def prepare_model_input(raw_features, stats):
    education = raw_features.get("education")

    if education is not None:
        education_num = stats["education_to_num"].get(education, 0)
    else:
        education_num = 0

    fnlwgt = raw_features.get("fnlwgt", stats["fnlwgt_median"])

    model_input = {
        "age": raw_features.get("age", 40),
        "workclass": raw_features.get("workclass", "Private"),
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": education_num,
        "marital-status": raw_features.get("marital-status", "Never-married"),
        "occupation": raw_features.get("occupation", "Unknown"),
        "relationship": raw_features.get("relationship", "Not-in-family"),
        "race": raw_features.get("race", "White"),
        "sex": raw_features.get("sex", "Male"),
        "capital-gain": raw_features.get("capital-gain", 0),
        "capital-loss": raw_features.get("capital-loss", 0),
        "hours-per-week": raw_features.get("hours-per-week", 40),
        "native-country": raw_features.get("native-country", "United-States"),
    }

    return model_input


if __name__ == "__main__":
    stats = load_training_stats()

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

    print("Model input:")
    for k, v in model_input.items():
        print(f"{k}: {v}") 



        