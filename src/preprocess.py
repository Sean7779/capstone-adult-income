import yaml
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_data(data_path):
    df = pd.read_csv(data_path)
    # Normalize income labels: strip spaces and trailing dots
    df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
    return df




def split_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def prepare_data(config_path="configs/config.yaml"):
    config = load_config(config_path)
    df = load_data(config["dataset"]["path"])

    X, y = split_features_target(df, config["dataset"]["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y
    )

    preprocessor = build_preprocessor(X)

    return X_train, X_test, y_train, y_test, preprocessor


