import yaml
import mlflow
import mlflow.sklearn
import joblib

from src.preprocess import prepare_data

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def create_model(model_name, params):
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    elif model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
def save_model(model, model_path="models/best_model.joblib"):

 joblib.dump(model, model_path)
 print(f"\nBest model saved to: {model_path}")


def train_and_evaluate_model(model_name, classifier, params, X_train, X_test, y_train, y_test, preprocessor):
    with mlflow.start_run(run_name=model_name):
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", classifier),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=">50K")
        recall = recall_score(y_test, y_pred, pos_label=">50K")
        f1 = f1_score(y_test, y_pred, pos_label=">50K")

        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        #mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n{model_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")



        return {
            "model_name": model_name,
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def run_all_models(config_path="configs/config.yaml"):
    config = load_config(config_path)

    mlflow.set_experiment(config["training"]["experiment_name"])

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(config_path)

    model_configs = config["training"]["models"]
    results = []

    for model_config in model_configs:
        model_name = model_config["name"]
        params = model_config["params"]

        classifier = create_model(model_name, params)

        result = train_and_evaluate_model(
            model_name=model_name,
            classifier=classifier,
            params=params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
        )
        results.append(result)

    best_result = max(results, key=lambda x: x["f1"])

    print("\nBest model based on F1-score:")
    print(f"{best_result['model_name']} with F1-score = {best_result['f1']:.4f}")

    save_model(best_result["model"])


if __name__ == "__main__":
    run_all_models() 

    
















