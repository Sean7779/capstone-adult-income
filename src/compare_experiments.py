import mlflow
import pandas as pd


EXPERIMENT_NAME = "adult_income_capstone"


def compare_runs():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"]
    )

    if runs_df.empty:
        print("No runs found in the experiment.")
        return

    finished_runs = runs_df[runs_df["status"] == "FINISHED"]

    if finished_runs.empty:
        print("No FINISHED runs found in the experiment.")
        return

    columns_to_show = [
        "run_id",
        "params.model_name",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1_score",
        "status",
        "start_time"
    ]

    available_columns = [col for col in columns_to_show if col in finished_runs.columns]

    print("\nFINISHED runs sorted by F1-score:\n")
    print(finished_runs[available_columns].to_string(index=False))

    best_run = finished_runs.iloc[0]

    print("\nBest FINISHED run:")
    print(f"Run ID:      {best_run['run_id']}")
    print(f"Model Name:  {best_run['params.model_name']}")
    print(f"Accuracy:    {best_run['metrics.accuracy']:.4f}")
    print(f"Precision:   {best_run['metrics.precision']:.4f}")
    print(f"Recall:      {best_run['metrics.recall']:.4f}")
    print(f"F1-score:    {best_run['metrics.f1_score']:.4f}")


if __name__ == "__main__":
    compare_runs() 

    

