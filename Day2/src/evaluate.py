import mlflow
import pandas as pd

EXPERIMENT_NAME = "churn-day2"
METRIC = "f1"

def main():
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.candidate = 'true'",
        order_by=[f"metrics.{METRIC} DESC"]
    )

    best = runs[0]
    print(f"🏆 Best model run_id: {best.info.run_id}")
    print(f"{METRIC}: {best.data.metrics[METRIC]}")

if __name__ == "__main__":
    main()
