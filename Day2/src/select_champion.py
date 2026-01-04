import mlflow

EXPERIMENT = "churn-day2"

def main():
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)

    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["metrics.f1 DESC"]
    )

    champion = runs[0]

    client.set_tag(champion.info.run_id, "model_role", "champion")
    print(f"🥇 Champion selected: {champion.info.run_id}")

if __name__ == "__main__":
    main()
