"""Serves the flows on schedules, visible in the Prefect UI.

    python orchestration/deploy.py

  - continuous-training: every Monday 03:00 (checks monitoring, retrains only if needed)
  - training-pipeline:   first day of each month 04:00, with a data refresh
A separate worker is not needed: `serve` runs the schedules from this process.
"""
from flows import continuous_training, training_pipeline
from prefect import serve

if __name__ == "__main__":
    serve(
        continuous_training.to_deployment(name="weekly-monitoring-check", cron="0 3 * * 1",
                                          description="Checks drift and forecast accuracy; retrains only if needed."),
        training_pipeline.to_deployment(name="monthly-full-retrain", cron="0 4 1 * *", parameters={"refresh": True},
                                        description="Refreshes the data and retrains everything."),
    )
