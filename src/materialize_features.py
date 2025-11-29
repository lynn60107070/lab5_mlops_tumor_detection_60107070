from datetime import datetime, timedelta

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import MaterializationComputeResource

SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60107070"
FEATURE_STORE_NAME = "lab5-featurestore-tumor-60107070"


def main():
    client = MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=FEATURE_STORE_NAME,
    )

    compute = MaterializationComputeResource(
        instance_type="Standard_E4s_v3"  # valid instance type
    )

    spark_conf = {
        "spark.driver.cores": "2",
        "spark.driver.memory": "4g",
        "spark.executor.instances": "2",
        "spark.executor.cores": "2",
        "spark.executor.memory": "4g",
    }

    start_time = datetime(2000, 1, 1)
    end_time = datetime.utcnow() - timedelta(hours=1)

    poller = client.feature_sets.begin_backfill(
        name="tumor_features",
        version="1",
        feature_window_start_time=start_time,
        feature_window_end_time=end_time,
        compute_resource=compute,
        spark_configuration=spark_conf,
        data_status=["None", "Incomplete"],
    )

    result = poller.result()
    print("Backfill started successfully!")

    # Azure returns a LIST of job IDs, even if it's just one job
    if hasattr(result, "job_ids"):
        print("Job IDs:", result.job_ids)
    elif hasattr(result, "job_id"):
        print("Job ID:", result.job_id)
    else:
        print("No job ID found in result. Raw object:", result)

if __name__ == "__main__":
    main()