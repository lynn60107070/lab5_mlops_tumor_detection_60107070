# pipeline_job.py
# Gold pipeline WITHOUT Feature Store materialization.
# Uses Silver features data asset directly, writes all Gold outputs
# into storage account: lab5tumorstor60107070, container: raw, under directory: gold/

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, load_component
from azure.ai.ml.dsl import pipeline

# --------------------------------------------------------------------
# Workspace configuration
# --------------------------------------------------------------------
SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60107070"
WORKSPACE_NAME = "GoodReadsReview-Analysis-60107070"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# --------------------------------------------------------------------
# Silver features location (Phase 2 output data asset)
# From job:
# Data asset: azureml_0d1ae5ce-25f4-43fa-bd75-4e4116a8764d_output_data_features_output:1
# Asset URI:
#   azureml:azureml_0d1ae5ce-25f4-43fa-bd75-4e4116a8764d_output_data_features_output:1
# --------------------------------------------------------------------
FEATURES_URI = "azureml:azureml_0d1ae5ce-25f4-43fa-bd75-4e4116a8764d_output_data_features_output:1"

# --------------------------------------------------------------------
# Datastore for Gold outputs
# Existing datastore mapping:
#   name: lab5_adls_60107070
#   account_name: lab5tumorstor60107070
#   container_name: raw
# --------------------------------------------------------------------
GOLD_BASE_PATH = "azureml://datastores/lab5_adls_60107070/paths/gold"

# --------------------------------------------------------------------
# Load components from YAML
# --------------------------------------------------------------------
feature_retrieval_comp = load_component(
    source="components/component_a_feature_retrieval.yml"
)

feature_selection_comp = load_component(
    source="components/component_b_feature_selection.yml"
)

train_eval_comp = load_component(
    source="components/component_c_train_eval.yml"
)

# --------------------------------------------------------------------
# Pipeline definition
# --------------------------------------------------------------------
@pipeline(
    default_compute="goodreads-vm60107070",
    description="Tumor detection Gold pipeline (Silver → split → GA → train/eval, outputs to gold/)",
)
def tumor_gold_pipeline():

    # Silver-layer features data asset: image_id, label, f1..fN
    features_input = Input(type="uri_folder", path=FEATURES_URI)
    feature_set_version = "1"

    # Component A: Silver → train/test split
    retrieval_job = feature_retrieval_comp(
        features_input=features_input,
    )

    # Component B: Baseline + GA feature selection on train set
    selection_job = feature_selection_comp(
        train_input=retrieval_job.outputs.train_output,
    )

    # Component C: Train final model with GA-selected features, evaluate on test
    train_eval_job = train_eval_comp(
        train_input=retrieval_job.outputs.train_output,
        test_input=retrieval_job.outputs.test_output,
        selected_features_input=selection_job.outputs.selected_features_output,
        feature_set_version=feature_set_version,
    )

    # ----------------------------------------------------------------
    # Route all important outputs into storage account lab5tumorstor60107070
    # container raw, directory gold/...
    # ----------------------------------------------------------------

    # Train/test parquet
    retrieval_job.outputs.train_output = Output(
        type="uri_folder",
        path=f"{GOLD_BASE_PATH}/data/train/",
    )

    retrieval_job.outputs.test_output = Output(
        type="uri_folder",
        path=f"{GOLD_BASE_PATH}/data/test/",
    )

    # Metrics from feature selection
    selection_job.outputs.ga_metrics_output = Output(
        type="uri_file",
        path=f"{GOLD_BASE_PATH}/metrics/ga_metrics.json",
    )

    selection_job.outputs.baseline_metrics_output = Output(
        type="uri_file",
        path=f"{GOLD_BASE_PATH}/metrics/baseline_metrics.json",
    )

    # Final model + metrics
    train_eval_job.outputs.model_output = Output(
        type="uri_folder",
        path=f"{GOLD_BASE_PATH}/model_output/",
    )

    train_eval_job.outputs.metrics_output = Output(
        type="uri_file",
        path=f"{GOLD_BASE_PATH}/metrics/metrics.json",
    )

    return {
        "model_output": train_eval_job.outputs.model_output,
        "metrics_output": train_eval_job.outputs.metrics_output,
        "ga_metrics": selection_job.outputs.ga_metrics_output,
        "baseline_metrics": selection_job.outputs.baseline_metrics_output,
    }


# --------------------------------------------------------------------
# Submit pipeline
# --------------------------------------------------------------------
if __name__ == "__main__":
    job = tumor_gold_pipeline()
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Pipeline job submitted: {returned_job.name}")
