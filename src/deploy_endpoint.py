from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.core.exceptions import ResourceNotFoundError
import os
import datetime

# Workspace config (can also be provided via env, but constants are fine here)
SUBSCRIPTION_ID = os.getenv("AZUREML_SUBSCRIPTION_ID", "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3")
RESOURCE_GROUP = os.getenv("AZUREML_RESOURCE_GROUP", "rg-60107070")
WORKSPACE_NAME = os.getenv("AZUREML_WORKSPACE_NAME", "GoodReadsReview-Analysis-60107070")

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# Gold model location in your datastore (lab5_adls_60107070 / raw / gold/model_output/)
MODEL_PATH = "azureml://datastores/lab5_adls_60107070/paths/gold/model_output/"
MODEL_NAME = "tumor_ga_model"
ENDPOINT_NAME = "tumor-endpoint-60107070"
DEPLOYMENT_NAME = "blue"


def register_model():
    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model = Model(
        name=MODEL_NAME,
        path=MODEL_PATH,
        type="custom_model",
        description="Tumor GA-selected model from Gold pipeline",
        tags={"source": "lab5_gold_pipeline", "registered_at_utc": ts},
    )
    registered = ml_client.models.create_or_update(model)
    return registered


def get_or_create_endpoint():
    try:
        endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
        return endpoint
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            auth_mode="key",
            description="Tumor classification endpoint (GA-selected model)",
        )
        poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
        endpoint = poller.result()
        return endpoint


def get_or_create_environment():
    env_name = "tumor-endpoint-env"
    env_version = "2"
    try:
        env = ml_client.environments.get(name=env_name, version=env_version)
        return env
    except ResourceNotFoundError:
        env = Environment(
            name=env_name,
            version=env_version,
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file="envs/endpoint_env.yml",
        )
        env = ml_client.environments.create_or_update(env)
        return env


def create_or_update_deployment(model, endpoint, environment):
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model.id,
        environment=environment.id,
        code_configuration=CodeConfiguration(
            code="src",
            scoring_script="score.py",
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )

    poller = ml_client.online_deployments.begin_create_or_update(deployment)
    deployment_result = poller.result()

    # Route 100% traffic to this deployment
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    poller_ep = ml_client.online_endpoints.begin_create_or_update(endpoint)
    endpoint_result = poller_ep.result()

    return deployment_result, endpoint_result


def main():
    model = register_model()
    endpoint = get_or_create_endpoint()
    env = get_or_create_environment()
    create_or_update_deployment(model, endpoint, env)
    print(f"Endpoint {ENDPOINT_NAME} deployed with model {model.name}:{model.version}")


if __name__ == "__main__":
    main()