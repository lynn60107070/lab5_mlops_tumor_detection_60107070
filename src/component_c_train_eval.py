import argparse
import json
import os
import glob
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

def load_single_parquet_from_folder(path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        return pd.read_parquet(path)
    files = glob.glob(os.path.join(path, "*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {path}")
    return pd.read_parquet(files[0])


def subset_xy(df: pd.DataFrame, feature_names, label_col="label"):
    X = df[feature_names].values
    y = df[label_col].values
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, required=True)
    parser.add_argument("--test_input", type=str, required=True)
    parser.add_argument("--selected_features_input", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    parser.add_argument("--feature_set_version", type=str, default="1")
    args = parser.parse_args()

    # Load data
    train_df = load_single_parquet_from_folder(args.train_input)
    test_df = load_single_parquet_from_folder(args.test_input)

    # Load selected features
    with open(args.selected_features_input, "r") as f:
        sel = json.load(f)
    selected_features = sel["selected_features"]

    # Extract X, y
    X_train, y_train = subset_xy(train_df, selected_features)
    X_test, y_test = subset_xy(test_df, selected_features)

    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Metrics dict
    metrics = {
        "accuracy": float(acc),
        "train_time_seconds": float(train_time),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "num_train_samples": int(len(train_df)),
        "num_test_samples": int(len(test_df)),
        "num_features": int(len(selected_features)),
        "selected_features": selected_features,
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "feature_set_version": args.feature_set_version,
    }

    # Save metrics
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)

    # Persist model
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.joblib")
    joblib.dump(clf, model_path)

    # Optional: write a model metadata json in the same folder
    meta_path = os.path.join(args.model_output, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "selected_features": selected_features,
                "feature_set_version": args.feature_set_version,
                "training_timestamp_utc": metrics["training_timestamp_utc"],
            },
            f,
            indent=2,
        )

    print(f"test_accuracy={acc}")
    print(f"confusion_matrix={cm.tolist()}")

    # TODO: model registration block (fill with azure.ai.ml MLClient code)
    # from azure.ai.ml import MLClient
    # from azure.identity import DefaultAzureCredential
    #
    # ml_client = MLClient(
    #     DefaultAzureCredential(),
    #     subscription_id=os.environ["AZUREML_SUBSCRIPTION_ID"],
    #     resource_group_name=os.environ["AZUREML_RESOURCE_GROUP"],
    #     workspace_name=os.environ["AZUREML_WORKSPACE_NAME"],
    # )
    #
    # model = ml_client.models.create_or_update(
    #     Model(
    #         name="tumor_detection_ga",
    #         path=args.model_output,
    #         type="mlflow_model" or "custom_model",
    #         tags={
    #             "selected_features": json.dumps(selected_features),
    #             "feature_set_version": args.feature_set_version,
    #             "training_timestamp_utc": metrics["training_timestamp_utc"],
    #         },
    #     )
    # )
    # print(f"Registered model: {model.name}:{model.version}")


if __name__ == "__main__":
    main()
