import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


def encode_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        b = f.read()
    return base64.b64encode(b).decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, default="blue")
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Local folder with subfolders yes/ and no/ containing test images.",
    )
    args = parser.parse_args()

    # Workspace config from env (same as training)
    subscription_id = os.getenv("AZUREML_SUBSCRIPTION_ID", "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3")
    resource_group = os.getenv("AZUREML_RESOURCE_GROUP", "rg-60107070")
    workspace_name = os.getenv("AZUREML_WORKSPACE_NAME", "GoodReadsReview-Analysis-60107070")

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    test_dir = Path(args.test_dir)
    yes_dir = test_dir / "yes"
    no_dir = test_dir / "no"

    image_paths = []
    labels = []

    for p in sorted(yes_dir.glob("*")):
        if p.is_file():
            image_paths.append(p)
            labels.append("tumor")
    for p in sorted(no_dir.glob("*")):
        if p.is_file():
            image_paths.append(p)
            labels.append("no_tumor")

    latencies = []
    preds = []

    for img_path, label in zip(image_paths, labels):
        img_b64 = encode_image_base64(img_path)
        body = json.dumps({"image_base64": img_b64})

        t0 = time.perf_counter()
        raw = ml_client.online_endpoints.invoke(
            endpoint_name=args.endpoint_name,
            deployment_name=args.deployment_name,
            request_body=body,
            request_headers={"Content-Type": "application/json"},
        )
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        resp = json.loads(raw)
        pred = resp.get("prediction", "unknown")
        preds.append(pred)

        print(f"{img_path.name}: pred={pred}, true={label}, latency_ms={latency_ms:.2f}")

    latencies = np.array(latencies, dtype=float)
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    correct = sum(1 for p, y in zip(preds, labels) if str(p) == str(y))
    accuracy = correct / len(labels) if labels else 0.0

    print("----- Summary -----")
    print(f"Num samples      : {len(labels)}")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Avg latency (ms) : {avg_latency:.2f}")
    print(f"p95 latency (ms) : {p95_latency:.2f}")


if __name__ == "__main__":
    main()