import argparse
import json
import os
import time
import glob

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# GA feature selection from sklearn-genetic-opt
# Package installed in env as: sklearn-genetic-opt
# Import path is: sklearn_genetic
from sklearn_genetic import GAFeatureSelectionCV


def load_single_parquet_from_folder(path: str) -> pd.DataFrame:
    """Load a single parquet file (or first parquet in folder)."""
    if os.path.isfile(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        raise RuntimeError(f"Unsupported file type: {path}")

    files = glob.glob(os.path.join(path, "*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {path}")
    return pd.read_parquet(files[0])


def baseline_feature_selection(df: pd.DataFrame, label_col: str = "label"):
    """Simple baseline: VarianceThreshold + RandomForest accuracy."""
    id_cols = ["image_id"]
    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]

    X = df[feature_cols].values
    y = df[label_col].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Remove zero-variance features
    vt = VarianceThreshold(threshold=0.0)
    X_train_sel = vt.fit_transform(X_train)
    X_val_sel = vt.transform(X_val)

    selected_mask = vt.get_support()
    selected_features = [f for f, keep in zip(feature_cols, selected_mask) if keep]

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_val_sel)
    acc = accuracy_score(y_val, y_pred)

    metrics = {
        "baseline_accuracy": float(acc),
        "baseline_num_features": int(len(selected_features)),
    }

    return selected_features, metrics


def ga_feature_selection(df: pd.DataFrame, label_col: str = "label"):
    """Genetic Algorithm feature selection using GAFeatureSelectionCV."""
    id_cols = ["image_id"]
    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]

    X = df[feature_cols].values
    y = df[label_col].values

    # Split for GA fitness evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_estimator = RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    # Use only arguments guaranteed by sklearn-genetic-opt
    ga_selector = GAFeatureSelectionCV(
        estimator=base_estimator,
        cv=3,
        scoring="accuracy",
        population_size=30,   # >= 20
        generations=15,       # >= 10
        crossover_probability=0.5,
        mutation_probability=0.2,
        n_jobs=-1,
        verbose=True,
    )

    t0 = time.time()
    ga_selector.fit(X_train, y_train)
    runtime = time.time() - t0

    # support_ is a boolean mask over features
    mask = ga_selector.support_
    selected_features = [f for f, keep in zip(feature_cols, mask) if keep]

    # Evaluate on hold-out validation using selected features
    X_train_sel = X_train[:, mask]
    X_val_sel = X_val[:, mask]

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_val_sel)
    acc = accuracy_score(y_val, y_pred)

    metrics = {
        "ga_accuracy": float(acc),
        "ga_num_features": int(len(selected_features)),
        "ga_runtime_seconds": float(runtime),
    }

    return selected_features, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, required=True)
    parser.add_argument("--selected_features_output", type=str, required=True)
    parser.add_argument("--baseline_metrics_output", type=str, required=True)
    parser.add_argument("--ga_metrics_output", type=str, required=True)
    args = parser.parse_args()

    df_train = load_single_parquet_from_folder(args.train_input)

    # Baseline
    baseline_selected, baseline_metrics = baseline_feature_selection(df_train)

    # GA
    ga_selected, ga_metrics = ga_feature_selection(df_train)

    # Persist selected feature names (GA is the final set)
    os.makedirs(os.path.dirname(args.selected_features_output), exist_ok=True)
    with open(args.selected_features_output, "w") as f:
        json.dump(
            {
                "selected_features": ga_selected,
                "baseline_selected_features": baseline_selected,
            },
            f,
            indent=2,
        )

    # Metrics
    os.makedirs(os.path.dirname(args.baseline_metrics_output), exist_ok=True)
    with open(args.baseline_metrics_output, "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    os.makedirs(os.path.dirname(args.ga_metrics_output), exist_ok=True)
    with open(args.ga_metrics_output, "w") as f:
        json.dump(ga_metrics, f, indent=2)

    print("Baseline metrics:", baseline_metrics)
    print("GA metrics:", ga_metrics)
    print("GA selected features:", len(ga_selected))


if __name__ == "__main__":
    main()
