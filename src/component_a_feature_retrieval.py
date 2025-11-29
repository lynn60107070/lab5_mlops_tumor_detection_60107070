import argparse
import os
import glob
import time

import pandas as pd
from sklearn.model_selection import train_test_split


def load_parquet_folder_or_file(path: str) -> pd.DataFrame:
    # If it's a single file
    if os.path.isfile(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        raise RuntimeError(f"Unsupported file type: {path}")

    # If it's a folder of parquet files
    files = glob.glob(os.path.join(path, "*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {path}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, axis=0, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_input", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--test_output", type=str, required=True)
    args = parser.parse_args()

    t0 = time.time()

    # Silver layer Parquet already contains: image_id, label, f1..fN
    df = load_parquet_folder_or_file(args.features_input)

    if "label" not in df.columns:
        raise RuntimeError("Expected column 'label' in Silver features parquet.")

    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    train_path = os.path.join(args.train_output, "train.parquet")
    test_path = os.path.join(args.test_output, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"num_rows_total={len(df)}")
    print(f"num_rows_train={len(train_df)}")
    print(f"num_rows_test={len(test_df)}")
    print(f"split_time_seconds={time.time() - t0}")


if __name__ == "__main__":
    main()
