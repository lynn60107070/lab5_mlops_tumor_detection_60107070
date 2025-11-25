import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient

# ---------------- CONFIG ----------------

# Local dataset root
LOCAL_DATASET_DIR = Path(
    r"C:\Users\lynny\Downloads\cloud_lab5\lab5_mlops_tumor_detection_60107070\data\brain_tumor_dataset"
)

# ADLS Gen2 / Blob config
STORAGE_CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
CONTAINER_NAME = "raw"                     # the container you created
REMOTE_ROOT_PREFIX = "tumor_images"        # will create tumor_images/yes and tumor_images/no

# ----------------------------------------


def get_blob_service_client() -> BlobServiceClient:
    conn_str = os.environ.get(STORAGE_CONNECTION_STRING_ENV)
    if not conn_str:
        raise RuntimeError(
            f"Environment variable {STORAGE_CONNECTION_STRING_ENV} is not set"
        )
    return BlobServiceClient.from_connection_string(conn_str)


def upload_folder(container_client, local_folder: Path, remote_subdir: str):
    if not local_folder.is_dir():
        raise RuntimeError(f"Local folder does not exist: {local_folder}")

    # Allow common image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for path in local_folder.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue

        blob_path = f"{REMOTE_ROOT_PREFIX}/{remote_subdir}/{path.name}"
        blob_client = container_client.get_blob_client(blob_path)

        # Idempotency: skip if already uploaded
        if blob_client.exists():
            print(f"SKIP (exists): {blob_path}")
            continue

        with open(path, "rb") as f:
            blob_client.upload_blob(f)
        print(f"UPLOADED: {blob_path}")


def main():
    # Connect to storage
    blob_service_client = get_blob_service_client()

    # Ensure container exists (idempotent)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    try:
        container_client.create_container()
        print(f"Container created: {CONTAINER_NAME}")
    except Exception:
        print(f"Container already exists or cannot be created: {CONTAINER_NAME}")

    # Upload yes/ and no/ folders
    yes_local = LOCAL_DATASET_DIR / "yes"
    no_local = LOCAL_DATASET_DIR / "no"

    upload_folder(container_client, yes_local, "yes")
    upload_folder(container_client, no_local, "no")


if __name__ == "__main__":
    main()
