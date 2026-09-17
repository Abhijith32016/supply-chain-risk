"""
azure_blob_utils.py
-----------------
Uploads local project files (raw data, cleaned data, model, dashboard
outputs) to Azure Blob Storage, and can download them back.

Prereqs:
    pip install azure-storage-blob
    Set environment variable AZURE_STORAGE_CONNECTION_STRING
    (Azure Portal -> Storage account -> Access keys -> Connection string)

Run:
    python azure_blob_utils.py upload
    python azure_blob_utils.py download
"""

import os
import sys
from azure.storage.blob import BlobServiceClient

CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "supply-chain-data"

FILES_TO_UPLOAD = {
    "raw/supply_chain_raw.csv": "data/supply_chain_raw.csv",
    "processed/supply_chain_clean.csv": "data/supply_chain_clean.csv",
    "models/disruption_model.joblib": "models/disruption_model.joblib",
    "reports/eda_summary.txt": "outputs/eda_summary.txt",
    "reports/model_metrics.txt": "outputs/model_metrics.txt",
}


def get_container_client():
    if not CONNECTION_STRING:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. "
            "Run: export AZURE_STORAGE_CONNECTION_STRING='<your-connection-string>'"
        )
    service_client = BlobServiceClient.from_connection_string(
        CONNECTION_STRING, connection_timeout=120, read_timeout=120
    )
    container_client = service_client.get_container_client(CONTAINER_NAME)
    if not container_client.exists():
        container_client = service_client.create_container(CONTAINER_NAME)
        print(f"Created container '{CONTAINER_NAME}'")
    return container_client


def upload_all():
    container_client = get_container_client()
    for blob_path, local_path in FILES_TO_UPLOAD.items():
        if not os.path.exists(local_path):
            print(f"SKIP (not found): {local_path}")
            continue
        with open(local_path, "rb") as f:
            container_client.upload_blob(name=blob_path, data=f, overwrite=True)
        print(f"Uploaded {local_path} -> {CONTAINER_NAME}/{blob_path}")


def download_all(dest_dir="downloaded"):
    container_client = get_container_client()
    os.makedirs(dest_dir, exist_ok=True)
    for blob in container_client.list_blobs():
        target = os.path.join(dest_dir, blob.name.replace("/", "_"))
        os.makedirs(os.path.dirname(target), exist_ok=True) if os.path.dirname(target) else None
        with open(target, "wb") as f:
            f.write(container_client.download_blob(blob.name).readall())
        print(f"Downloaded {blob.name} -> {target}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "upload"
    if action == "upload":
        upload_all()
    elif action == "download":
        download_all()
    else:
        print("Usage: python azure_blob_utils.py [upload|download]")
