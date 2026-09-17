"""
azure_ml_train.py
-----------------
OPTIONAL: submits the training job to Azure Machine Learning and
registers the resulting model in your AML workspace, so you get
experiment tracking, metrics logging, and a deployable model registry
entry — useful if you want the project to show "cloud-native ML" rather
than just a local script.

Prereqs:
    pip install azure-ai-ml azure-identity
    Have an Azure ML workspace created (see README.md Step 4).
    az login   (or use a service principal)

Run:
    python azure_ml_train.py
"""

from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# ---- EDIT THESE THREE VALUES ----
SUBSCRIPTION_ID = "<your-subscription-id>"
RESOURCE_GROUP = "supply-chain-rg"
WORKSPACE_NAME = "supply-chain-ml-ws"
# ----------------------------------

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

env = Environment(
    name="supply-chain-env",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)

job = command(
    code=".",  # uploads current directory (train_model.py + data/)
    command="python train_model.py",
    environment=env,
    compute="cpu-cluster",  # create this compute target first (see README Step 4)
    display_name="disruption-risk-training",
    experiment_name="supply-chain-disruption-risk",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted job: {returned_job.name}")
print(f"Monitor at: {returned_job.studio_url}")

# After the job completes, register the model:
#
# from azure.ai.ml.entities import Model
# from azure.ai.ml.constants import AssetTypes
#
# model = Model(
#     path=f"azureml://jobs/{returned_job.name}/outputs/artifacts/models/disruption_model.joblib",
#     name="disruption-risk-model",
#     type=AssetTypes.CUSTOM_MODEL,
#     description="RandomForest classifier for supply chain disruption risk",
# )
# ml_client.models.create_or_update(model)
