# Cloud-Based Supply Chain Disruption Risk Analytics

An end-to-end project that ingests supply chain data, cleans and analyzes it,
trains a machine learning model to predict disruption risk, and surfaces the
results in an interactive dashboard — backed by Azure cloud services.

## Project structure

```
supply-chain-risk/
├── generate_data.py       # creates a synthetic dataset (skip if you have real data)
├── data_cleaning.py       # cleaning + feature engineering
├── eda_analysis.py        # exploratory/statistical analysis + charts
├── train_model.py         # trains RandomForest disruption classifier
├── azure_blob_utils.py    # upload/download data & model to Azure Blob Storage
├── azure_ml_train.py      # (optional) run training as an Azure ML job
├── dashboard.py           # Streamlit interactive dashboard
├── requirements.txt
├── environment.yml        # conda env used by the Azure ML job
├── data/                  # raw + cleaned CSVs land here
├── models/                # trained model (.joblib) lands here
└── outputs/                # charts, metrics, EDA summary
```

---

## Part A — Run everything locally first (get it working before touching Azure)

### 1. Set up environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get data
Either generate synthetic data:
```bash
python generate_data.py
```
or drop a real dataset (e.g. Kaggle "DataCo Smart Supply Chain" or
"Supply Chain Shipment Pricing Data") into `data/supply_chain_raw.csv` —
just make sure the column names in `data_cleaning.py` line up, or edit
that script to match your columns.

### 3. Clean and engineer features
```bash
python data_cleaning.py
```

### 4. Run exploratory/statistical analysis
```bash
python eda_analysis.py
```
Produces charts in `outputs/`: correlation heatmap, disruption rate by
transport mode, delay distribution, monthly trend — plus a text summary.

### 5. Train the ML model
```bash
python train_model.py
```
Trains a RandomForestClassifier to predict `disruption_flag`, and saves
metrics (precision/recall/ROC-AUC), a confusion matrix, feature
importance chart, and the model file.

### 6. Launch the dashboard
```bash
streamlit run dashboard.py
```
Opens a browser dashboard with KPIs, risk-by-region/mode charts, a
high-risk supplier table, and a live "predict risk for a new order" tool.

Get this whole loop working locally before moving to Azure — it's much
faster to debug data/model issues on your laptop.

---

## Part B — Move it to Azure (using your student credits)

### Step 1: Install Azure CLI and log in
```bash
az login
az account show   # confirm you're on the right subscription
```

### Step 2: Create a resource group
Keeping everything in one resource group makes it easy to find and to
delete later so you don't burn credits.
```bash
az group create --name supply-chain-rg --location eastus
```

### Step 3: Create a Storage Account + Blob container (cloud storage layer)
```bash
az storage account create \
  --name supplychainstorage$RANDOM \
  --resource-group supply-chain-rg \
  --location eastus \
  --sku Standard_LRS

# Save the connection string
az storage account show-connection-string \
  --name <your-storage-account-name> \
  --resource-group supply-chain-rg \
  --output tsv
```
Set it as an environment variable:
```bash
export AZURE_STORAGE_CONNECTION_STRING="<paste the connection string here>"
```
Then upload your project's data, model, and reports:
```bash
python azure_blob_utils.py upload
```
This creates a `supply-chain-data` container with `raw/`, `processed/`,
`models/`, and `reports/` folders — this is your cloud data lake.

### Step 4 (optional but recommended for the "scalable processing" part):
Create an Azure Machine Learning workspace so training runs in the cloud
with experiment tracking:
```bash
az extension add -n ml
az ml workspace create --name supply-chain-ml-ws --resource-group supply-chain-rg

# Create a small compute cluster (auto-scales to 0 nodes when idle, so it's cheap)
az ml compute create --name cpu-cluster --type AmlCompute \
  --min-instances 0 --max-instances 2 --size Standard_DS2_v2 \
  --resource-group supply-chain-rg --workspace-name supply-chain-ml-ws
```
Edit the top of `azure_ml_train.py` with your subscription ID, resource
group, and workspace name, then:
```bash
python azure_ml_train.py
```
This submits `train_model.py` as a cloud job and gives you a Studio URL
to watch it run, view logs, and register the resulting model.

### Step 5 (optional): Automate ingestion with Azure Data Factory
For the "cloud-based data analytics architecture" collecting/orchestrating
piece, set up a simple pipeline in the Azure Portal:
1. Create an Azure Data Factory resource (`az datafactory create` or via
   Portal: Create a resource → Data Factory).
2. In the ADF Studio, create a pipeline with a **Copy Data** activity:
   source = your original data location (on-prem file, API, or another
   storage account), sink = the `raw/` folder in your Blob container.
3. Add a **Trigger** (schedule, e.g. daily) so new supply chain data
   lands in Blob Storage automatically — this is the piece that makes
   the architecture "cloud-based" rather than a one-off script.
4. Optionally chain a **Databricks Notebook** or **Azure Function**
   activity after Copy Data to run `data_cleaning.py` automatically on
   every new file drop.

### Step 6: Deploy the dashboard so it's actually "in the cloud"
Simplest option — Azure App Service:
```bash
az appservice plan create --name supply-chain-plan --resource-group supply-chain-rg --sku B1 --is-linux
az webapp create --name supply-chain-dashboard --resource-group supply-chain-rg \
  --plan supply-chain-plan --runtime "PYTHON:3.10"

# Deploy your code (from the project folder)
az webapp up --name supply-chain-dashboard --resource-group supply-chain-rg
```
Then in the Portal, set the **Startup Command** for the Web App to:
```
python -m streamlit run dashboard.py --server.port 8000 --server.address 0.0.0.0
```
Your dashboard will be live at `https://supply-chain-dashboard.azurewebsites.net`.

Alternative: if you'd rather present in Power BI (common for this kind
of academic project), skip Streamlit and instead:
1. In Power BI Desktop, use **Get Data → Azure Blob Storage**, point it
   at your `processed/supply_chain_clean.csv`.
2. Build visuals: KPI cards (disruption rate), bar charts (by region/
   mode), a map (if you add lat/long), and a table of high-risk
   suppliers. Power BI is free with your student account.

---

## Part C — Wrapping up for submission

- Take screenshots of: the EDA charts, the model metrics, and the live
  dashboard — these go straight into your report/slides.
- Note your architecture: **Storage** (Blob) → **Processing** (Python
  scripts or Azure ML) → **Analytics** (pandas/sklearn) →
  **Visualization** (Streamlit or Power BI).
- **Cost control**: stop/delete resources you're not actively using.
  When you're fully done, delete everything in one shot:
  ```bash
  az group delete --name supply-chain-rg --yes --no-wait
  ```

## Extending this project further
- Swap RandomForest for XGBoost or a time-series model (Prophet) to
  forecast disruption risk trends rather than just classify individual
  orders.
- Add real-time ingestion with **Azure Event Hubs** if you want to
  simulate live IoT/tracking data instead of static CSVs.
- Add **Azure Cognitive Services (Text Analytics)** to mine supplier
  news or shipment notes for sentiment as an extra risk signal.
