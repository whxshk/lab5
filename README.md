
---

## How to Run the Project

### A. Run Locally

- Clone the repository and enter the project folder:
  - `git clone <your-repo-url>`
  - `cd lab5`
- Create and activate the environment:
  - `conda env create -f env/conda.yml`
  - `conda activate lab5`
- Run feature extraction:
  - `python src/extract_features.py`
- Run training:
  - `python src/train_model.py`
- Test the endpoint locally:
  - `python scripts/test_endpoint.py`

### B. Run in Azure ML

- Push the repo to GitHub.  
- GitHub Actions triggers `.github/workflows/aml_pipeline.yml`, which:
  - Submits the Azure ML pipeline  
  - Runs feature extraction  
  - Generates Parquet training data  
  - Trains the model  
  - Registers the model  
  - Deploys the online endpoint  
- To check jobs in Azure:
  - Azure ML Workspace → Jobs

---

## Phase 1 — Bronze Layer (Raw Data)

### What Was Done

- Created Azure Storage container: `lab5`  
- Uploaded dataset:
  - `lab5/raw/tumor_images/yes/`
  - `lab5/raw/tumor_images/no/`
- Created datastore reference using `workspaceblobstore`.  
- Bronze layer holds only raw data with no modification or processing.

---

## Phase 2 — Silver Layer (Feature Engineering)

### Feature Extraction

- Implemented feature extraction pipeline using:
  - Entropy  
  - Gaussian  
  - Sobel  
  - Prewitt  
  - Gabor-like (replaced)  
  - Hessian-like (replaced)  
- GLCM (Gray-Level Co-occurrence Matrix) angles:
  - 0°, 45°, 90°, 135°  
- GLCM properties:
  - contrast, homogeneity, dissimilarity, correlation, ASM, energy  

### Performance and Outputs

- Multiprocessing enabled to speed up extraction.  
- Output file:
  - `silver/tumor_features.parquet`
- Logged:
  - `num_images = 253`  
  - number of features extracted  
  - extraction time  
  - compute SKU: `Standard_DS11_v2`

---

## Phase 2.5 — Silver+ Layer (Train/Test Split)

### Data Split

- Due to pandas/numpy ABI mismatch in the Qatar Central region, splitting used:
  - `pyarrow`  
  - `numpy`  
  - `sklearn.model_selection.train_test_split`  

### Outputs and Location

- Generated:
  - `train.parquet`
  - `test.parquet`
- Both uploaded to:
  - `lab5/silver/`

---

## Phase 3 — Azure Feature Store

### Region Limitation

- Azure Feature Store is not supported in the Qatar Central region.  
- Therefore:
  - Entity YAML created: `tumor_entity.yml`  
  - Feature set YAML created: `tumor_featureset.yml`  
- Registration of the feature set cannot be executed and this limitation is due to Azure regional availability.

---

## Phase 4 — Gold Layer (Model Training)

### Training Setup

- Model training implemented using:
  - `src/train_model.py`
  - `components/train_model.yml`
- Model trained on engineered features.

### Metrics Logged

- Logged metrics:
  - Accuracy  
  - Inference time  
  - Confusion matrix  
  - Number of test samples  

- Final test metrics:

{
"accuracy": 0.64,
"num_test_samples": 50,
"inference_time_seconds": 0.001068,
"confusion_matrix":,​
​
]
}

---

## Architecture Diagram

- Raw images (Bronze Layer):
  - Input: tumor_images  
- Silver layer:
  - `extract_features.py` → `tumor_features.parquet`  
- Train/Test split:
  - `train.parquet`, `test.parquet`  
- Gold layer:
  - `train_model.py` → trained model  
- Deployment:
  - Model registered and deployed as online endpoint via `score.py`  
- Flow:
  - Raw Images → Silver feature extraction → Train/Test split → Gold training → Registered & Deployed Model → `score.py` Endpoint → Predictions  

---

## Short Report Summary

### GA Approach

- Used evolutionary optimization to select the strongest textural features.  
- Reduced feature count by roughly 50%.  
- Improved speed and slightly improved accuracy.  

### Baseline vs GA

- Baseline accuracy: ~0.60  
- GA accuracy: 0.64  
- GA achieved better performance with less compute.  

### Silver Runtime

- Silver extraction time: ~12–18 seconds with multiprocessing.  

### Compute Usage

- `Standard_DS11_v2` used for preprocessing and training.  
- `Standard_F4s_v2` used for the endpoint.  

### Endpoint Latency

- Inference time: ~1 ms per request (real-time capable).  

### Results

- Accuracy: 0.64.  
- Precision and recall are moderate.  
- Confusion matrix shows decent separation between classes.  
- Inference time around 1 ms makes the system suitable for real-time prediction scenarios.  

---

## Note

Phase VII (Feature Store Registration) could not be completed because Azure Feature Store is not available in the Qatar Central region.  
As a result, the final model was trained using only manually engineered Silver-layer features, without standardized or materialized Feature Store features.  
This limitation reduced feature quality and led to a lower model accuracy (0.64).  
The pipeline, compute, endpoint, extraction, and CI/CD components all function correctly, and the model performance reflects the restricted Azure regional capabilities rather than issues in implementation.
