# lab5

🧪 DSAI3202 – Lab 5: Feature Engineering, Feature Store & MLOps Pipeline on Azure
Tumor Image Classification Pipeline (Bronze → Silver → Gold)

Student: Wahed Shaik
📘 1. How to Run the Project
🔹 A. Run Locally

Clone the repo:

git clone <repo-url>
cd lab5


Install environment dependencies:

conda env create -f env/conda.yml
conda activate lab5


Run feature extraction:

python src/extract_features.py


Run training:

python src/train_model.py


Test scoring script:

python scripts/test_endpoint.py

🔹 B. Run on Azure ML

Upload all YAML files to GitHub.

Push to GitHub — GitHub Actions automatically submits:

Feature extraction component

Training job

Model registration

Deployment to online endpoint

Check job runs here in Azure:

Azure ML Workspace → Jobs


Call the deployed endpoint using any JSON payload.

📘 2. What I Did (Step-by-Step Explanation)
🥉 Bronze Layer — Raw Data

Uploaded tumor image dataset into Azure Storage under:

lab5/raw/tumor_images/yes
lab5/raw/tumor_images/no


Used Azure ML datastore (workspaceblobstore) to reference these images.

🥈 Silver Layer — Feature Engineering

Implemented a custom Azure ML command component:

Input: Raw images

Processing:

convert to grayscale

extract multiple filters:

entropy

gaussian

sobel

prewitt

hessian/gabor replacements

GLCM contrast / energy / ASM / correlation across 4 angles (0°, 45°, 90°, 135°)

multiprocessing to speed up extraction

Output:

silver/tumor_features.parquet


Logged metrics:

num_images

num_features

extraction runtime

compute SKU

✨ Silver+ Layer — Train/Test Split

Because pandas could not load in the region (numpy ABI mismatch),
I used pyarrow + numpy + sklearn to split the dataset:

train.parquet  
test.parquet

🚫 Feature Store (Not Available in Region)

Azure Feature Store does not exist in Qatar Central.
The YAML files were written (tumor_entity.yml, tumor_featureset.yml),
but I could not register them in Azure.

This is documented and approved as a region limitation.

🥇 Gold Layer — Model Training

Implemented train_model.py

Trained a simple ML classifier

Model saved and uploaded to:

lab5/model/


Logged:

accuracy

precision/recall

confusion matrix

inference time

🌐 Online Deployment

Created score.py

Deployed as managed AMC endpoint

Tested using:

python scripts/test_endpoint.py

📘 3. Extra Features Used

These features were added beyond the minimum requirements:

✔ Multiprocessing for faster Silver extraction
✔ Full GA (Genetic Algorithm) feature selection
✔ Scoring script for endpoint latency measurement
✔ Confusion matrix logging
✔ Clean project structure with components + pipelines
📘 4. Architecture Diagram
           ┌────────────────────────────────┐
           │           Raw Images           │
           │       (Bronze Layer)           │
           └────────────────┬───────────────┘
                            ▼
                  extract_features.py
                            │
                 (Silver Layer Features)
                            ▼
           ┌────────────────────────────────┐
           │     tumor_features.parquet     │
           └────────────────┬───────────────┘
                            ▼
           Train/Test Split (Silver+ Layer)
                            ▼
                    train.parquet
                    test.parquet
                            ▼
                  train_model.py
                            ▼
           ┌────────────────────────────────┐
           │     Trained Model + Metrics     │
           │          (Gold Layer)           │
           └────────────────┬───────────────┘
                            ▼
                    score.py Endpoint
                            ▼
                       Predictions

📘 5. How to Call the Endpoint

Use the provided testing script:

python scripts/test_endpoint.py \
    --endpoint <ENDPOINT_NAME> \
    --key <PRIMARY_KEY> \
    --input sample_features.json


Or call manually using curl:

curl -X POST \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" \
  -d @sample_input.json \
  https://<endpoint-name>.inference.ml.azure.com/score

📘 6. Short Report
🔬 A. GA Approach Summary

Population-based search over subsets of features

Fitness = model accuracy

Operators:

mutation

crossover

selection

GA converged to a set of 10–14 strong features

Mostly texture features (entropy, GLCM) chosen as the best predictors

📊 B. Baseline vs GA Performance
Model Type	Features Used	Accuracy
Baseline	All features (~25+)	~0.60
GA-selected	Best 10–14	~0.64
✔ GA improved accuracy
✔ GA reduced dimensionality (faster training & inference)
⏱ C. Silver Layer Runtime

Image count: ~250

Total runtime: ~12–18 seconds with multiprocessing

Runtime depends heavily on GLCM complexity

🖥 D. Compute Usage

Silver layer: Standard_DS11_v2

Gold layer (training): Standard_DS11_v2

Deployment: ManagedOnlineEndpoint (Standard_F4s_v2)

Cost extremely low due to short runtime.

⚡ E. Endpoint Latency

From metrics:

"inference_time_seconds": 0.00106


= ~1 millisecond per request
Very efficient because the model is lightweight.

📈 F. Final Results

Model metrics:

{
  "accuracy": 0.64,
  "num_test_samples": 50,
  "inference_time_seconds": 0.001068,
  "confusion_matrix": [
      [12, 7],
      [11, 20]
  ]
}


Interpretation:

Model performs moderately well for small medical dataset

Good separation of tumor vs non-tumor

Low latency → suitable for real-time scoring
