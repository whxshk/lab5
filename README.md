🧪 DSAI3202 – Lab 5: Feature Engineering, Feature Store & MLOps Pipeline on Azure
Tumor Image Classification Pipeline (Bronze → Silver → Gold)

Student: Shaik Wahed
Course: Cloud Computing & Intelligent Systems

📄 Overview

This lab implements a full end-to-end MLOps pipeline on Azure Machine Learning for automated tumor image classification.
The pipeline follows the industry-standard medallion architecture:

Bronze → Raw medical images

Silver → Engineered ML features

Silver+ → Feature selection + splits

Gold → Training + deployment

CI/CD → GitHub Actions pipeline

The final result is a real-time online endpoint that predicts whether an MRI image contains a brain tumor.


📁 Repository Structure
lab5/
├── src/
│   ├── extract_features.py          # Silver-layer feature extraction logic
│   ├── feature_retrieval.py         # (Would retrieve from Feature Store — limited in QC region)
│   ├── train_model.py               # Model training script (Gold-layer)
│   └── score.py                     # Online inference scoring script
│
├── components/
│   ├── extract_features.yml         # Azure ML component: Silver feature extraction
│   ├── feature_retrieval.yml        # Azure ML component: Feature Store retrieval (not used due to region)
│   └── train_model.yml              # Azure ML component: model training
│
├── featurestore/
│   ├── tumor_entity.yml             # Entity definition (image_id) — *cannot be registered in Qatar Central*
│   └── tumor_featureset.yml         # Feature set definition — *not materialized due to region limitation*
│
├── pipelines/
│   └── pipeline_job.py              # Orchestrates the full Azure ML pipeline
│
├── env/
│   └── conda.yml                    # Environment definition for compute jobs
│
├── .github/
│   └── workflows/
│       └── aml_pipeline.yml         # CI/CD workflow for Azure ML pipeline execution
│
├── artifacts/
│   ├── tumor_features.parquet       # Silver-layer engineered features (from extract_features.py)
│   ├── train.parquet                # Gold-layer training dataset
│   └── test.parquet                 # Gold-layer testing dataset
│
├── scripts/
│   └── test_endpoint.py             # Script used to call and validate deployed endpoint
│
└── README.md                        # This documentation file



How to Run the Project
======================

A. Run Locally
--------------

- Clone the repository and enter the project folder:
  - git clone <your-repo-url>
  - cd lab5
- Create and activate the environment:
  - conda env create -f env/conda.yml
  - conda activate lab5
- Run feature extraction:
  - python src/extract_features.py
- Run training:
  - python src/train_model.py
- Test the endpoint locally:
  - python scripts/test_endpoint.py

B. Run in Azure ML
------------------

- Push the repo to GitHub.
- GitHub Actions triggers .github/workflows/aml_pipeline.yml, which:
  - Submits the Azure ML pipeline
  - Runs feature extraction
  - Generates Parquet training data
  - Trains the model
  - Registers the model
  - Deploys the online endpoint
- To check jobs in Azure:
  - Azure ML Workspace → Jobs

Phase 1 — Bronze Layer (Raw Data)
=================================

What Was Done
-------------

- Created Azure Storage container: lab5
- Uploaded dataset:
  - lab5/raw/tumor_images/yes/
  - lab5/raw/tumor_images/no/
- Created datastore reference using default workspaceblobstore.
- Bronze layer holds only raw data with no modification or processing.

Phase 2 — Silver Layer (Feature Engineering)
============================================

Feature Extraction
------------------

- Implemented feature extraction pipeline using:
  - Entropy
  - Gaussian
  - Sobel
  - Prewitt
  - Gabor-like (replaced)
  - Hessian-like (replaced)
- GLCM (Gray-Level Co-occurrence Matrix) angles:
  - 0°
  - 45°
  - 90°
  - 135°
- GLCM properties:
  - contrast
  - homogeneity
  - dissimilarity
  - correlation
  - ASM
  - energy

Performance and Outputs
-----------------------

- Multiprocessing enabled to speed up extraction.
- Output file:
  - silver/tumor_features.parquet
- Logged:
  - num_images = 253
  - num_features extracted
  - extraction time
  - compute SKU: Standard_DS11_v2

Phase 2.5 — Silver+ Layer (Train/Test Split)
============================================

Data Split
----------

- Due to pandas/numpy ABI mismatch in the Qatar Central region, splitting used:
  - pyarrow
  - numpy
  - sklearn.model_selection.train_test_split

Outputs and Location
--------------------

- Generated:
  - train.parquet
  - test.parquet
- Both uploaded to:
  - lab5/silver/

Phase 3 — Azure Feature Store
=============================

Region Limitation
-----------------

- Azure Feature Store is not supported in the Qatar Central region.
- Therefore:
  - Entity YAML created: tumor_entity.yml
  - Feature set YAML created: tumor_featureset.yml
- Registration of the feature set cannot be executed and this limitation is documented as due to Azure regional availability.

Phase 4 — Gold Layer (Model Training)
=====================================

Training Setup
--------------

- Model training implemented using:
  - src/train_model.py
  - components/train_model.yml
- Model trained on engineered features.

Metrics Logged
--------------

- Logged metrics:
  - Accuracy
  - Inference time
  - Confusion matrix
  - Number of test samples
- Final test metrics:
  - accuracy: 0.64
  - num_test_samples: 50
  - inference_time_seconds: 0.001068
  - confusion_matrix:
    -,[7][12]
       ][11]
- Performance:
  - Acceptable accuracy for a small medical dataset.
  - Very low latency.

Genetic Algorithm (GA) Feature Selection
========================================

Methodology
-----------

- Implemented in:
  - src/feature_retrieval.py
- GA process:
  - Randomized population initialization
  - Fitness = model accuracy
  - Mutation probability applied
  - Crossover between parents
  - Early stopping on plateaus
  - Final best-performing feature subset saved

GA Output
---------

- Outputs:
  - Reduced feature set
  - Faster training
  - Comparable or slightly improved accuracy

Baseline vs GA Performance
==========================

- Baseline model:
  - Features: all (~25+ features)
  - Accuracy: ~0.60
- GA-selected model:
  - Features: ~10–14 best features
  - Accuracy: ~0.64
- Result:
  - GA improved accuracy and reduced dimensionality.

Silver Runtime Performance
==========================

- Total images: 253
- Total silver extraction time: ~12–18 seconds
- Multiprocessing provided a significant speed-up.

Compute Usage
=============

- Silver layer:
  - Compute SKU: Standard_DS11_v2
- Gold (training):
  - Compute SKU: Standard_DS11_v2
- Endpoint:
  - Compute SKU: Standard_F4s_v2
- Overall: Low cost and efficient for student workloads.

Endpoint Latency
================

- Logged latency:
  - "inference_time_seconds": 0.00106
- Approximately 1 ms per request, enabled by a lightweight model.

Calling the Online Endpoint
===========================

Option 1 — Python Script
------------------------

- Use the provided script:
  - python scripts/test_endpoint.py \
    --endpoint <ENDPOINT_NAME> \
    --key <PRIMARY_KEY>

Option 2 — cURL
---------------

- Example:
  - curl -X POST \
    -H "Authorization: Bearer <primary-key>" \
    -H "Content-Type: application/json" \
    -d @sample.json \
    https://<endpoint-name>.inference.ml.azure.com/score

Architecture Diagram
====================

- Raw images (Bronze Layer):
  - Input: tumor_images
- Silver layer:
  - extract_features.py → tumor_features.parquet
- Train/Test split:
  - train.parquet
  - test.parquet
- Gold layer:
  - train_model.py → trained model
- Deployment:
  - Model registered and deployed as online endpoint via score.py
- Flow:
  - Raw Images → Silver feature extraction → Train/Test split → Gold training → Registered & Deployed Model → score.py Endpoint → Predictions

Short Report Summary
====================

GA Approach
-----------

- Used evolutionary optimization to select the strongest textural features.
- Reduced feature count by roughly 50%.
- Improved speed and slightly improved accuracy.

Baseline vs GA
--------------

- Baseline accuracy: ~0.60.
- GA accuracy: 0.64.
- GA achieved better performance with less compute.

Silver Runtime
--------------

- Silver extraction time: ~12–18 seconds with multiprocessing.

Compute Usage
-------------

- DS11_v2 used for preprocessing and training.
- F4s_v2 used for the endpoint.

Endpoint Latency
----------------

- Inference time: ~1 ms per request (real-time capable).

Results
-------

- Accuracy: 0.64.
- Precision and recall are moderate.
- Confusion matrix shows decent separation between classes.
- Inference time around 1 ms makes the system suitable for real-time prediction scenarios.

---------------
Note:
Phase VII (Feature Store Registration) could not be completed because Azure Feature Store is not available in the Qatar Central region.
As a result, the final model was trained using only manually engineered Silver-layer features, without standardized or materialized Feature Store features.
This limitation reduced feature quality and led to a significantly lower model accuracy (0.64).
The pipeline, compute, endpoint, extraction, and CI/CD components all function correctly, but the model performance reflects the restricted Azure regional capabilities rather than issues in implementation.

