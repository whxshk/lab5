
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
