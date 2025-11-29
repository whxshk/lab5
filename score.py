import json
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, prewitt, gaussian
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import graycomatrix, graycoprops
from io import BytesIO
import base64

# Global variables loaded in init()
MODEL = None
FEATURE_NAMES = None

def extract_glcm_features(gray):
    glcm = graycomatrix(
        (gray * 255).astype("uint8"),
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True,
        normed=True,
    )
    features = {}
    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    angles = [0, 45, 90, 135]
    for p in props:
        vals = graycoprops(glcm, p)[0]
        for i, v in enumerate(vals):
            features[f"{p}_{angles[i]}"] = float(v)
    return features

def extract_all_features(img):
    """Return a dict of all features for ONE image."""
    gray = rgb2gray(img)
    feats = {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "sobel": float(sobel(gray).mean()),
        "prewitt": float(prewitt(gray).mean()),
        "gaussian": float(gaussian(gray, sigma=1).mean()),
        "entropy": float(entropy((gray * 255).astype("uint8"), disk(5)).mean()),
    }
    feats.update(extract_glcm_features(gray))
    return feats

def init():
    """Azure ML calls this once when the container starts."""
    global MODEL, FEATURE_NAMES

    # We expect these files to be packaged with the model:
    # - knn_model.npz       (contains X_train, y_train, feature_names)
    # - selected_features.json (optional, but we can just use feature_names from npz)

    # They will typically be placed in ./ (root of model folder)
    import os

    model_path = os.path.join(os.getcwd(), "knn_model.npz")
    print("Loading model from:", model_path)
    data = np.load(model_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    FEATURE_NAMES = list(data["feature_names"])

    MODEL = {
        "X_train": X_train,
        "y_train": y_train,
    }

    print("Model loaded. Train shape:", X_train.shape)
    print("Features:", FEATURE_NAMES)

def predict_knn_1(x_vec, X_train, y_train):
    # 1-NN: find closest row in X_train
    dists = np.linalg.norm(X_train - x_vec, axis=1)
    idx = int(np.argmin(dists))
    return int(y_train[idx])

def run(raw_data):
    """
    Azure ML calls this per request.
    We expect JSON like:
      {
        "image_base64": "<base64-encoded-image>"
      }
    """
    try:
        data = json.loads(raw_data)
        b64 = data["image_base64"]

        # Decode base64 to bytes
        img_bytes = base64.b64decode(b64)
        img = imread(BytesIO(img_bytes))

        # Extract all features we know how to compute
        all_feats = extract_all_features(img)

        # Build feature vector in the same order as FEATURE_NAMES
        x_vec = np.array([all_feats[name] for name in FEATURE_NAMES], dtype=float).reshape(1, -1)

        # Predict
        y_pred = predict_knn_1(x_vec, MODEL["X_train"], MODEL["y_train"])

        # 0 = no tumor, 1 = tumor
        result = {
            "prediction": int(y_pred),
            "label": "tumor" if y_pred == 1 else "no_tumor"
        }

        return json.dumps(result)

    except Exception as e:
        # Return error message for debugging
        return json.dumps({"error": str(e)})
