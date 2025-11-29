import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, prewitt, gaussian
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import graycomatrix, graycoprops


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


def process_image(path):
    img = imread(path)
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


def main(input_dir, output_file):
    rows = []
    input_dir = Path(input_dir)

    for label in ["yes", "no"]:
        folder = input_dir / label
        for img_name in os.listdir(folder):
            img_path = folder / img_name
            if not img_path.is_file():
                continue

            features = process_image(img_path)
            features["image_id"] = img_name
            features["label"] = 1 if label == "yes" else 0
            rows.append(features)

    df = pd.DataFrame(rows)
    df.to_parquet(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_images")
    parser.add_argument("--output_features")
    args = parser.parse_args()

    main(args.input_images, args.output_features)
