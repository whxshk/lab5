# src/train_model.py
import argparse
import json
import time
import numpy as np
import pyarrow.parquet as pq

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_logreg(X, y, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        preds = sigmoid(z)
        grad_w = (1 / n_samples) * X.T @ (preds - y)
        grad_b = (1 / n_samples) * np.sum(preds - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def accuracy(X, y, w, b):
    preds = sigmoid(X @ w + b) >= 0.5
    return float((preds == y).mean())

def main(train_path, test_path, model_out, metrics_out):
    start = time.time()
    train = pq.read_table(train_path)
    test = pq.read_table(test_path)

    cols = train.column_names
    feature_cols = [c for c in cols if c not in ("label", "image_id")]

    X_train = np.column_stack([train[c].to_numpy() for c in feature_cols]).astype(float)
    y_train = train["label"].to_numpy().astype(float)

    X_test = np.column_stack([test[c].to_numpy() for c in feature_cols]).astype(float)
    y_test = test["label"].to_numpy().astype(float)

    w, b = train_logreg(X_train, y_train, lr=0.01, epochs=800)

    train_acc = accuracy(X_train, y_train, w, b)
    test_acc = accuracy(X_test, y_test, w, b)

    elapsed = time.time() - start

    np.savez(model_out, w=w, b=b, feature_cols=np.array(feature_cols))

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "num_features": len(feature_cols),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_time_seconds": elapsed,
    }
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", model_out)
    print("Metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model_out", default="model.npz")
    parser.add_argument("--metrics_out", default="metrics.json")
    args = parser.parse_args()
    main(args.train, args.test, args.model_out, args.metrics_out)
