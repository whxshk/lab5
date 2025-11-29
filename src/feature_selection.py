# src/feature_selection.py
import argparse
import json
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import random

def evaluate_subset(X, y, mask):
    """Simple logistic-model-like score: correlation with label."""
    if mask.sum() == 0:
        return 0.0
    subset = X[:, mask]
    scores = np.abs(np.corrcoef(subset.T, y)[-1][:-1])
    return float(np.nanmean(scores))

def genetic_algorithm(X, y, pop_size=20, generations=20, mutation_rate=0.1):
    n_features = X.shape[1]
    population = [np.random.randint(0, 2, n_features).astype(bool) for _ in range(pop_size)]

    for _ in range(generations):
        scored = [(evaluate_subset(X, y, m), m) for m in population]
        scored.sort(reverse=True)
        population = [m for _, m in scored[:pop_size//2]]

        while len(population) < pop_size:
            p1, p2 = random.sample(population[:5], 2)
            crossover = np.random.rand(n_features) < 0.5
            child = np.where(crossover, p1, p2).copy()

            if random.random() < mutation_rate:
                idx = random.randint(0, n_features - 1)
                child[idx] = not child[idx]

            population.append(child)

    best_score, best_mask = max([(evaluate_subset(X, y, m), m) for m in population], key=lambda x: x[0])
    return best_mask, best_score

def main(train_parquet, test_parquet, output_dir):
    train = pq.read_table(train_parquet)
    test = pq.read_table(test_parquet)

    feature_cols = [c for c in train.column_names if c not in ("label", "image_id")]

    X_train = np.column_stack([train[c].to_numpy() for c in feature_cols]).astype(float)
    y_train = train["label"].to_numpy().astype(float)

    # Run GA
    best_mask, score = genetic_algorithm(X_train, y_train)

    selected = [f for f, keep in zip(feature_cols, best_mask) if keep]

    with open(f"{output_dir}/selected_features.json", "w") as f:
        json.dump({"selected_features": selected, "score": score}, f, indent=2)

    # Save reduced datasets
    train_sel = {c: pa.array(train[c].to_numpy()) for c in selected}
    train_sel["label"] = train["label"]
    train_sel["image_id"] = train["image_id"]
    pq.write_table(pa.table(train_sel), f"{output_dir}/train_selected.parquet")

    test_sel = {c: pa.array(test[c].to_numpy()) for c in selected}
    test_sel["label"] = test["label"]
    test_sel["image_id"] = test["image_id"]
    pq.write_table(pa.table(test_sel), f"{output_dir}/test_selected.parquet")

    print("Selected features:", selected)
    print("GA score:", score)
    print("Saved selected train/test parquet files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--test_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.train_parquet, args.test_parquet, args.output_dir)
