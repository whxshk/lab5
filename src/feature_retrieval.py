# src/feature_retrieval.py
import argparse
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from sklearn.model_selection import train_test_split  # <-- if this import dies in THIS env,
                                                      # you can remove it here and just keep for GitHub

def main(input_parquet: str, output_dir: str):
    table = pq.read_table(input_parquet)
    cols = table.column_names
    feature_cols = [c for c in cols if c not in ("label", "image_id")]

    X = np.column_stack([table[col].to_numpy() for col in feature_cols])
    y = table["label"].to_numpy()
    image_ids = table["image_id"].to_numpy()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, image_ids, test_size=0.2, stratify=y, random_state=42
    )

    train_data = {col: pa.array(X_train[:, i]) for i, col in enumerate(feature_cols)}
    train_data["label"] = pa.array(y_train)
    train_data["image_id"] = pa.array(id_train)
    pq.write_table(pa.table(train_data), f"{output_dir}/train.parquet")

    test_data = {col: pa.array(X_test[:, i]) for i, col in enumerate(feature_cols)}
    test_data["label"] = pa.array(y_test)
    test_data["image_id"] = pa.array(id_test)
    pq.write_table(pa.table(test_data), f"{output_dir}/test.parquet")

    print("Saved train/test to", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.input_parquet, args.output_dir)
