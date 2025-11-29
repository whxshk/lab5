# src/ingest_images.py
import argparse
import os
import pyarrow as pa
import pyarrow.parquet as pq

def main(input_dir, output_manifest):
    yes_dir = os.path.join(input_dir, "yes")
    no_dir = os.path.join(input_dir, "no")

    if not os.path.isdir(yes_dir) or not os.path.isdir(no_dir):
        raise ValueError("Input directory must contain 'yes/' and 'no/' subfolders")

    rows = []
    for label_dir, label_value in [(yes_dir, 1), (no_dir, 0)]:
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            if os.path.isfile(fpath):
                rows.append({
                    "image_id": fname,
                    "label": label_value,
                    "path": fpath
                })

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_manifest)

    print("Ingest completed.")
    print(f"Total images: {len(rows)}")
    print(f"Manifest saved to {output_manifest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_manifest)
