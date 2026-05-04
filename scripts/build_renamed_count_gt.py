import argparse
import csv
import os


def read_mapping(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"old_name", "new_name"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("rename_mapping.csv must contain old_name,new_name")
        return {(row["old_name"] or "").strip(): (row["new_name"] or "").strip() for row in reader}


def read_counts(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "coins_count"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("coins_count_values.csv must contain image_name,coins_count")
        rows = []
        for row in reader:
            old_name = (row.get("image_name") or "").strip()
            count = int(row.get("coins_count") or 0)
            folder = (row.get("folder") or "").strip()
            rows.append((folder, old_name, count))
        return rows


def main():
    parser = argparse.ArgumentParser(description="Build single count GT CSV keyed by renamed image filenames.")
    parser.add_argument("--mapping", default="data/rename_mapping.csv")
    parser.add_argument("--counts", default="data/coins_count_values.csv")
    parser.add_argument("--out", default="data/coins_count_values_renamed.csv")
    args = parser.parse_args()

    mapping = read_mapping(args.mapping)
    counts = read_counts(args.counts)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    missing = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "coins_count", "original_image_name", "folder"])
        for folder, old_name, count in counts:
            new_name = mapping.get(old_name)
            if not new_name:
                missing += 1
                continue
            writer.writerow([new_name, count, old_name, folder])

    print(f"Wrote combined GT: {args.out}")
    print(f"Rows written: {len(counts) - missing}")
    print(f"Rows skipped (missing mapping): {missing}")


if __name__ == "__main__":
    main()
