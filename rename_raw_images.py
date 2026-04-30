"""
Rename the random-hash images in data/raw/ to coin_001.jpg, coin_002.jpg, ...
Order is the current alphabetical order of the filenames.

Usage
-----
    python rename_raw_images.py            # DRY RUN — prints what would happen
    python rename_raw_images.py --apply    # actually rename the files
    python rename_raw_images.py --apply --folder data/raw  # custom folder

A mapping CSV (old -> new) is written next to the folder so you can revert.
"""

import os
import sys
import argparse
import csv

EXTS = (".jpg", ".jpeg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/raw")
    ap.add_argument("--prefix", default="coin")
    ap.add_argument("--apply", action="store_true",
                    help="Actually perform the renames (default is dry-run)")
    args = ap.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        sys.exit(f"Folder not found: {folder}")

    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(EXTS))
    if not files:
        sys.exit(f"No images in {folder}")

    width = max(3, len(str(len(files))))   # at least 3 digits, more if needed
    plan = []
    for i, old in enumerate(files, start=1):
        ext = os.path.splitext(old)[1].lower()
        new = f"{args.prefix}_{i:0{width}d}{ext}"
        plan.append((old, new))

    print(f"{'DRY RUN' if not args.apply else 'APPLYING'}: "
          f"{len(plan)} file(s) in {folder}\n")
    for old, new in plan[:5]:
        print(f"  {old}  ->  {new}")
    if len(plan) > 10:
        print(f"  ... ({len(plan) - 10} more) ...")
    for old, new in plan[-5:]:
        print(f"  {old}  ->  {new}")

    if not args.apply:
        print("\nRun with --apply to perform the renames.")
        return

    # Two-pass rename via temporary names to avoid collisions
    temp_pairs = []
    for old, new in plan:
        old_path = os.path.join(folder, old)
        tmp_path = os.path.join(folder, f"__tmp_rename__{new}")
        os.rename(old_path, tmp_path)
        temp_pairs.append((tmp_path, os.path.join(folder, new), old, new))

    mapping_csv = os.path.join(os.path.dirname(folder.rstrip("/\\")) or ".",
                               "rename_mapping.csv")
    with open(mapping_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["old_name", "new_name"])
        for tmp, final, old, new in temp_pairs:
            os.rename(tmp, final)
            w.writerow([old, new])

    print(f"\nRenamed {len(plan)} files.")
    print(f"Mapping (old -> new) saved to: {mapping_csv}")


if __name__ == "__main__":
    main()
