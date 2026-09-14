"""
Dataset splitter - stratified image-level split with duplicate detection.

WHY THIS VERSION CHANGED (again)
----------------------------------
The previous version of this script tried PATIENT-LEVEL splitting by
inferring a patient ID from filenames (e.g. "P045_T1_02.jpg" -> "P045").

After running `--inspect` on the real dataset, two things became clear:

1. Most classes showed ZERO grouping - every image got its own unique ID.
   This means the filenames genuinely do not encode patient identity for
   most of this dataset.

2. Where grouping DID happen, some of it was WRONG, not just missing.
   Example from the real --inspect output:
       patient_id='4': ['4.jpeg', '456.jpeg']  (2 files)
   "4.jpeg" and "456.jpeg" are almost certainly two unrelated images that
   happen to share a leading digit. The regex was falsely merging unrelated
   files into a fake "patient" - which is actively worse than doing nothing,
   because it would silently force unrelated images into the same split and
   give false confidence that leakage protection was happening.

CONCLUSION: this dataset is a collected/aggregated dataset (per the project
owner) with no patient metadata anywhere. Patient-level splitting is not
recoverable from filenames alone. Trying to fake it via regex guessing does
more harm than good. This is now documented as a limitation in
CHANGES_README.md rather than silently worked around.

WHAT THIS SCRIPT DOES INSTEAD
-------------------------------
1. EXACT-DUPLICATE DETECTION: hashes the actual image bytes (content hash,
   not filename) so that two files with identical pixel content - e.g. the
   same image saved twice as "foo.jpg" and "foo.jpeg" - are recognized as
   duplicates. Only ONE copy of any duplicate is kept, and it is assigned
   to exactly one split. This prevents the same image (under a different
   filename/extension) from appearing in both train and test, which IS a
   real and fixable leakage risk we found in the data (see the
   "astro_infraT (1).jpeg" / "astro_infraT (1).jpg" pairs from --inspect).

2. STRATIFIED IMAGE-LEVEL SPLIT: after deduplication, each class's
   remaining unique images are split 70/15/15 (train/val/test) using ONE
   random seed. No other script re-splits anything - preprocess.py and
   train.py both consume these three folders as-is.

3. A manifest CSV + self-check that verifies no duplicate image (by
   content hash) ended up in more than one split.

This is a weaker guarantee than true patient-level splitting - if the
source data does contain multiple images per real-world patient under
totally unrelated filenames (no shared naming pattern AND no metadata),
this script cannot detect that, because there is no signal left to detect
it from. This limitation is intentional and documented, not hidden.
"""

import os
import csv
import random
import shutil
import hashlib
import argparse
from collections import defaultdict

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

SOURCE_DIR = r"C:\Users\bibha\Desktop\ML Project\sample\dataset\_raw_merged"

BASE_DEST = "dataset"
TRAIN_DEST = os.path.join(BASE_DEST, "Training")
VAL_DEST = os.path.join(BASE_DEST, "Validation")
TEST_DEST = os.path.join(BASE_DEST, "Testing")

VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

MANIFEST_PATH = os.path.join(BASE_DEST, "split_manifest.csv")
DUPLICATE_LOG_PATH = os.path.join(BASE_DEST, "duplicate_report.csv")


# ----------------------------------------------------------------------
# DUPLICATE DETECTION (content hash, not filename)
# ----------------------------------------------------------------------

def file_hash(path, chunk_size=65536):
    """Return a SHA-256 hash of the file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# CORE LOGIC
# ----------------------------------------------------------------------

def scan_source(source_dir):
    """
    Return {category: {content_hash: [filenames]}} and a flat duplicate
    report. Grouping key is the image's CONTENT HASH, so identical images
    saved under different filenames/extensions are recognized as the same
    underlying image.
    """
    data = {}
    dup_report_rows = []

    for category in sorted(os.listdir(source_dir)):
        cat_path = os.path.join(source_dir, category)
        if not os.path.isdir(cat_path):
            continue
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(VALID_EXTS)]
        if not images:
            continue

        groups = defaultdict(list)
        for img in images:
            full_path = os.path.join(cat_path, img)
            h = file_hash(full_path)
            groups[h].append(img)

        for h, files in groups.items():
            if len(files) > 1:
                dup_report_rows.append({
                    "class": category,
                    "content_hash": h,
                    "duplicate_files": "; ".join(files),
                    "kept": files[0],
                    "dropped": "; ".join(files[1:]),
                })

        data[category] = groups

    return data, dup_report_rows


def inspect(source_dir, sample_classes=5, sample_groups=8):
    """Dry run: show duplicate detection results, copy nothing."""
    data, dup_rows = scan_source(source_dir)
    print("=" * 70)
    print("DUPLICATE-DETECTION INSPECTION (dry run, nothing copied)")
    print("=" * 70)
    for category in list(data.keys())[:sample_classes]:
        groups = data[category]
        n_images = sum(len(v) for v in groups.values())
        n_unique = len(groups)
        n_dupes = n_images - n_unique
        print(f"\nClass: {category}")
        print(f"  {n_images} files -> {n_unique} unique images "
              f"({n_dupes} exact duplicate file(s) found)")
        shown = 0
        for h, files in groups.items():
            if len(files) > 1:
                print(f"    duplicate content, keeping '{files[0]}', "
                      f"dropping {files[1:]}")
                shown += 1
            if shown >= sample_groups:
                break
    total_dupes = len(dup_rows)
    print(f"\nTotal duplicate groups found across all classes: {total_dupes}")
    print("=" * 70)
    print("Re-run WITHOUT --inspect to perform the actual stratified split "
          "(duplicates will be removed automatically, one copy kept).")
    print("=" * 70)


def stratified_split(items, train_ratio, val_ratio, test_ratio, rng):
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def do_split(source_dir):
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0"

    rng = random.Random(RANDOM_SEED)
    data, dup_rows = scan_source(source_dir)

    for d in (TRAIN_DEST, VAL_DEST, TEST_DEST):
        os.makedirs(d, exist_ok=True)

    manifest_rows = []

    for category, groups in data.items():
        # one representative filename per unique content hash
        unique_hashes = list(groups.keys())
        train_h, val_h, test_h = stratified_split(
            unique_hashes, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, rng
        )

        train_cat_dir = os.path.join(TRAIN_DEST, category)
        val_cat_dir = os.path.join(VAL_DEST, category)
        test_cat_dir = os.path.join(TEST_DEST, category)
        for d in (train_cat_dir, val_cat_dir, test_cat_dir):
            os.makedirs(d, exist_ok=True)

        counts = {"train": 0, "val": 0, "test": 0}

        def place(hash_list, split_name, dest_dir):
            for h in hash_list:
                fname = groups[h][0]  # keep only the first copy of any duplicate
                src = os.path.join(source_dir, category, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copy(src, dst)
                counts[split_name] += 1
                manifest_rows.append({
                    "filename": fname,
                    "class": category,
                    "content_hash": h,
                    "split": split_name,
                    "duplicate_count": len(groups[h]),
                })

        place(train_h, "train", train_cat_dir)
        place(val_h, "val", val_cat_dir)
        place(test_h, "test", test_cat_dir)

        n_dupes_dropped = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
        print(f"Processed {category}: {len(groups)} unique images "
              f"({n_dupes_dropped} duplicate file(s) dropped) -> "
              f"{counts['train']} train, {counts['val']} val, {counts['test']} test")

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "class", "content_hash", "split", "duplicate_count"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nManifest written to {MANIFEST_PATH} (use this to audit the split).")

    with open(DUPLICATE_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "content_hash", "duplicate_files", "kept", "dropped"]
        )
        writer.writeheader()
        writer.writerows(dup_rows)
    print(f"Duplicate report written to {DUPLICATE_LOG_PATH} "
          f"({len(dup_rows)} duplicate group(s) found).")

    verify_no_content_leakage(MANIFEST_PATH)


def verify_no_content_leakage(manifest_path):
    """
    Hard self-check: no image CONTENT HASH should appear in more than one
    split. Since duplicates are collapsed to one copy before splitting,
    this should always pass by construction - this check exists to catch
    any future code change that breaks that guarantee.
    """
    hash_splits = defaultdict(set)
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            hash_splits[row["content_hash"]].add(row["split"])

    leaked = {h: splits for h, splits in hash_splits.items() if len(splits) > 1}

    print("\n" + "=" * 70)
    if leaked:
        print(f"LEAKAGE CHECK: FAILED - {len(leaked)} image content hash(es) "
              f"appear in more than one split! This should not be possible "
              f"given how do_split() works - investigate immediately.")
        for h, splits in list(leaked.items())[:10]:
            print(f"    content_hash='{h[:16]}...' appears in: {sorted(splits)}")
    else:
        print(f"LEAKAGE CHECK: PASSED - no duplicate image content spans "
              f"multiple splits ({len(hash_splits)} unique images total).")
        print("NOTE: This confirms no IDENTICAL image file ended up in two "
              "splits. It does NOT confirm patient-level separation - this "
              "dataset's filenames do not contain recoverable patient IDs "
              "(see module docstring / CHANGES_README.md). This is a "
              "documented limitation of the source data, not an oversight.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified image-level dataset splitter with duplicate detection."
    )
    parser.add_argument("--inspect", action="store_true",
                         help="Dry run: show duplicate detection results, copy nothing.")
    parser.add_argument("--source", default=SOURCE_DIR,
                         help="Path to merged raw source dataset (one folder per class).")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.source)
    else:
        do_split(args.source)
        print("\nDataset splitting complete! (Training / Validation / Testing)")