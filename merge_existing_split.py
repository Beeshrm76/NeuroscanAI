"""
Merge an existing (leaky) dataset/Training + dataset/Testing split back into
one raw folder per class, so split_data.py has a proper unsplit source to
work from.

WHEN TO USE THIS
-----------------
Only if you no longer have the original downloaded dataset folder (before
any train/test split), and the only copies of your images are already
inside dataset/Training and dataset/Testing from the OLD random split.

This merges both back together per class into dataset/_raw_merged/<class>/,
so that all of a given class's images (regardless of which old split they
were randomly thrown into) are back in one place. You then point
split_data.py's SOURCE_DIR at dataset/_raw_merged and run the new
patient-level split from there.

Note: if the same filename happens to exist in both Training and Testing
for the same class (shouldn't normally happen, but flagged just in case),
the Testing copy will overwrite the Training copy with an identical
filename - this script will warn you if that happens so you can check.

Usage:
    python merge_existing_split.py
"""

import os
import shutil

TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"
MERGED_DIR = "dataset/_raw_merged"

VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def merge():
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
        print(f"Expected both '{TRAIN_DIR}' and '{TEST_DIR}' to exist. "
              f"Check your paths.")
        return

    classes = set(os.listdir(TRAIN_DIR)) | set(os.listdir(TEST_DIR))
    os.makedirs(MERGED_DIR, exist_ok=True)

    total_copied = 0
    total_collisions = 0

    for class_name in sorted(classes):
        dest_class_dir = os.path.join(MERGED_DIR, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)

        seen_filenames = set()
        for source_dir in (TRAIN_DIR, TEST_DIR):
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(VALID_EXTS):
                    continue
                if fname in seen_filenames:
                    total_collisions += 1
                    print(f"  [WARNING] Duplicate filename '{fname}' in class "
                          f"'{class_name}' found in both Training and Testing - "
                          f"keeping the Testing copy. Check this file manually "
                          f"if it matters (could be a real duplicate, or just a "
                          f"coincidental filename clash between two different "
                          f"images).")
                seen_filenames.add(fname)
                src = os.path.join(class_dir, fname)
                dst = os.path.join(dest_class_dir, fname)
                shutil.copy(src, dst)
                total_copied += 1

        print(f"Merged '{class_name}': {len(seen_filenames)} unique images "
              f"-> {dest_class_dir}")

    print(f"\nDone. {total_copied} files copied, {total_collisions} filename "
          f"collision(s) flagged above.")
    print(f"\nNow set SOURCE_DIR in split_data.py to the full path of "
          f"'{MERGED_DIR}' (or pass --source pointing at it), and run "
          f"split_data.py --inspect.")


if __name__ == "__main__":
    merge()
