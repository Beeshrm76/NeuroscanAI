"""
find_cross_class_duplicates.py

WHY THIS EXISTS
----------------
split_data.py's duplicate detection only dedupes WITHIN a single class
folder before splitting. It cannot know if the exact same image also
exists in a DIFFERENT class folder (e.g. the same file physically present
in both "Meningioma T1" and "Meningioma T1C+").

When that happens, the split's own leakage self-check catches it AFTER
the fact (because the same content hash ends up in two different splits,
e.g. one copy assigned to train while processing class A, another copy
assigned to test while processing class B) - but it only reports 3-5
at a time, requiring you to fix and re-run repeatedly.

This script instead scans the WHOLE _raw_merged folder up front, hashes
every image regardless of which class folder it's in, and reports every
group of files that share identical content but live in DIFFERENT
classes - all at once, before you ever run split_data.py again.

USAGE
-----
    python find_cross_class_duplicates.py

This is read-only - it does NOT delete or move anything. It just prints
a report and writes cross_class_duplicates.csv so you can review and
decide what to remove.
"""

import os
import csv
import hashlib
from collections import defaultdict

SOURCE_DIR = r"C:\Users\bibha\Desktop\ML Project\sample\dataset\_raw_merged"
VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp")
REPORT_PATH = "cross_class_duplicates.csv"


def file_hash(path, chunk_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    hash_to_locations = defaultdict(list)  # hash -> [(class, filename), ...]

    classes = sorted(
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    )

    print(f"Scanning {len(classes)} class folders under {SOURCE_DIR} ...")

    total_files = 0
    for category in classes:
        cat_path = os.path.join(SOURCE_DIR, category)
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(VALID_EXTS)]
        for img in images:
            full_path = os.path.join(cat_path, img)
            h = file_hash(full_path)
            hash_to_locations[h].append((category, img))
            total_files += 1

    print(f"Hashed {total_files} files total across all classes.\n")

    # A cross-class duplicate = same hash appears under 2+ DIFFERENT class names
    cross_class_groups = {
        h: locs for h, locs in hash_to_locations.items()
        if len({cls for cls, _ in locs}) > 1
    }

    print("=" * 70)
    print(f"CROSS-CLASS DUPLICATE REPORT: {len(cross_class_groups)} duplicate "
          f"image(s) found across DIFFERENT classes")
    print("=" * 70)

    rows = []
    for h, locs in cross_class_groups.items():
        classes_involved = sorted({cls for cls, _ in locs})
        print(f"\ncontent_hash={h[:16]}...")
        for cls, fname in locs:
            print(f"    [{cls}]  {fname}")
        rows.append({
            "content_hash": h,
            "classes_involved": " | ".join(classes_involved),
            "locations": " ; ".join(f"{cls}/{fname}" for cls, fname in locs),
            "num_copies": len(locs),
        })

    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["content_hash", "classes_involved", "locations", "num_copies"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print(f"Full report written to {REPORT_PATH}")
    print("This script did NOT delete or move anything - review the report, "
          "then decide which copies to remove before re-running split_data.py.")
    print("=" * 70)

    # Also summarize which class PAIRS are most affected, since a repeated
    # pair (like "Meningioma T1" <-> "Meningioma T1C+") suggests a systematic
    # filing issue in the source data rather than one-off mistakes.
    pair_counts = defaultdict(int)
    for h, locs in cross_class_groups.items():
        classes_involved = sorted({cls for cls, _ in locs})
        if len(classes_involved) == 2:
            pair_counts[tuple(classes_involved)] += 1

    if pair_counts:
        print("\nMost affected class pairs:")
        for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
            print(f"    {pair[0]}  <->  {pair[1]}: {count} duplicate image(s)")


if __name__ == "__main__":
    main()