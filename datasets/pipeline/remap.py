#!/usr/bin/env python3
"""
Drop specified classes from a YOLO dataset and remap remaining class IDs
to be contiguous (closing the gaps left by the removed classes).

Rewrites every label .txt file in train/ and valid/ in place, and writes
a new data.yaml with the compacted class list.

Usage:
    python remap_drop_classes.py --dataset PATH_TO_DATASET
"""

import argparse
import shutil
import yaml
from pathlib import Path

SPLITS = ["train", "valid"]

# Edit this list if you want to drop different classes.
CLASSES_TO_DROP = ["turnstile_gate", "podium", "pillar", "room_sign", "head_hazard"]


def load_yaml_names(dataset_root: Path):
    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def build_id_remap(old_names, classes_to_drop):
    """Returns (new_names, old_id_to_new_id dict). Dropped classes map to None."""
    new_names = [n for n in old_names if n not in classes_to_drop]
    old_id_to_new_id = {}
    for old_id, name in enumerate(old_names):
        if name in classes_to_drop:
            old_id_to_new_id[old_id] = None
        else:
            old_id_to_new_id[old_id] = new_names.index(name)
    return new_names, old_id_to_new_id


def remap_label_file(label_path: Path, id_map: dict, stats: dict):
    lines_out = []
    dropped_here = 0
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            new_id = id_map.get(old_id)
            if new_id is None:
                dropped_here += 1
                continue
            lines_out.append(f"{new_id} {' '.join(parts[1:])}")

    if dropped_here:
        stats["dropped_boxes"] += dropped_here

    if lines_out:
        with open(label_path, "w") as f:
            f.write("\n".join(lines_out) + "\n")
        return "kept"
    else:
        # No boxes left in this file at all (either it was empty already,
        # or every box in it belonged to a dropped class).
        label_path.unlink()
        return "emptied"


def process_split(dataset_root: Path, split: str, id_map: dict, stats: dict):
    lbl_dir = dataset_root / split / "labels"
    img_dir = dataset_root / split / "images"
    if not lbl_dir.exists():
        print(f"  split '{split}' not found, skipping")
        return

    label_files = list(lbl_dir.glob("*.txt"))
    emptied = 0
    for label_path in label_files:
        result = remap_label_file(label_path, id_map, stats)
        if result == "emptied":
            emptied += 1
            # Remove the now-orphaned image too, so images/ and labels/ stay in sync.
            stem = label_path.stem
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                img_path = img_dir / f"{stem}{ext}"
                if img_path.exists():
                    img_path.unlink()
                    break

    print(f"  [{split}] processed {len(label_files)} label files, "
          f"{emptied} images removed (no boxes left after drop)")


def write_new_yaml(dataset_root: Path, new_names: list):
    yaml_path = dataset_root / "data.yaml"
    backup_path = dataset_root / "data.yaml.bak"
    shutil.copy2(yaml_path, backup_path)
    with open(yaml_path, "w") as f:
        f.write(f"train: {dataset_root.resolve()}/train/images\n")
        f.write(f"val: {dataset_root.resolve()}/valid/images\n")
        f.write(f"nc: {len(new_names)}\n")
        f.write(f"names: {new_names}\n")
    print(f"\nWrote updated data.yaml (backup saved as {backup_path.name})")


def main():
    parser = argparse.ArgumentParser(description="Drop unused classes and remap YOLO label IDs.")
    parser.add_argument("--dataset", type=Path, required=True,
                         help="Path to the merged YOLO dataset (contains data.yaml, train/, valid/)")
    parser.add_argument("--classes-to-drop", nargs="+", default=CLASSES_TO_DROP,
                         help="Class names to remove")
    args = parser.parse_args()

    dataset_root = args.dataset
    old_names = load_yaml_names(dataset_root)
    print(f"Current classes ({len(old_names)}): {old_names}")

    missing = [c for c in args.classes_to_drop if c not in old_names]
    if missing:
        raise SystemExit(f"These classes-to-drop are not in data.yaml: {missing}")

    new_names, id_map = build_id_remap(old_names, args.classes_to_drop)
    print(f"\nDropping: {args.classes_to_drop}")
    print(f"New classes ({len(new_names)}): {new_names}")
    print(f"\nID remap (old -> new, None = dropped):")
    for old_id, name in enumerate(old_names):
        print(f"  {old_id:2d} {name:20s} -> {id_map[old_id]}")

    stats = {"dropped_boxes": 0}
    print("\nRewriting label files...")
    for split in SPLITS:
        process_split(dataset_root, split, id_map, stats)

    write_new_yaml(dataset_root, new_names)

    print(f"\nDone. Total individual bounding boxes dropped: {stats['dropped_boxes']} "
          f"(expect 0, since these classes had no instances)")


if __name__ == "__main__":
    main()