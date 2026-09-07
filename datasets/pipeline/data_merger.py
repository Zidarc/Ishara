#!/usr/bin/env python3
"""
Merge two YOLO-format datasets (train/valid images+labels + data.yaml)
into a single output dataset, remapping class IDs if the two datasets
have different class name lists.

Usage:
    python merge_yolo_datasets.py

Edit the paths/config below, or pass them as CLI args:
    python merge_yolo_datasets.py --dataset-a PATH_A --dataset-b PATH_B --output PATH_OUT
"""

import argparse
import shutil
import yaml
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "valid"]


def load_yaml_names(dataset_root: Path):
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    # names can be a list or a dict {id: name}
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def build_unified_classes(names_a, names_b):
    """Union of class names, preserving dataset A's order first."""
    unified = list(names_a)
    for name in names_b:
        if name not in unified:
            unified.append(name)
    return unified


def remap_label_file(src_label_path: Path, dst_label_path: Path, id_map: dict):
    """Rewrite a YOLO label file, remapping class IDs via id_map (old_id -> new_id)."""
    lines_out = []
    with open(src_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            if old_id not in id_map:
                # class not in this dataset's mapping (shouldn't happen) -> skip
                continue
            new_id = id_map[old_id]
            lines_out.append(f"{new_id} {' '.join(parts[1:])}")
    with open(dst_label_path, "w") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))


def merge_split(dataset_root: Path, split: str, out_root: Path, id_map: dict, prefix: str, stats: dict):
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"  [{prefix}] skipping split '{split}' (not found)")
        return

    out_img_dir = out_root / split / "images"
    out_lbl_dir = out_root / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    copied = 0
    for img_path in images:
        stem = img_path.stem
        label_path = lbl_dir / f"{stem}.txt"
        if not label_path.exists():
            continue  # skip unlabeled images

        # Prefix filenames to avoid collisions between the two source datasets
        new_stem = f"{prefix}_{stem}"
        out_img_path = out_img_dir / f"{new_stem}{img_path.suffix}"
        out_lbl_path = out_lbl_dir / f"{new_stem}.txt"

        if out_img_path.exists():
            print(f"    WARNING: collision on {out_img_path.name}, skipping duplicate")
            continue

        shutil.copy2(img_path, out_img_path)
        remap_label_file(label_path, out_lbl_path, id_map)
        copied += 1

    stats[(prefix, split)] = copied
    print(f"  [{prefix}] {split}: copied {copied} image/label pairs")


def write_merged_yaml(out_root: Path, class_names: list):
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {out_root.resolve()}/train/images\n")
        f.write(f"val: {out_root.resolve()}/valid/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    print(f"\nWrote merged data.yaml -> {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge two YOLO datasets into one, remapping class IDs.")
    parser.add_argument("--dataset-a", type=Path,
                         default=Path(r"E:\GitProjects\Ishara\Models\model_code\data\processed\labeled_dataset_1280x1280"))
    parser.add_argument("--dataset-b", type=Path,
                         default=Path(r"E:\GitProjects\Ishara\Models\model_code\data\processed\labeled_dataset_merged"))
    parser.add_argument("--output", type=Path,
                         default=Path(r"E:\GitProjects\Ishara\Models\model_code\data\processed\labeled_dataset_final"))
    args = parser.parse_args()

    dataset_a, dataset_b, out_root = args.dataset_a, args.dataset_b, args.output

    if out_root.exists():
        raise SystemExit(f"Output directory already exists, refusing to overwrite: {out_root}")

    print(f"Dataset A: {dataset_a}")
    print(f"Dataset B: {dataset_b}")
    print(f"Output:    {out_root}\n")

    names_a = load_yaml_names(dataset_a)
    names_b = load_yaml_names(dataset_b)
    print(f"Dataset A classes ({len(names_a)}): {names_a}")
    print(f"Dataset B classes ({len(names_b)}): {names_b}")

    unified_classes = build_unified_classes(names_a, names_b)
    print(f"\nUnified classes ({len(unified_classes)}): {unified_classes}")

    id_map_a = {i: unified_classes.index(name) for i, name in enumerate(names_a)}
    id_map_b = {i: unified_classes.index(name) for i, name in enumerate(names_b)}

    out_root.mkdir(parents=True, exist_ok=False)
    stats = {}

    print("\nMerging dataset A...")
    for split in SPLITS:
        merge_split(dataset_a, split, out_root, id_map_a, prefix="a", stats=stats)

    print("\nMerging dataset B...")
    for split in SPLITS:
        merge_split(dataset_b, split, out_root, id_map_b, prefix="b", stats=stats)

    write_merged_yaml(out_root, unified_classes)

    total = sum(stats.values())
    print(f"\nDone. Total image/label pairs merged: {total}")
    for (prefix, split), count in stats.items():
        print(f"  {prefix}/{split}: {count}")


if __name__ == "__main__":
    main()