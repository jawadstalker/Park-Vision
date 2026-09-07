import argparse
import os
import random
import shutil

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_split_dirs(root):
    splits = []
    for split in ("train", "valid", "val", "test"):
        images_dir = os.path.join(root, split, "images")
        labels_dir = os.path.join(root, split, "labels")
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            splits.append((images_dir, labels_dir))

    if not splits:
        images_dir = os.path.join(root, "images")
        labels_dir = os.path.join(root, "labels")
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            splits.append((images_dir, labels_dir))

    return splits


def load_class_names(root):
    yaml_path = os.path.join(root, "data.yaml")
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config.get("names")


def remap_label_file(source_path, dest_path, keep_class_index):
    kept = 0
    with open(source_path) as f_in, open(dest_path, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            if class_id == keep_class_index:
                f_out.write("0 " + " ".join(parts[1:]) + "\n")
                kept += 1
    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Convert a downloaded external YOLO dataset (e.g. a Roboflow export) into "
        "this project's naming convention, keeping only one target class."
    )
    parser.add_argument("source_dir", help="Root of the extracted external dataset export.")
    parser.add_argument("output_images_dir")
    parser.add_argument("output_labels_dir")
    parser.add_argument(
        "--source-name",
        default="external",
        help="Tag used as the 'street' field in output filenames (default: external).",
    )
    parser.add_argument(
        "--lighting",
        default="day",
        choices=["day", "dusk", "night"],
        help="Lighting condition to tag these images with (default: day).",
    )
    parser.add_argument(
        "--keep-class",
        default="car",
        help="Only annotations for this class name are kept, remapped to class id 0 (default: car).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=20,
        help="Split the output across this many pseudo-street tags (source-name1..N), round-robin, "
        "so organize_dataset.py's per-street train/val/test split doesn't dump the whole external "
        "dataset into a single bucket (default: 20).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Randomly sample at most this many images from the source dataset (default: use all).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for --limit sampling.")
    args = parser.parse_args()

    class_names = load_class_names(args.source_dir)
    if class_names is None:
        raise SystemExit(f"Could not find data.yaml under {args.source_dir}.")
    if args.keep_class not in class_names:
        raise SystemExit(f"Class '{args.keep_class}' not found in dataset classes: {class_names}")
    keep_index = class_names.index(args.keep_class)

    splits = find_split_dirs(args.source_dir)
    if not splits:
        raise SystemExit(
            "Could not find images/labels folders under source_dir "
            "(expected train/valid/test/images+labels, or a flat images/labels)."
        )

    pairs = []
    for images_dir, labels_dir in splits:
        for name in sorted(os.listdir(images_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = os.path.join(labels_dir, stem + ".txt")
            if not os.path.exists(label_path):
                continue
            pairs.append((os.path.join(images_dir, name), label_path, ext))

    if not pairs:
        raise SystemExit("No matching image/label pairs found under source_dir.")

    if args.limit is not None and args.limit < len(pairs):
        random.seed(args.seed)
        pairs = random.sample(pairs, args.limit)

    os.makedirs(args.output_images_dir, exist_ok=True)
    os.makedirs(args.output_labels_dir, exist_ok=True)

    total_boxes = 0
    for index, (image_path, label_path, ext) in enumerate(pairs):
        chunk_number = (index % args.chunks) + 1
        new_stem = f"{args.source_name}{chunk_number}_{args.lighting}_{index:05d}"
        shutil.copy2(image_path, os.path.join(args.output_images_dir, new_stem + ext))
        kept = remap_label_file(
            label_path,
            os.path.join(args.output_labels_dir, new_stem + ".txt"),
            keep_index,
        )
        total_boxes += kept

    print(
        f"Prepared {len(pairs)} images ({total_boxes} '{args.keep_class}' boxes kept) -> "
        f"{args.output_images_dir} / {args.output_labels_dir}"
    )


if __name__ == "__main__":
    main()
