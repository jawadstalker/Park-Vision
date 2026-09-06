import argparse
import os
import random
import shutil
from collections import defaultdict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_street(filename):
    return filename.split("_")[0]


def collect_pairs(images_dir, labels_dir):
    pairs = []
    for name in os.listdir(images_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(label_path):
            continue
        pairs.append((os.path.join(images_dir, name), label_path, stem))
    return pairs


def split_by_street(pairs, test_ratio, val_ratio, seed):
    random.seed(seed)
    by_street = defaultdict(list)
    for pair in pairs:
        street = parse_street(os.path.basename(pair[0]))
        by_street[street].append(pair)

    streets = list(by_street.keys())
    random.shuffle(streets)

    total = len(pairs)
    target_test = total * test_ratio

    test_streets = []
    test_count = 0
    for street in streets:
        if test_count >= target_test:
            break
        test_streets.append(street)
        test_count += len(by_street[street])

    remaining_streets = [s for s in streets if s not in test_streets]
    remaining_pairs = [p for s in remaining_streets for p in by_street[s]]
    random.shuffle(remaining_pairs)

    val_share = val_ratio / max(1e-9, (1 - test_ratio))
    val_count = int(round(len(remaining_pairs) * val_share))
    val_pairs = remaining_pairs[:val_count]
    train_pairs = remaining_pairs[val_count:]
    test_pairs = [p for s in test_streets for p in by_street[s]]

    return train_pairs, val_pairs, test_pairs


def write_split(pairs, split_name, output_root):
    images_out = os.path.join(output_root, "images", split_name)
    labels_out = os.path.join(output_root, "labels", split_name)
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    for image_path, label_path, stem in pairs:
        shutil.copy2(image_path, os.path.join(images_out, os.path.basename(image_path)))
        shutil.copy2(label_path, os.path.join(labels_out, stem + ".txt"))


def write_data_yaml(output_root, class_names):
    content = (
        f"path: {os.path.abspath(output_root)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    with open(os.path.join(output_root, "data.yaml"), "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Split a labeled YOLO dataset into train/val/test, keeping test streets separate."
    )
    parser.add_argument("images_dir", help="Directory with labeled images.")
    parser.add_argument("labels_dir", help="Directory with YOLO .txt label files matching image filenames.")
    parser.add_argument("output_dir", help="Directory to write the organized dataset.")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", default=["car"])
    args = parser.parse_args()

    pairs = collect_pairs(args.images_dir, args.labels_dir)
    if not pairs:
        raise RuntimeError("No matching image/label pairs found.")

    train_pairs, val_pairs, test_pairs = split_by_street(
        pairs, args.test_ratio, args.val_ratio, args.seed
    )

    write_split(train_pairs, "train", args.output_dir)
    write_split(val_pairs, "val", args.output_dir)
    write_split(test_pairs, "test", args.output_dir)
    write_data_yaml(args.output_dir, args.classes)

    print(f"train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}")
    print(f"Wrote dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
