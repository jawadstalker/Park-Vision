import argparse
import os
from collections import Counter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(
        description="Report street and lighting distribution of a labeled image folder."
    )
    parser.add_argument("images_dir")
    args = parser.parse_args()

    street_counts = Counter()
    lighting_counts = Counter()
    total = 0

    for name in os.listdir(args.images_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        street, lighting = parts[0], parts[1]
        street_counts[street] += 1
        lighting_counts[lighting] += 1
        total += 1

    if total == 0:
        print("No images found matching the '<street>_<lighting>_<index>' naming convention.")
        return

    print(f"Total images: {total}")
    print("By street:")
    for street, count in sorted(street_counts.items(), key=lambda x: -x[1]):
        print(f"  {street}: {count} ({count / total:.1%})")
    print("By lighting (target: day 50%, dusk 30%, night 20%):")
    for lighting, count in sorted(lighting_counts.items(), key=lambda x: -x[1]):
        print(f"  {lighting}: {count} ({count / total:.1%})")


if __name__ == "__main__":
    main()
