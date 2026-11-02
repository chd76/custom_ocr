"""How far page categorization is."""

from collections import Counter

from common import MODEL, list_images, load_config, read_labels, read_pending


def main():
    cfg = load_config()
    images = list_images(cfg["images"])
    labels, pending = read_labels(cfg), read_pending(cfg)
    stems = {p.stem for p in images}
    checked = {s: c for s, c in labels.items() if s in stems and s not in pending}
    waiting = sum(1 for s in pending if s in stems)
    counts = Counter(checked.values())
    print(f"pages:                              {len(images)}")
    print(f"  categorized and checked:          {len(checked)}   (used for training)")
    print(f"  suggested by the model, to check: {waiting}   -> label.py --review")
    print(f"  no category yet:                  {len(images) - len(checked) - waiting}   -> label.py / autolabel.py")
    print("checked pages per category:")
    for c in cfg["categories"]:
        print(f"  {c:14s} {counts.get(c, 0)}")
    print(f"trained model: {'yes' if MODEL.exists() else 'no'}")


if __name__ == "__main__":
    main()
