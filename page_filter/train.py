"""Train the page filter (keep / mixed / trash, YOLO classification) on your categorized pages.

Pages the model suggested but you haven't checked yet are not used. A part of the pages is held
out; after training, a table shows for every category how its held-out pages were classified,
so you can see which categories get mixed up.

Usage:
    train.py
    train.py --epochs 80
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict

from common import MODEL, RUNS, WORK, link_or_copy, list_images, load_config, read_labels, read_pending

DATA = WORK / "dataset"


def main():
    ap = argparse.ArgumentParser(description="Train the page categorizer")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = load_config()
    m = cfg["model"]
    cats = cfg["categories"]
    labels, pending = read_labels(cfg), read_pending(cfg)
    images = {p.stem: p for p in list_images(cfg["images"])}
    by_cat = defaultdict(list)
    for stem, cat in labels.items():
        if stem in images and stem not in pending and cat in cats:
            by_cat[cat].append(images[stem])
    too_few = {c: len(v) for c, v in by_cat.items() if len(v) < 2}
    for c in too_few:  # a category needs pages in both the training and the held-out part
        del by_cat[c]
    total = sum(len(v) for v in by_cat.values())
    if total < 10 or len(by_cat) < 2:
        raise SystemExit("Categorize at least 10 pages in 2+ categories (2+ pages each) first (label.py).")
    missing = [c for c in cats if c not in by_cat]
    if missing:
        print(f"not trained yet (fewer than 2 checked pages): {', '.join(missing)}")

    # split per category, so every category with enough pages is represented in the held-out part
    rng = random.Random(m["seed"])
    shutil.rmtree(DATA, ignore_errors=True)
    held_out = []
    print(f"{'category':14s} train  held-out")
    for cat in (c for c in cats if c in by_cat):
        pages = sorted(by_cat[cat])
        rng.shuffle(pages)
        n_val = max(1, round(len(pages) * m["val_fraction"]))  # at least 1 held out, at least 1 for training
        for i, p in enumerate(pages):
            split = "val" if i < n_val else "train"
            link_or_copy(p, DATA / split / cat / p.name)
            if split == "val":
                held_out.append((p, cat))
        print(f"{cat:14s} {len(pages) - n_val:5d}  {n_val:8d}")
    print(f"{total} checked pages")

    from ultralytics import YOLO
    YOLO(m["base"]).train(data=str(DATA), epochs=args.epochs or m["epochs"], imgsz=m["imgsz"], batch=m["batch"],
                          fliplr=m["fliplr"], device=0, project=str(RUNS), name="page_filter", exist_ok=True)

    if held_out:
        model = YOLO(str(MODEL))
        table = defaultdict(Counter)
        for p, cat in held_out:
            r = model.predict(str(p), imgsz=m["imgsz"], verbose=False, device=0)[0]
            table[cat][r.names[r.probs.top1]] += 1
        right = sum(table[c][c] for c in table)
        print(f"\nheld-out pages right: {right}/{len(held_out)} = {100 * right / len(held_out):.1f} %")
        print("true category -> predicted as")
        for cat in cats:
            if table[cat]:
                print(f"  {cat:14s} " + ", ".join(f"{k}: {v}" for k, v in table[cat].most_common()))
        skip = set(cfg["skip"])  # the mistakes that matter: useful pages thrown away, trash kept
        lost = sum(n for t, row in table.items() if t not in skip for p, n in row.items() if p in skip)
        kept = sum(n for t, row in table.items() if t in skip for p, n in row.items() if p not in skip)
        useful = sum(sum(row.values()) for t, row in table.items() if t not in skip)
        trash = sum(sum(row.values()) for t, row in table.items() if t in skip)
        print(f"\nuseful pages wrongly thrown away: {lost}/{useful}   <- the important number (text would be lost)")
        print(f"trash pages wrongly kept:         {kept}/{trash}   (only costs some processing time)")
    print(f"\nmodel: {MODEL}")


if __name__ == "__main__":
    main()
