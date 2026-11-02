"""Let the model categorize the next pages without a category, so you only check them (label.py --review).

Usage:
    autolabel.py               next 100 pages without a category
    autolabel.py --limit 300   (0 = all)
"""

import argparse

from common import MODEL, list_images, load_config, read_labels, read_pending, write_labels, write_pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    cfg = load_config()
    if not MODEL.exists():
        raise SystemExit(f"No model at {MODEL}. Run train.py first.")
    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    labels, pending = read_labels(cfg), read_pending(cfg)
    todo = [p for p in list_images(cfg["images"]) if p.stem not in labels]
    if args.limit:
        todo = todo[: args.limit]
    print(f"categorizing {len(todo)} pages")
    for i in range(0, len(todo), 32):
        for p, r in zip(todo[i:i + 32], model.predict([str(q) for q in todo[i:i + 32]], imgsz=cfg["model"]["imgsz"],
                                                     verbose=False, device=0)):
            labels[p.stem] = r.names[r.probs.top1]
            pending[p.stem] = float(r.probs.top1conf)
    write_labels(cfg, labels)
    write_pending(cfg, pending)
    print(f"{len(pending)} pages waiting for review -> label.py --review (training skips them until checked)")


if __name__ == "__main__":
    main()
