"""Categorize ALL pages and write one table, for the other steps (e.g. skip ads and classifieds).

output/page_categories.csv, one row per page:
    page, file, page_number, page_id, category, confidence, second_choice, second_confidence,
    source ("your label" if you categorized/checked the page, else "model"), check (yes = low confidence),
    use_page (no = the page is in the skip list, e.g. trash; the rest of the pipeline skips it)

Usage:
    predict.py                     all pages in the images folder of config.yaml
    predict.py --pages FOLDER      another folder
    predict.py --models-only       ignore your labels, always use the model (e.g. to compare)
"""

import argparse
import csv

from common import MODEL, OUTPUT, list_images, load_config, page_info, read_labels, read_pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", help="folder of page images (default: images in config.yaml)")
    ap.add_argument("--models-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if not MODEL.exists():
        raise SystemExit(f"No model at {MODEL}. Run train.py first.")
    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    labels, pending = read_labels(cfg), read_pending(cfg)
    pages = list_images(args.pages or cfg["images"])
    rows = []
    for i in range(0, len(pages), 32):
        batch = pages[i:i + 32]
        for p, r in zip(batch, model.predict([str(q) for q in batch], imgsz=cfg["model"]["imgsz"],
                                             verbose=False, device=0)):
            top = r.probs.top5[:2]
            conf = [float(r.probs.data[k]) for k in top]
            num, pid = page_info(p.stem)
            row = {"page": p.stem, "file": p.name, "page_number": num, "page_id": pid,
                   "category": r.names[top[0]], "confidence": round(conf[0], 3),
                   "second_choice": r.names[top[1]] if len(top) > 1 else "",
                   "second_confidence": round(conf[1], 3) if len(top) > 1 else "", "source": "model"}
            if not args.models_only and p.stem in labels and p.stem not in pending:
                row.update(category=labels[p.stem], source="your label")
            row["check"] = "yes" if row["source"] == "model" and row["confidence"] < cfg["review_below"] else ""
            row["use_page"] = "no" if row["category"] in cfg["skip"] else "yes"  # the rest of the pipeline reads this
            rows.append(row)
        print(f"{min(i + 32, len(pages))}/{len(pages)} pages", end="\r", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "page_categories.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"\n{len(rows)} pages -> {out}")
    print("pages per category: " + ", ".join(f"{c}: {n}" for c, n in sorted(counts.items(), key=lambda x: -x[1])))
    print(f"low confidence (check = yes): {sum(1 for r in rows if r['check'])}")
    print(f"use_page: yes {sum(1 for r in rows if r['use_page'] == 'yes')}, no {sum(1 for r in rows if r['use_page'] == 'no')}")


if __name__ == "__main__":
    main()
