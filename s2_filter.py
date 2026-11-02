"""Stage 2 - keep / mixed / trash for every page -> output/registry/page_filter.csv (key = page_id).

Runs in the PROJECT environment (.venv, it has ultralytics). Reads page_filter/ but never changes it.
    1. a page you labeled in page_filter (and checked)  -> your label is followed (trash = skipped)
    2. any other page                                   -> the page_filter model decides
       the model may only trash a page when it is at least page_filter.trash_min_confidence sure;
       otherwise the page is kept and the reason is written down.
Every page gets a row, also the skipped ones (use_page = no, with the reason).
Model predictions are saved, so a rerun only predicts new pages (--redo predicts all again).
    python s2_filter.py
"""

from pathlib import Path

import yaml

from common import FILTER_CSV, load_settings, path_setting, read_csv, read_pages, select_pages, stage_args, write_csv


def read_page_filter_labels(folder):
    labels = {r["page"]: r["category"] for r in read_csv(folder / "labels.csv") if r.get("page")}
    pending_file = folder / "pending_review.txt"
    pending = set()
    if pending_file.exists():
        pending = {line.split(",")[0].strip() for line in pending_file.read_text(encoding="utf-8-sig").splitlines()
                   if line.strip()}
    return {k: v for k, v in labels.items() if k not in pending}  # pre-labels you haven't checked don't count


def decide(row, label, s):
    pf = s["page_filter"]
    use = set(pf["use"])
    if label:
        row.update(decision=label, source="your label",
                   reason="your page_filter label")
    else:
        cat, conf = row["model_category"], float(row["model_confidence"])
        row.update(decision=cat, source="model", reason=f"page_filter model, {conf:.2f} sure")
        if cat not in use and conf < pf["trash_min_confidence"]:
            row["decision"] = row["model_second"] if row["model_second"] in use else pf["use"][0]
            row["reason"] = (f"model said {cat} but only {conf:.2f} sure (< {pf['trash_min_confidence']}): "
                             f"kept as {row['decision']}")
    row["use_page"] = "yes" if row["decision"] in use else "no"
    return row


def main():
    s = load_settings(stage_args(__doc__).parse_args())
    pf = s["page_filter"]
    folder = path_setting(pf["folder"])
    root = path_setting(s["input_root"])
    labels = read_page_filter_labels(folder)
    pages = select_pages(read_pages(), s)
    old = {r["page_id"]: r for r in read_csv(FILTER_CSV)}

    todo = [r for r in pages if r["stem"] not in labels and (s["redo"] or not old.get(r["page_id"], {}).get("model_category"))]
    if todo:
        from ultralytics import YOLO
        model_path = folder / pf["model"]
        if not model_path.exists():
            raise SystemExit(f"No page_filter model at {model_path}")
        imgsz = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))["model"]["imgsz"]
        model = YOLO(str(model_path))
        print(f"page_filter model on {len(todo)} pages without your label ...")
        for i in range(0, len(todo), pf["batch"]):
            batch = todo[i:i + pf["batch"]]
            for r, res in zip(batch, model.predict([str(root / p["file"]) for p in batch], imgsz=imgsz,
                                                   verbose=False, device=0)):
                top = res.probs.top5[:2]
                old[r["page_id"]] = {"model_category": res.names[top[0]],
                                    "model_confidence": round(float(res.probs.data[top[0]]), 3),
                                    "model_second": res.names[top[1]],
                                    "model_second_confidence": round(float(res.probs.data[top[1]]), 3)}
            print(f"  {min(i + pf['batch'], len(todo))}/{len(todo)}", end="\r", flush=True)
        print()

    out = {}
    for r in pages:
        prev = old.get(r["page_id"], {})
        row = {k: r[k] for k in ("page_id", "date", "newspaper_id", "newspaper", "page", "page_version", "stem")}
        row["your_label"] = labels.get(r["stem"], "")
        for k in ("model_category", "model_confidence", "model_second", "model_second_confidence"):
            row[k] = prev.get(k, "")
        out[r["page_id"]] = decide(row, row["your_label"], s)
    # keep rows of pages outside this selection
    merged = {**{k: v for k, v in old.items() if "decision" in v}, **out}
    fields = ["page_id", "date", "newspaper_id", "newspaper", "page", "page_version", "decision", "use_page", "source",
              "reason", "your_label", "model_category", "model_confidence", "model_second", "model_second_confidence",
              "stem"]
    write_csv(FILTER_CSV, sorted(merged.values(), key=lambda r: r["page_id"]), fields)  # page_id sorts by date, paper, page

    rows = list(out.values())
    count = lambda key, val: sum(1 for r in rows if r[key] == val)
    print(f"{len(rows)} pages -> {FILTER_CSV}")
    print(f"  your label: {count('source', 'your label')}   model: {count('source', 'model')}")
    print("  decision: " + "   ".join(f"{c}: {count('decision', c)}" for c in sorted({r['decision'] for r in rows})))
    print(f"  used (go to PaddleOCR): {count('use_page', 'yes')}   skipped: {count('use_page', 'no')}")
    kept = sum(1 for r in rows if "kept as" in r["reason"])
    if kept:
        print(f"  model said trash but wasn't sure enough, kept: {kept}")


if __name__ == "__main__":
    main()
