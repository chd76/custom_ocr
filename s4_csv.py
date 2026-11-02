"""Stage 4 - the output table of one run.

output/runs/<run_name>/days/<date>.csv   the table of one day
output/runs/<run_name>/all_data.csv      all days in one table (the days joined)
output/runs/runs.csv                     one line per run: when, settings, counts

Only days that changed (new or redone pages, new filter decisions, other skip_labels / id_widths) are built again;
all_data.csv is then joined from the day files. --redo builds every day again.

Every row has an id in the FIRST column, plus every part of it as its own column:
    row_type = text_group     20110612-07739-008-2-003-001   article 3, group 1
    row_type = skipped_block  20110612-07739-008-2-000-001   not a text group (image, page header, page number ...;
                                                              see skip_labels), article 000, numbered 1..N per page
    row_type = page           20110612-00451-005-1-000-000   a page without results (trashed, not run yet, error)
Rows are in page order, and inside a page in Paddle's reading order. For the corpus keep row_type = text_group.
More about each page (file, scanner image id, filter details): output/registry/pages.csv and page_filter.csv, by page_id.

Article = one PaddleOCR region (PP-DocBlockLayout), numbered 1..N per page in Paddle's reading order; groups 1..N inside
each article, also in reading order. Only articles with at least one text group get a number.
    python s4_csv.py
"""

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime

from common import (FILTER_CSV, RUNS_CSV, load_settings, make_id, raw_path, read_csv, read_pages,
                    replace_file, run_dir, stage_args, write_csv, write_json)

FIELDS = ["id", "row_type", "date", "newspaper_id", "newspaper", "page", "page_version", "article", "group", "category",
          "text", "ocr_conf", "x1", "y1", "x2", "y2", "page_decision", "note", "page_id", "article_id", "run_name"]
FORMAT_VERSION = 3  # change when the table layout changes, so every day is built again


def merge_text_blocks(blocks, skip, page_height):
    """Join a text block with the one before it (in reading order) when both are 'text' in the same article and the
    second starts right below the first in the same column (PaddleOCR often splits one column of text in two)."""
    out = []
    for b in blocks:
        prev = next((o for o in reversed(out) if o["label"] not in skip and (o["text"] or "").strip()), None)
        same_column = prev and min(prev["bbox"][2], b["bbox"][2]) - max(prev["bbox"][0], b["bbox"][0]) \
            >= 0.5 * min(prev["bbox"][2] - prev["bbox"][0], b["bbox"][2] - b["bbox"][0])
        right_below = prev and 0 <= b["bbox"][1] - prev["bbox"][3] <= 0.02 * page_height
        if (prev and out[-1] is prev and b["label"] == prev["label"] == "text" and (b["text"] or "").strip()
                and b["article_region"] == prev["article_region"] and same_column and right_below):
            lines = prev["lines"] + b["lines"]
            confs = [(c, n) for c, n in ((prev["ocr_conf"], prev["lines"]), (b["ocr_conf"], b["lines"])) if c is not None]
            prev.update(text=prev["text"].rstrip() + "\n" + b["text"].strip(), lines=lines,
                        bbox=[min(prev["bbox"][0], b["bbox"][0]), min(prev["bbox"][1], b["bbox"][1]),
                              max(prev["bbox"][2], b["bbox"][2]), max(prev["bbox"][3], b["bbox"][3])],
                        ocr_conf=sum(c * n for c, n in confs) / max(1, sum(n for _, n in confs)) if confs else None)
        else:
            out.append(dict(b))
    return out


def page_rows(s, p, f, skip):
    """All rows of one page."""
    base = {"date": p["date"], "newspaper_id": p["newspaper_id"], "newspaper": p["newspaper"], "page": p["page"],
            "page_version": p["page_version"], "page_decision": f.get("decision", ""), "page_id": p["page_id"],
            "run_name": s["run_name"]}
    ids = (s, p["date"], p["newspaper_id"], p["page"], p["page_version"])
    raw = raw_path(s, p)
    if not f:
        status, note = "not filtered yet", "not filtered yet"
    elif f["use_page"] != "yes":
        status, note = "not used", f"not used: {f['reason']}"
    elif not raw.exists() and raw.with_suffix(".error.txt").exists():
        status, note = "OCR error", f"OCR error, details in {raw.with_suffix('.error.txt').name} next to the raw files"
    elif not raw.exists():
        status, note = "OCR not run yet", "OCR not run yet"
    else:
        status = None
    if status:
        return status, [{**base, "id": make_id(*ids, 0, 0), "row_type": "page", "article": 0, "group": 0,
                         "note": note}]

    data = json.loads(raw.read_text(encoding="utf-8"))
    blocks = sorted(data["blocks"], key=lambda b: b["paddle_index"])
    if s.get("merge_text_blocks"):
        blocks = merge_text_blocks(blocks, skip, data["height"])
    rows, art_num, art_size, n_skipped = [], {}, Counter(), 0
    for b in blocks:
        text = (b["text"] or "").strip()
        x1, y1, x2, y2 = b["bbox"]
        row = {**base, "category": b["label"], "text": text, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
               "ocr_conf": "" if b["ocr_conf"] is None else round(b["ocr_conf"], 3)}
        if b["label"] in skip or not text:
            n_skipped += 1
            row.update(row_type="skipped_block", article=0, group=n_skipped, id=make_id(*ids, 0, n_skipped),
                       note=f"{b['label']} is in skip_labels" if b["label"] in skip else "no text")
        else:
            art = art_num.setdefault(b["article_region"], len(art_num) + 1)
            art_size[art] += 1
            row.update(row_type="text_group", article=art, group=art_size[art],
                       id=make_id(*ids, art, art_size[art]), article_id=make_id(*ids, art))
        rows.append(row)
    if not art_num:
        rows.insert(0, {**base, "id": make_id(*ids, 0, 0), "row_type": "page", "article": 0, "group": 0,
                        "note": "OCR done, but no text groups found"})
    return "OCR done", rows


def fingerprint(s, day_pages, filt):
    """Changes when anything that goes into the day's table changes."""
    h = hashlib.sha1(json.dumps([FORMAT_VERSION, s["skip_labels"], s["id_widths"], str(s["run_name"]),
                                 bool(s.get("merge_text_blocks"))]).encode())
    for p in day_pages:
        f = filt.get(p["page_id"], {})
        raw = raw_path(s, p)
        st = raw.stat() if raw.exists() else None
        h.update(json.dumps([p["page_id"], p["newspaper"], f.get("decision"), f.get("use_page"), f.get("reason"),
                             st and (st.st_mtime_ns, st.st_size), raw.with_suffix(".error.txt").exists()]).encode())
    return h.hexdigest()


def join_days(day_files, out):
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as dst:
        dst.write(",".join(FIELDS) + "\r\n")
        for path in day_files:
            with path.open(encoding="utf-8-sig", newline="") as src:
                src.readline()  # header
                shutil.copyfileobj(src, dst)
    replace_file(tmp, out)


def update_runs_csv(s, totals):
    paddle = {k: v for k, v in (s["paddle"]["init"] or {}).items() if v is not None}
    row = {"run_name": s["run_name"], "built": datetime.now().strftime("%Y-%m-%d %H:%M"), **totals,
           "paddle_settings": json.dumps(paddle, ensure_ascii=False),
           "skip_labels": " ".join(s["skip_labels"])}
    rows = [r for r in read_csv(RUNS_CSV) if r["run_name"] != str(s["run_name"])] + [row]
    write_csv(RUNS_CSV, rows, list(row.keys()))


def main():
    s = load_settings(stage_args(__doc__).parse_args())
    out_dir = run_dir(s)
    days_dir = out_dir / "days"
    skip = set(s["skip_labels"])
    filt = {r["page_id"]: r for r in read_csv(FILTER_CSV)}
    by_day = defaultdict(list)
    for p in read_pages():
        by_day[p["date"]].append(p)

    info_file = days_dir / "_build_info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() and not s["redo"] else {}
    built, removed = 0, 0
    for date, pages in sorted(by_day.items()):
        fp = fingerprint(s, pages, filt)
        day_file = days_dir / f"{date}.csv"
        if day_file.exists() and info.get(date, {}).get("fingerprint") == fp:
            continue
        rows, status = [], Counter()
        for p in sorted(pages, key=lambda p: p["page_id"]):
            f = filt.get(p["page_id"], {})
            st, prow = page_rows(s, p, f, skip)
            status[st] += 1
            rows += prow
        groups = [r for r in rows if r["row_type"] == "text_group"]
        if len(groups) != len({g["id"] for g in groups}):
            raise SystemExit(f"{date}: text group IDs are not unique - check id_widths in settings.yaml")
        write_csv(day_file, rows, FIELDS)
        info[date] = {"fingerprint": fp, "pages": len(pages), "status": dict(status),
                      "text_groups": len(groups), "articles": len({g["article_id"] for g in groups}),
                      "skipped_blocks": sum(1 for r in rows if r["row_type"] == "skipped_block")}
        built += 1
        print(f"  {date}: {len(pages)} pages, {info[date]['text_groups']} text groups, "
              f"{info[date]['articles']} articles -> days/{date}.csv")

    for old in days_dir.glob("*.csv"):  # a day folder that was removed from the input
        if old.stem not in by_day:
            old.unlink()
            info.pop(old.stem, None)
            removed += 1
    info = {d: v for d, v in info.items() if d in by_day}
    write_json(info_file, info)

    out = out_dir / "all_data.csv"
    if built or removed or not out.exists():
        join_days([days_dir / f"{d}.csv" for d in sorted(by_day)], out)

    status = Counter()
    for v in info.values():
        status.update(v["status"])
    totals = {"days": len(info), "pages": sum(v["pages"] for v in info.values()),
              "pages_ocr_done": status["OCR done"], "pages_not_used": status["not used"],
              "pages_not_run_yet": status["OCR not run yet"], "pages_ocr_error": status["OCR error"], "pages_not_filtered_yet": status["not filtered yet"],
              "text_groups": sum(v["text_groups"] for v in info.values()),
              "articles": sum(v["articles"] for v in info.values()),
              "skipped_blocks": sum(v["skipped_blocks"] for v in info.values())}
    update_runs_csv(s, totals)
    print(f"run '{s['run_name']}': {built} day(s) built, {len(by_day) - built} unchanged -> {out}")
    print("  " + "   ".join(f"{k}: {v}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
