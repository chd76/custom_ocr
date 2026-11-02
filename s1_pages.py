"""Stage 1 - list every page of every day folder under input_root -> output/registry/pages.csv and newspapers.csv.

input_root/
    12 HAZİRAN 2011/                 day folder  -> date 20110612
        AKŞAM ANKARA(6573)/          newspaper   -> name AKŞAM ANKARA, id 6573
            11_350977398.jpg         page        -> page 11, scanner image id 350977398

Every page gets its page_id, e.g. 20110612-06573-011-1 (date, newspaper, page, version).
version = 1 normally; 2, 3 ... when the same page number exists more than once in one newspaper and day (by image id).
New day folders are simply picked up the next time this runs; nothing else needs to change.
    python s1_pages.py
"""

from collections import Counter, defaultdict

from common import (NEWSPAPERS_CSV, PAGES_CSV, IdTooWide, load_settings, make_id, parse_day, parse_newspaper,
                    parse_page, path_setting, stage_args, write_csv)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
FIELDS = ["page_id", "date", "newspaper_id", "newspaper", "page", "page_version", "image_id", "stem", "file",
          "day_folder", "newspaper_folder"]


def main():
    s = load_settings(stage_args(__doc__).parse_args())
    root = path_setting(s["input_root"])
    rows, ignored, problems = [], [], []
    for day in sorted(p for p in root.iterdir() if p.is_dir()):
        date = parse_day(day.name)
        if not date:
            ignored.append(day.name)
            continue
        for paper in sorted(p for p in day.iterdir() if p.is_dir()):
            name, nid = parse_newspaper(paper.name)
            if nid is None:
                problems.append(f"no newspaper id in the folder name: {day.name}/{paper.name}")
                continue
            for img in sorted(f for f in paper.iterdir() if f.suffix.lower() in IMAGE_EXTS):
                page, image_id = parse_page(img.stem)
                if page is None:
                    problems.append(f"no page number in the file name: {day.name}/{paper.name}/{img.name}")
                    continue
                rows.append({"date": date, "newspaper_id": nid, "newspaper": name, "page": page, "image_id": image_id,
                             "stem": img.stem, "file": str(img.relative_to(root)), "day_folder": day.name,
                             "newspaper_folder": paper.name})
        loose = [f for f in day.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if loose:
            problems.append(f"{len(loose)} pages directly in {day.name} (not in a newspaper folder)")

    rows.sort(key=lambda r: (r["date"], r["newspaper_id"], r["page"], r["image_id"]))
    seen = Counter()
    for r in rows:
        seen[(r["date"], r["newspaper_id"], r["page"])] += 1
        r["page_version"] = seen[(r["date"], r["newspaper_id"], r["page"])]
        try:
            r["page_id"] = make_id(s, r["date"], r["newspaper_id"], r["page"], r["page_version"])
        except IdTooWide as e:
            raise SystemExit(f"{r['file']}: {e}")
    write_csv(PAGES_CSV, rows, FIELDS)

    papers = defaultdict(lambda: {"names": set(), "days": set(), "pages": 0})
    for r in rows:
        e = papers[r["newspaper_id"]]
        e["names"].add(r["newspaper"])
        e["days"].add(r["date"])
        e["pages"] += 1
    write_csv(NEWSPAPERS_CSV, [{"newspaper_id": k, "newspaper": " | ".join(sorted(v["names"])), "days": len(v["days"]),
                                "first_day": min(v["days"]), "last_day": max(v["days"]), "pages": v["pages"]}
                               for k, v in sorted(papers.items())])

    days = Counter(r["date"] for r in rows)
    print(f"{len(rows)} pages, {len(papers)} newspapers, {len(days)} days -> {PAGES_CSV.parent}")
    for d, n in sorted(days.items()):
        print(f"  {d}: {n} pages")
    dup = [k for k, n in seen.items() if n > 1]
    if dup:
        print(f"  {len(dup)} page numbers appear more than once in one newspaper and day -> page_version 2, 3 ...:"
              f" {', '.join(f'{d} newspaper {n} page {p}' for d, n, p in dup[:5])}")
    for p in problems:
        print(f"  ! skipped: {p}")
    if ignored:
        print(f"  (folders in input_root that are not day folders, ignored: {', '.join(ignored)})")


if __name__ == "__main__":
    main()
