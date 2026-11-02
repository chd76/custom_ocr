"""Shared helpers for all stages. Only needs the standard library + PyYAML, so every environment can import it.

Output layout:
    output/registry/            stable, does not depend on PaddleOCR settings
        pages.csv               every page ever seen (stage 1), key = page_id
        newspapers.csv          newspaper id -> name, first/last day, pages (stage 1)
        page_filter.csv         keep / mixed / trash + reason per page (stage 2)
    output/draws/<draw>.csv     the pages of one random page draw (select.draw)
    output/runs/runs.csv        one line per run (stage 4)
    output/runs/<run_name>/
        settings_used.yaml      the settings of the last PaddleOCR run
        raw/<date>/<newspaper>/<page_id>.json   PaddleOCR result per page (stage 3)
        days/<date>.csv         the table of one day (stage 4)
        all_data.csv            all days in one table (stage 4)
        check/                  pictures (stage 5)
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import yaml

# a Windows console can't show every Turkish letter (ı, İ, ş, ğ): show "?" instead of crashing; files are always UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
REGISTRY = OUTPUT / "registry"
PAGES_CSV = REGISTRY / "pages.csv"
NEWSPAPERS_CSV = REGISTRY / "newspapers.csv"
FILTER_CSV = REGISTRY / "page_filter.csv"
RUNS = OUTPUT / "runs"
RUNS_CSV = RUNS / "runs.csv"
DRAWS = OUTPUT / "draws"

MONTHS = {"OCAK": 1, "SUBAT": 2, "MART": 3, "NISAN": 4, "MAYIS": 5, "HAZIRAN": 6, "TEMMUZ": 7,
          "AGUSTOS": 8, "EYLUL": 9, "EKIM": 10, "KASIM": 11, "ARALIK": 12}
_ASCII = str.maketrans("İIŞĞÜÖÇıişğüöç", "IISGUOCiisguoc")


# ---------------------------------------------------------------- settings
def stage_args(description):
    """Every stage takes --settings, --set key=value (repeatable) and --redo."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--settings", default=str(HERE / "settings.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="change one setting for this run, e.g. --set paddle.init.text_det_thresh=0.25")
    ap.add_argument("--redo", action="store_true", help="redo pages that already have results")
    return ap


def load_settings(args):
    s = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))
    for item in args.set:
        key, _, value = item.partition("=")
        node = s
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    if getattr(args, "redo", False):
        s["redo"] = True
    return s


def path_setting(value, base=HERE):
    return (Path(base) / value).resolve()


def python_for(s, stage):
    """Interpreter of a stage: settings python.<stage> (or python.default) is an environment folder or a python file."""
    p = path_setting(s["python"].get(stage) or s["python"]["default"])
    if p.is_dir():
        p = p / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return p


def run_dir(s):
    return RUNS / str(s["run_name"])


# ---------------------------------------------------------------- names, dates
def fix_turkish(name):
    """Folder names were unpacked with the wrong code page ('HAZÝRAN' = 'HAZİRAN'); repair them."""
    try:
        name = name.encode("cp1252").decode("cp1254")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return re.sub(r"\s+", " ", name).strip()


def parse_day(folder_name):
    """'12 HAZİRAN 2011' -> '20110612' (None if the folder isn't a day folder)."""
    m = re.fullmatch(r"\s*(\d{1,2})\s+(\S+)\s+(\d{4})\s*", fix_turkish(folder_name))
    if not m:
        return None
    month = MONTHS.get(m.group(2).translate(_ASCII).upper())
    return f"{m.group(3)}{month:02d}{int(m.group(1)):02d}" if month else None


def parse_newspaper(folder_name):
    """'AKŞAM ANKARA(6573)' -> ('AKŞAM ANKARA', 6573)."""
    m = re.fullmatch(r"(.*?)\s*\((\d+)\)\s*", fix_turkish(folder_name))
    return (m.group(1), int(m.group(2))) if m else (fix_turkish(folder_name), None)


def parse_page(stem):
    """'11_350977398' -> (11, '350977398')."""
    head, _, tail = stem.partition("_")
    return (int(head), tail) if head.isdigit() else (None, stem)


# ---------------------------------------------------------------- ids
class IdTooWide(ValueError):
    pass


def make_id(s, date, newspaper, page, version, article=None, group=None):
    """Fixed-width text id, each level starts with the one above it:
    page    20110612-07739-008-2
    article 20110612-07739-008-2-003
    group   20110612-07739-008-2-003-001     (article 000 = not in an article; group 000 = the whole page)"""
    w = s["id_widths"]
    parts = [str(date)]
    for name, value in (("newspaper", newspaper), ("page", page), ("version", version), ("article", article),
                        ("group", group)):
        if value is None:
            break
        value = int(value)
        if value < 0 or len(str(value)) > w[name]:
            raise IdTooWide(f"{name} = {value} does not fit in {w[name]} digits (settings.yaml -> id_widths)")
        parts.append(f"{value:0{w[name]}d}")
    return "-".join(parts)


def raw_path(s, page_row):
    """PaddleOCR result of one page: runs/<run>/raw/<date>/<newspaper>/<page_id>.json"""
    pid = page_row["page_id"]
    return run_dir(s) / "raw" / page_row["date"] / pid.split("-")[1] / f"{pid}.json"


# ---------------------------------------------------------------- files
def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def replace_file(tmp, path):
    """Put a finished temporary file in place (a stopped run never leaves a half-written file)."""
    try:
        os.replace(tmp, path)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Can't write {path}: it is open in another program (Excel?). Close it and run again.")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel shows Turkish letters
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    replace_file(tmp, path)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    replace_file(tmp, path)


def read_pages():
    rows = read_csv(PAGES_CSV)
    if not rows:
        raise SystemExit(f"No page list at {PAGES_CSV}. Run the 'pages' stage first.")
    return rows


def select_pages(rows, s):
    """select.days / newspapers / pages (file name without .jpg, or page_id) / draw / limit from settings.yaml."""
    sel = s.get("select") or {}
    days = {str(d) for d in sel.get("days") or []}
    papers = {str(int(n)) for n in sel.get("newspapers") or []}
    wanted = {str(p).replace("_", "") for p in sel.get("pages") or []}  # YAML reads 8_350977330 as the number 8350977330
    out = [r for r in rows if (not days or r["date"] in days) and (not papers or r["newspaper_id"] in papers)
           and (not wanted or r["stem"].replace("_", "") in wanted or r["page_id"] in wanted)]
    if sel.get("draw"):
        drawn = draw_pages(out, sel)
        out = [r for r in out if r["page_id"] in drawn]
    elif sel.get("random"):
        raise SystemExit("select.random needs a select.draw name, so the drawn pages are saved and can be used again")
    return out[: sel["limit"]] if sel.get("limit") else out


# ---------------------------------------------------------------- random page draws
def used_test_pages(skip_draw=None):
    """page_ids already looked at: pages of every run that had a page selection (whole-day runs, which select
    nothing, don't count) and of every saved draw except skip_draw."""
    used = set()
    for d in RUNS.iterdir() if RUNS.exists() else []:
        if not (d / "raw").is_dir():
            continue
        f = d / "settings_used.yaml"
        sel = ((yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("select") or {}) if f.exists() else None
        if sel is not None and not any(sel.get(k) for k in ("days", "newspapers", "pages", "limit", "draw")):
            continue  # a whole-day run
        used |= {p.stem for p in (d / "raw").rglob("*.json")}
    for f in DRAWS.glob("*.csv"):
        if f.stem != skip_draw:
            used |= {r["page_id"] for r in read_csv(f)}
    return used


def draw_pages(rows, sel):
    """page_ids of the draw select.draw. The first time, select.random used (keep/mixed) pages are drawn at random
    from rows and saved in output/draws/<draw>.csv; after that the saved list is used, so every run of a draw
    reads exactly the same pages."""
    import random
    from datetime import datetime

    name = str(sel["draw"])
    path = DRAWS / f"{name}.csv"
    if path.exists():
        saved = read_csv(path)
        if sel.get("random") and int(sel["random"]) != len(saved):
            print(f"  draw {name} already exists with {len(saved)} pages: select.random={sel['random']} is ignored")
        return {r["page_id"] for r in saved}
    if not sel.get("random"):
        raise SystemExit(f"Draw {name} doesn't exist yet ({path}). Set select.random=N to draw N pages for it.")
    filt = {r["page_id"]: r for r in read_csv(FILTER_CSV)}
    if not filt:
        raise SystemExit(f"No {FILTER_CSV}. Run the 'filter' stage first: a draw takes only keep/mixed pages.")
    usable = [r for r in rows if filt.get(r["page_id"], {}).get("use_page") == "yes"]
    exclude = used_test_pages(skip_draw=name) if sel.get("exclude_used") else set()
    pool = [r for r in usable if r["page_id"] not in exclude]
    n = int(sel["random"])
    if n > len(pool):
        raise SystemExit(f"Draw {name}: {n} pages wanted, but only {len(pool)} pages can be drawn")
    seed = random.SystemRandom().randrange(10 ** 9)
    picked = sorted(random.Random(seed).sample(pool, n), key=lambda r: r["page_id"])
    made = f"{datetime.now():%Y-%m-%d %H:%M}"
    write_csv(path, [{"page_id": r["page_id"], "newspaper": r["newspaper"], "page": r["page"], "stem": r["stem"],
                      "draw": name, "made": made, "seed": seed, "exclude_used": bool(sel.get("exclude_used")),
                      "drawn_from": len(pool), "left_out_as_used": len(usable) - len(pool)} for r in picked])
    print(f"  draw {name}: {n} pages drawn at random from {len(pool)} keep/mixed pages"
          f" ({len(usable) - len(pool)} left out as already used) -> {path}")
    return {r["page_id"] for r in picked}
