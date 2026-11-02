"""Compare runs side by side: one HTML page (and optionally a PDF) with a summary of each run (models, settings, speed,
counts) and, for every page, each run's check picture and its text groups (article.group, category, text) in reading order.
    python compare_runs.py cmp_current cmp_v6 cmp_vl
    python compare_runs.py sample50 sample50_v6 --pdf --title "PaddleOCR: old settings vs v6"
-> output/compare_<runs>.html (open it in a browser) and with --pdf also output/compare_<runs>.pdf (printed with Chrome)
Reads the day files (runs/<run>/days/*.csv), so it works while all_data.csv is open in Excel.
"""

import argparse
import html
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from common import OUTPUT, RUNS, raw_path, read_csv

CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
          "google-chrome", "chromium", "chromium-browser"]

CSS = """body{font-family:system-ui,Segoe UI,sans-serif;margin:16px;background:#fafafa;color:#222}
h1{margin-bottom:4px} .sub{color:#555;margin-top:0}
h2{margin-top:40px;border-top:2px solid #888;padding-top:12px;font-size:18px}
table.sum{border-collapse:collapse;font-size:12px;margin:12px 0}
table.sum th,table.sum td{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}
table.sum th{background:#eee}
.grid{display:grid;grid-template-columns:repeat(%d,minmax(0,1fr));gap:12px}
.col{background:#fff;border:1px solid #ddd;padding:8px;font-size:12px}
.col img{width:100%%;border:1px solid #ccc}
.g{margin:5px 0;padding:3px 6px;border-left:4px solid #bbb;break-inside:avoid}
.g b{color:#555;font-size:10px}
.doc_title,.paragraph_title{border-left-color:#d33;font-weight:600}
.stats{color:#666;font-size:11px;margin-bottom:6px}
.legend{font-size:12px;color:#444}
.single{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:14px;align-items:start}
.single .pic img{width:100%%;border:1px solid #ccc}
@page{size:A4 landscape;margin:10mm}
@media print{body{background:#fff;margin:0}.page{break-before:page}h2{margin-top:0;border-top:none}
 .col{border:none;padding:0 4px}a{color:inherit;text-decoration:none}}"""


def read_run(run):
    """Text groups per page_id, from the run's day files."""
    days = sorted((RUNS / run / "days").glob("*.csv"))
    if not days:
        raise SystemExit(f"run '{run}' has no tables - run the csv stage for it first")
    by_page = defaultdict(list)
    for d in days:
        for r in read_csv(d):
            if r["row_type"] == "text_group":
                by_page[r["page_id"]].append(r)
    return by_page


def describe(run):
    """Models and changed settings of a run, from its settings_used.yaml."""
    f = RUNS / run / "settings_used.yaml"
    if not f.exists():
        return "?", ""
    s = yaml.safe_load(f.read_text(encoding="utf-8"))
    init = s["paddle"]["init"]
    rec = init.get("text_recognition_model_name") or f"default for lang {init.get('lang')} ({init.get('ocr_version')})"
    models = (f"PP-StructureV3: layout {init.get('layout_detection_model_name') or 'PP-DocLayout_plus-L'}, "
              f"text detection {init.get('text_detection_model_name') or 'default'}, text recognition {rec}")
    notes = []
    if (s["paddle"].get("line_owner") or {}).get("enabled"):
        notes.append("line owner on")
    thr = init.get("layout_threshold")
    if isinstance(thr, dict) and any(v > 1 for v in thr.values()):
        notes.append("images/charts ignored (layout threshold 1.01)")
    return models, "; ".join(notes)


def find_chrome():
    for c in CHROME:
        if Path(c).exists() or shutil.which(c):
            return shutil.which(c) or c
    return None


def to_pdf(html_file, pdf_file, image_px, quality):
    """Print with Chrome/Edge. The pictures are first copied smaller (image_px wide, JPEG quality), so the PDF can be emailed."""
    from PIL import Image
    chrome = find_chrome()
    if not chrome:
        raise SystemExit("No Chrome/Edge found to print the PDF; open the HTML and print it to PDF by hand.")
    page = html_file.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for n, src in enumerate(sorted(set(re.findall(r"<img src='([^']+)'>", page)))):
            small = tmp / f"img{n}.jpg"
            im = Image.open(OUTPUT / src)
            im.thumbnail((image_px, image_px * 2))
            im.save(small, quality=quality, optimize=True)
            page = page.replace(f"<img src='{src}'>", f"<img src='{small.name}'>")
        printable = tmp / "print.html"
        printable.write_text(page, encoding="utf-8")
        # own profile: doesn't touch the Chrome you have open
        subprocess.run([chrome, "--headless=new", "--disable-gpu", f"--user-data-dir={tmp / 'profile'}",
                        "--no-pdf-header-footer", "--virtual-time-budget=60000", f"--print-to-pdf={pdf_file}",
                        printable.as_uri()], check=True, capture_output=True, timeout=900)
    print(f"PDF -> {pdf_file} ({pdf_file.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", help="run names (folders in output/runs)")
    ap.add_argument("--pdf", action="store_true", help="also print the page to PDF (Chrome/Edge)")
    ap.add_argument("--title", default="", help="title on top of the page")
    ap.add_argument("--pdf-image-px", type=int, default=1100, help="picture width in the PDF (smaller = smaller file)")
    ap.add_argument("--pdf-quality", type=int, default=60, help="JPEG quality of the pictures in the PDF")
    args = ap.parse_args()
    runs = args.runs
    if not runs:  # one run = a report, two or more = a comparison
        ready = sorted(d.name for d in RUNS.iterdir() if (d / "days").exists()) if RUNS.exists() else []
        raise SystemExit("Give two or more run names, e.g.\n    python compare_runs.py cmp_current cmp_v6 cmp_vl\n"
                         f"runs with results: {', '.join(ready) or 'none yet'}")
    tables = {run: read_run(run) for run in runs}
    pages = sorted(set.union(*(set(t) for t in tables.values())))  # a page missing in a run = no text groups there
    pages_csv = {r["page_id"]: r for r in read_csv(OUTPUT / "registry" / "pages.csv")}

    def seconds(run, pid):
        f = raw_path({"run_name": run}, pages_csv[pid])
        return json.loads(f.read_text(encoding="utf-8")).get("seconds") if f.exists() else None

    out = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(args.title or 'Compare runs')}</title>"
           f"<style>{CSS % len(runs)}</style>",
           f"<h1>{html.escape(args.title or 'Comparison of OCR runs')}</h1>",
           f"<p class='sub'>{len(pages)} newspaper pages (12 June 2011, Turkish newspapers)"
           f"{f', each processed by {len(runs)} runs' if len(runs) > 1 else ''}. For each page: the page with the found "
           f"articles drawn on it (one colour and number per article, grey = skipped, e.g. page headers) and the text "
           f"groups in reading order.</p>",
           "<table class='sum'><tr><th>Run</th><th>Models</th><th>Settings</th><th>Pages with text</th>"
           "<th>Text groups</th><th>Articles</th><th>Characters</th><th>Seconds / page (laptop RTX 5070)</th></tr>"]
    for run in runs:
        t = tables[run]
        secs = [x for x in (seconds(run, pid) for pid in pages) if x is not None]
        models, notes = describe(run)
        out.append(f"<tr><td><b>{html.escape(run)}</b></td><td>{html.escape(models)}</td><td>{html.escape(notes)}</td>"
                   f"<td>{len(t)} / {len(pages)}</td><td>{sum(len(v) for v in t.values())}</td>"
                   f"<td>{sum(len({g['article'] for g in v}) for v in t.values())}</td>"
                   f"<td>{sum(len(g['text']) for v in t.values() for g in v):,}</td>"
                   f"<td>{f'{sum(secs) / len(secs):.1f}' if secs else ''}</td></tr>")
    out.append("</table><p class='legend'>Labels: <b>article.group category</b>. Categories are PaddleOCR block types: "
               "doc_title = headline, paragraph_title = subheading, text = body text, abstract = lead, "
               "figure_title = caption, aside_text = side text.</p>")
    for pid in pages:
        p = pages_csv[pid]
        out.append(f"<div class='page'><h2>{pid} &mdash; {html.escape(p['newspaper'])}, page {p['page']}</h2>"
                   f"<div class='{'grid' if len(runs) > 1 else ''}'>")
        for run in runs:
            groups = tables[run].get(pid, [])
            if not groups:
                out.append(f"<div class='col'><h3>{html.escape(run)}</h3><p><b>no text groups on this page</b> "
                           f"(e.g. the whole page was taken as one image)</p></div>")
                continue
            img = f"runs/{run}/check/{p['date']}/{pid.split('-')[1]}/{pid}.jpg"
            sec = seconds(run, pid)
            stats = (f"<div class='stats'>{'' if sec is None else f'{sec} s &middot; '}{len(groups)} groups &middot; "
                     f"{len({g['article'] for g in groups})} articles &middot; "
                     f"{sum(len(g['text']) for g in groups):,} characters</div>")
            if len(runs) == 1:  # report: picture left, text right
                out.append(f"<div class='single'><div class='pic'>{stats}<a href='{img}' target='_blank'>"
                           f"<img src='{img}'></a></div><div class='col'>")
            else:
                out.append(f"<div class='col'><h3>{html.escape(run)}</h3>{stats}"
                           f"<a href='{img}' target='_blank'><img src='{img}'></a>")
            for g in groups:
                out.append(f"<div class='g {html.escape(g['category'])}'><b>{g['article']}.{g['group']} "
                           f"{html.escape(g['category'])}</b><br>{html.escape(g['text']).replace(chr(10), '<br>')}</div>")
            out.append("</div></div>" if len(runs) == 1 else "</div>")
        out.append("</div></div>")
    dst = OUTPUT / (f"report_{runs[0]}.html" if len(runs) == 1 else f"compare_{'_'.join(runs)}.html")
    dst.write_text("\n".join(out), encoding="utf-8")
    print(f"{len(pages)} pages -> {dst}")
    if args.pdf:
        to_pdf(dst, dst.with_suffix(".pdf"), args.pdf_image_px, args.pdf_quality)


if __name__ == "__main__":
    main()
