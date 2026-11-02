# newspaper_pipeline: scanned newspaper pages → one CSV of text groups

A pipeline that turns scanned Turkish newspaper pages into a table of text groups with stable IDs, for a corpus for
computational social science research.

Scanned pages of every day → **page filter** (keep / mixed / trash) → **PaddleOCR PP-StructureV3** (layout blocks,
article areas, reading order, text with PP-OCRv6) → **output table** with IDs (date – newspaper – page – article – group).

Python only. All settings are in `settings.yaml`; every stage can be switched on or off and run on its own.

> **No data is included.** This repository holds code, settings and the trained page-filter model only:
> no page images, no OCR output, no page lists. The scans and anything made from them are subject to the rights of
> the newspapers and the data provider.

## Folder layout

```
<parent folder>/
  newspaper_pipeline/               <- this repository
    settings.yaml                   all settings (paths, models, thresholds, IDs)
    run.py                          runs the stages
    common.py, s1_pages.py, s2_filter.py, s3_paddle.py, s4_csv.py, s5_check.py
    compare_runs.py                 HTML / PDF report of one run, or several runs side by side
    page_filter/                    page filter tool (label, review, train) + the trained model
    setup/                          setup_envs.py, requirements-*.txt, lock-*.txt (exact versions)
  newspaper_data/                   the page images, OUTSIDE the repository (settings: input_root)
    12 HAZİRAN 2011/AKŞAM(451)/1_350976...jpg
```

Input: one folder per day (e.g. `12 HAZİRAN 2011`), holding one folder per newspaper named `NAME(id)`, holding the pages
`<page number>_<image id>.jpg`. Folder names with wrongly decoded Turkish letters (`HAZÝRAN`) are repaired automatically.

## Setup (once per computer)

Needs [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with a recent driver (CPU works but is much slower).
Works on Windows and Linux:
```
python setup/setup_envs.py
```
It creates two environments and checks that both see the GPU:

| Environment | For | Main packages |
|---|---|---|
| `.venv-main` | page filter (stage 2) | ultralytics, PyTorch |
| `.venv-paddle` | all other stages | paddlepaddle-gpu, paddleocr |

They are separate because PyTorch and PaddlePaddle bring their own CUDA libraries. `--locked` installs the exact versions in
`setup/lock-*.txt`; `--torch-cuda` / `--paddle-cuda` pick other CUDA builds (default cu128 / cu129, needed for RTX 50xx).
Set `input_root` in `settings.yaml` to where the day folders are. PaddleOCR downloads its models on first use
(into `~/.paddlex/official_models`).

## Run

From this folder (on Linux the interpreter is `.venv-paddle/bin/python`):
```
.venv-paddle/Scripts/python.exe run.py
```
That runs the stages that are `true` under `stages:` in `settings.yaml`, each in its own environment. Other ways:
```
.venv-paddle/Scripts/python.exe run.py --stages paddle,csv
.venv-paddle/Scripts/python.exe run.py --set run_name=test --set select.limit=10
.venv-paddle/Scripts/python.exe s3_paddle.py --set select.newspapers=[451]
```
Pages that already have a result are skipped, so a run can be stopped (Ctrl+C) and started again; `--redo` does them again.

## Stages

| Stage | Script | What it does | Result (in `output/`) |
|---|---|---|---|
| pages | `s1_pages.py` | lists every page of every day folder and gives it its page_id | `registry/pages.csv`, `registry/newspapers.csv` |
| filter | `s2_filter.py` | keep / mixed / trash per page: your own label in `page_filter/labels.csv` first, else the page-filter model. Trash is skipped, but every page keeps a row with the reason | `registry/page_filter.csv` |
| paddle | `s3_paddle.py` | PaddleOCR PP-StructureV3 on the used pages (keep + mixed) | `runs/<run_name>/raw/<date>/<newspaper>/<page_id>.json` |
| csv | `s4_csv.py` | the output table, per day and all days together | `runs/<run_name>/days/<date>.csv`, `all_data.csv` |
| check | `s5_check.py` | page pictures with articles (colour + number) and groups (`article.group category`) drawn on | `runs/<run_name>/check/...jpg` |

### What the default does (and what is not plain PaddleOCR)

**PaddleOCR's own settings** (only the values are chosen; see `settings.yaml`, `paddle.init`):
- Text line detection and recognition: **PP-OCRv6_medium_det / PP-OCRv6_medium_rec**.
- Layout: PP-DocLayout_plus-L; article areas: PP-DocBlockLayout (`use_region_detection: true`).
- `text_det_unclip_ratio: 1.3` (Paddle default 1.5), so line boxes don't reach into neighbouring blocks.
- `layout_threshold` 1.01 for the classes image and chart: they are never detected, so a big photo box cannot swallow
  the text printed on it.
- Orientation, unwarping, tables, formulas, charts and seals are switched off.

**Three small changes to PP-StructureV3's own steps** (in `s3_paddle.py`, rule-based, switchable in `settings.yaml`):

| Change | Why |
|---|---|
| `tag_blocks_with_article` | PP-StructureV3 knows which article area each block is in but does not return it; this keeps the number. No text or order changes. |
| `drop_bogus_lines` (`paddle.drop_bogus_lines`) | Drops line pieces with no text and pieces more than 2.5× taller than the block's normal lines (they glue lines together and shuffle the text). A tall 1–2 letter piece is a drop cap: its letter is glued to its line. |
| `own_lines` (`paddle.line_owner`) | PP-StructureV3 gives a text line to every layout block it touches by more than 3 px and re-reads a straight cut of it for each block. That produces thin strips read as nonsense, half lines, and tilted text cut through several lines. Here a line goes only to the block holding most of it (a line running across two blocks side by side is still split). |

**Our own steps outside PaddleOCR**: the page filter (stage 2), joining text blocks that Paddle split inside one column
(`merge_text_blocks`), leaving page headers / footers / page numbers out of the text (`skip_labels`), and the IDs.

## The table (`days/<date>.csv` and `all_data.csv`)

The **id is always the first column**; every part of it is also its own column. Rows are in page order, and inside a page in
Paddle's reading order. Every page appears, also the unused ones.

| `row_type` | One row per | id example |
|---|---|---|
| `text_group` | text group (one text chunk) | `20110612-07739-008-2-003-001` |
| `skipped_block` | block that is not a text group (image, page header/footer, page number, table; see `skip_labels`) | `20110612-07739-008-2-000-004` |
| `page` | page without results: trashed by the filter, not run yet, or an error | `20110612-00451-005-1-000-000` |

Columns: `id, row_type, date, newspaper_id, newspaper, page, page_version, article, group, category, text, ocr_conf,
x1, y1, x2, y2, page_decision, note, page_id, article_id, run_name`. For the corpus keep `row_type = text_group`.

**IDs** are text with fixed widths, each level starting with the one above it: page `20110612-07739-008-2`
(date, newspaper id, page, version), article `…-003`, group `…-003-001`. Read them as **text** (pandas `dtype=str`,
R `colClasses="character"`). The page_id never changes; article and group numbers depend on the OCR settings, so every row
has its `run_name`. `category` is PaddleOCR's block type: `doc_title` (headline), `paragraph_title` (subheading), `text`,
`abstract` (lead), `figure_title` (caption), `aside_text`.

## Page filter

`page_filter/` holds a small image classifier (Ultralytics YOLO26n-cls, fine-tuned) that sorts whole pages into
keep / mixed / trash, and the tools to label pages, review the model's suggestions and retrain it (see
`page_filter/README.md`). The model may only trash a page when it is at least `page_filter.trash_min_confidence` (0.90) sure.
`page_filter/labels.csv` is empty here: add your own labels (`page,category`); they always override the model.

## Reports, tests and comparisons

```
.venv-paddle/Scripts/python.exe compare_runs.py <run>                 one run: output/report_<run>.html (--pdf for a PDF)
.venv-paddle/Scripts/python.exe compare_runs.py <run_a> <run_b>       runs side by side: output/compare_<runs>.html
```
Give each try its own `run_name`. For fair comparisons use a random page draw:
`--set select.draw=draw01 --set select.random=50` draws 50 keep/mixed pages the first time and saves them in
`output/draws/draw01.csv`; later runs with `select.draw=draw01` use exactly the same pages.

## Known limitations

- PP-OCRv6 has no Turkish language model behind it: ı/i, ğ/g, ş/s mix-ups and words like "dosyalan" for "dosyaları".
- PaddleOCR adds a space at a line end only after the letters A–Z, so a line ending in ı ş ğ ç ö ü is glued to the next
  line's first word ("tutulmasıgelecek").
- A word hyphenated at the bottom of a column continues in the next text group and stays split.
- Tilted text (e.g. in a rotated box) is read correctly but its lines can come out in the wrong order.
- Article grouping is PaddleOCR's region detection and can put several short items into one article.
- Speed (laptop RTX 5070): about 3 s per page.

## License and citations

The code in this repository is licensed under the **GNU Affero General Public License v3.0** (`LICENSE`), because it uses
Ultralytics YOLO (AGPL-3.0). The page-filter model in `page_filter/runs/` is fine-tuned from Ultralytics YOLO26 weights
(AGPL-3.0). PaddleOCR, PaddlePaddle, PaddleX and the PaddleOCR models are Apache-2.0.

**If you use this pipeline, please cite the tools it is built on.** All components, versions, licenses and BibTeX
entries are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); the main ones are:

- PaddleOCR 3.0 Technical Report (Cui et al., 2025), arXiv:2507.05595: PP-StructureV3, PP-OCR models, layout models.
- Ultralytics YOLO26 (Jocher et al., 2026), arXiv:2606.03748: page filter.
