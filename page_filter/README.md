# Page filter: keep / mixed / trash

A whole-page image classifier with three labels. It decides **which pages go into the pipeline**.
Part of `newspaper_pipeline`: stage `filter` follows your labels here (`labels.csv`) first and uses this model for the rest.
It only **reads** the page images in `images` (`config.yaml`; default: the same `newspaper_data` folder as the pipeline).
The trained model is `runs/page_filter/weights/best.pt`; `labels.csv` is empty in this repository (no page data).

**Run the tools** from the `newspaper_pipeline` folder with the main environment, for example:
```
.venv-main/Scripts/python.exe page_filter/label.py
.venv-main/Scripts/python.exe page_filter/label.py --review
.venv-main/Scripts/python.exe page_filter/train.py
```

| Key | Label | Meaning | In the pipeline |
|---|---|---|---|
| 1 | **keep** | the page has articles worth extracting: news, front page, columns, magazine, sports, interviews… | used |
| 2 | **mixed** | real articles **and** a big non-article part (about ⅓ or more ads / classifieds) | used |
| 3 | **trash** | nothing worth extracting: full-page ad, classifieds, puzzles, TV guide, service pages | **skipped** |

## Tools

| Run | What it does |
|---|---|
| `label.py` | one key per page (1 / 2 / 3); the next page opens. Zoom with the mouse wheel, F = whole page, H = help |
| `label.py --review` | pages the model labeled: **D** if the suggestion is right, or the right number |
| `label.py --labeled` | go over all labeled pages again |
| `train.py` | trains the classifier; prints a mix-up table and the two numbers that matter (below) |
| `autolabel.py` | the model labels the next 100 pages, which then go to `label.py --review` |
| `predict.py` | labels **all** pages and writes `output/page_categories.csv` with a **`use_page`** column (yes / no) |
| `status.py` | progress per label |

## The two numbers `train.py` prints
- **useful pages wrongly thrown away:** keep or mixed pages the model called trash. **This is the important one,** because that text would be lost.
- **trash pages wrongly kept:** only costs some processing time later.

**Tips:**
- If useful pages get thrown away, label more **trash-looking but useful** pages (e.g. pages with a big ad and a small article = mixed).
- When in doubt between mixed and trash, choose **mixed**.
