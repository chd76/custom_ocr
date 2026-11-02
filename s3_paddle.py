"""Stage 3 - PaddleOCR PP-StructureV3 on every used page (use_page = yes in page_filter.csv).

For each page: layout blocks (text, titles, ...), the article area (PP-DocBlockLayout region) of every block,
Paddle's reading order and the text (PP-OCRv6 text detection + recognition).
Result: output/runs/<run_name>/raw/<date>/<newspaper>/<page_id>.json   (one file per page)
Pages that already have a result are skipped, so it can be stopped and started again (--redo = do them again).
All PaddleOCR arguments are in settings.yaml (paddle); a new run_name keeps runs apart.

Three small changes to PP-StructureV3's own steps (switchable in settings.yaml, see README.md):
    tag_blocks_with_article   keeps the number of the article area each block belongs to (Paddle computes it but
                              does not return it); changes no text and no order
    drop_bogus_lines          drops line pieces with no text and pieces much taller than a block's normal lines
    own_lines                 a text line touching several blocks goes only to the block holding most of it

    python s3_paddle.py
    python s3_paddle.py --set run_name=test --set select.limit=5
"""

import os
import time
import traceback

import yaml

from common import (FILTER_CSV, load_settings, path_setting, raw_path, read_csv, read_pages, run_dir, select_pages,
                    stage_args, write_json)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def not_none(d):
    return {k: v for k, v in (d or {}).items() if v is not None}


def to_list(a):
    return a.tolist() if hasattr(a, "tolist") else list(a)


# ---------------------------------------------------------------- changes to PP-StructureV3's own steps
def tag_blocks_with_article():
    """PP-StructureV3 sorts the regions (article areas) first, then the blocks inside each region,
    but it doesn't keep the region number. Same sorting here, plus: every block remembers its region."""
    from paddlex.inference.pipelines.layout_parsing import pipeline_v2 as pv2

    def sort_layout_parsing_blocks(self, layout_parsing_page):
        out = []
        for r_idx, region in enumerate(pv2.xycut_enhanced(layout_parsing_page)):
            for block in pv2.xycut_enhanced(region):
                block.article_region = r_idx
                out.append(block)
        return out

    pv2._LayoutParsingPipelineV2.sort_layout_parsing_blocks = sort_layout_parsing_blocks


def weighted_conf(texts, scores):
    """Confidence of a block: mean of its lines' scores, weighted by line length (a 2-letter fragment counts little)."""
    w = [max(1, len(t)) for t in texts]
    return round(sum(float(s) * x for s, x in zip(scores, w)) / sum(w), 4) if w else None


def drop_bogus_lines(tall_factor):
    """Before PP-StructureV3 sorts the text lines of a block, drop pieces that can't be text lines: pieces with no text,
    and pieces much taller than the block's normal lines (only in blocks with 3+ lines, so a one-line headline is never
    touched). Such a piece - e.g. a tall strip over the stacked hyphens at a column's right edge - otherwise glues all
    lines it overlaps into one 'line', which Paddle then orders left to right, shuffling the text.
    A tall piece of 1-2 letters is a drop cap (the big first letter of an article): its letter is glued to the start
    of the line beside it ("E" + "RSAN" -> "ERSAN") before the piece is removed."""
    import statistics
    from paddlex.inference.pipelines.layout_parsing import layout_objects as lo
    orig = lo.LayoutBlock.group_boxes_into_lines
    if getattr(orig, "_drops_bogus", False):
        return

    def group_boxes_into_lines(self, ocr_rec_res, line_height_iou_threshold):
        boxes, texts = ocr_rec_res.get("boxes", []), list(ocr_rec_res.get("rec_texts", []))
        n = len(texts)
        heights = [b[3] - b[1] for b, tx in zip(boxes, texts) if (tx or "").strip()]
        med = statistics.median(heights) if len(heights) >= 3 else None
        tall = lambda i: med is not None and boxes[i][3] - boxes[i][1] > tall_factor * med
        for i in range(n):  # a tall piece of 1-2 letters is a DROP CAP: glue it to the first line beside it
            letter = (texts[i] or "").strip()
            if tall(i) and 1 <= len(letter) <= 2 and letter.isalpha():
                beside = [j for j in range(n) if j != i and not tall(j) and (texts[j] or "").strip()
                          and boxes[j][0] >= boxes[i][2] - 0.5 * med
                          and boxes[j][1] < boxes[i][3] and boxes[j][3] > boxes[i][1]]
                if beside:
                    j = min(beside, key=lambda j: boxes[j][1])
                    texts[j] = letter + texts[j].lstrip()
        keep = [i for i in range(n) if (texts[i] or "").strip() and not tall(i)]
        if len(keep) < n and keep:
            ocr_rec_res = dict(ocr_rec_res, rec_texts=texts)
            ocr_rec_res = {k: ([v[i] for i in keep] if hasattr(v, "__len__") and not isinstance(v, (str, dict))
                               and len(v) == n else v) for k, v in ocr_rec_res.items()}
        return orig(self, ocr_rec_res, line_height_iou_threshold)

    group_boxes_into_lines._drops_bogus = True
    lo.LayoutBlock.group_boxes_into_lines = group_boxes_into_lines


def own_lines(split_share, split_height):
    """PP-StructureV3 gives a text line to EVERY layout block it touches by more than 3 px, then cuts the line once per
    block as a STRAIGHT rectangle and re-reads each cut ('Replace the OCR information of the hurdles' in pipeline_v2).
    That makes thin nonsense strips, halves of lines and, for tilted text,
    straight cuts through 2-3 lines. Here a line touching several blocks goes only to the block
    holding most of it, so Paddle's own (tilted) reading of the whole line is kept. Paddle still splits a line that
    really runs across blocks side by side: the second block holds at least split_share of the line and both hold
    at least split_height of its height."""
    from paddlex.inference.pipelines.layout_parsing import pipeline_v2 as pv2
    if getattr(pv2.get_sub_regions_ocr_res, "_owns_lines", False):
        return
    orig_match, orig_remove = pv2.get_sub_regions_ocr_res, pv2.remove_overlap_blocks
    state = {"boxes": None, "owner": {}}

    def remove_overlap_blocks(layout_det_res, *a, **k):  # first step of standardized_data: remember the final blocks
        res = orig_remove(layout_det_res, *a, **k)
        state["boxes"] = [(list(map(float, b["coordinate"])), b["label"].lower()) for b in res["boxes"]
                          if b["label"].lower() not in ("formula", "table", "seal")]
        state["owner"] = {}
        return res

    def owners(i, line):
        """Blocks line i may go to: the one holding most of it, plus a side-by-side neighbour."""
        if i not in state["owner"]:
            x1, y1, x2, y2 = line
            area, h = max(1.0, (x2 - x1) * (y2 - y1)), max(1.0, y2 - y1)
            hits = []
            for k, (b, _) in enumerate(state["boxes"]):
                w, hh = min(x2, b[2]) - max(x1, b[0]), min(y2, b[3]) - max(y1, b[1])
                if w > 3 and hh > 3:  # Paddle's own rule (get_overlap_boxes_idx)
                    hits.append((w * hh / area, hh / h, k))
            hits.sort(reverse=True)
            keep = {hits[0][2]} if hits else set()
            keep |= {k for share, hshare, k in hits[1:]
                     if share >= split_share and hshare >= split_height and hits[0][1] >= split_height}
            state["owner"][i] = keep
        return state["owner"][i]

    def get_sub_regions_ocr_res(overall_ocr_res, object_boxes, *a, return_match_idx=False, **k):
        res = orig_match(overall_ocr_res, object_boxes, *a, return_match_idx=return_match_idx, **k)
        if not (return_match_idx and state["boxes"] and len(object_boxes) == 1):
            return res
        box = list(map(float, object_boxes[0]))
        me = next((k for k, (b, _) in enumerate(state["boxes"]) if b == box), None)
        if me is None:
            return res
        boxes = overall_ocr_res["rec_boxes"]
        return res[0], [i for i in res[1] if me in owners(i, list(map(float, boxes[i])))]

    get_sub_regions_ocr_res._owns_lines = True
    pv2.get_sub_regions_ocr_res, pv2.remove_overlap_blocks = get_sub_regions_ocr_res, remove_overlap_blocks


def block_ocr_conf(bbox, rec_boxes, rec_scores, rec_texts):
    """Confidence of the full-page text lines whose centre lies inside the block."""
    x1, y1, x2, y2 = bbox
    inside = [(t, s) for b, s, t in zip(rec_boxes, rec_scores, rec_texts)
              if x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2]
    return weighted_conf([t for t, _ in inside], [s for _, s in inside]), len(inside)


# ---------------------------------------------------------------- engine
class StructureEngine:
    def __init__(self, s):
        tag_blocks_with_article()
        dbl = s["paddle"].get("drop_bogus_lines")
        if dbl and dbl.get("enabled", True):
            drop_bogus_lines(float(dbl.get("tall_factor", 2.5)))
        lo = s["paddle"].get("line_owner") or {}
        if lo.get("enabled"):
            own_lines(float(lo.get("split_share", 0.3)), float(lo.get("split_height", 0.6)))
        from paddleocr import PPStructureV3
        self.pipeline = PPStructureV3(**not_none(s["paddle"]["init"]))
        self.predict_args = not_none(s["paddle"].get("predict"))

    def __call__(self, path):
        res = next(iter(self.pipeline.predict(path, **self.predict_args)))
        ocr = res["overall_ocr_res"]
        rec_boxes, rec_scores = to_list(ocr.get("rec_boxes", [])), to_list(ocr.get("rec_scores", []))
        rec_texts = list(ocr.get("rec_texts", []))
        blocks = []
        for b in res["parsing_res_list"]:
            bbox = [int(v) for v in b.bbox]
            conf, lines = block_ocr_conf(bbox, rec_boxes, rec_scores, rec_texts)
            blocks.append({"paddle_index": b.index, "label": b.label, "article_region": getattr(b, "article_region", 0),
                           "bbox": bbox, "text": b.content or "", "ocr_conf": conf, "lines": lines})
        regions = [[int(v) for v in x["coordinate"]] for x in res["region_det_res"].get("boxes", [])]
        return res["width"], res["height"], blocks, regions


def main():
    s = load_settings(stage_args(__doc__).parse_args())
    root = path_setting(s["input_root"])
    out_dir = run_dir(s)
    filt = {r["page_id"]: r for r in read_csv(FILTER_CSV)}
    if not filt:
        raise SystemExit(f"No {FILTER_CSV}. Run the 'filter' stage first.")
    pages = select_pages(read_pages(), s)
    missing = [r for r in pages if r["page_id"] not in filt]
    if missing:
        print(f"  ! {len(missing)} selected pages have no filter decision yet (run the 'filter' stage): skipped")
    used = [r for r in pages if filt.get(r["page_id"], {}).get("use_page") == "yes"]
    todo = [r for r in used if s["redo"] or not raw_path(s, r).exists()]
    print(f"{len(pages)} selected pages, {len(used)} used (keep/mixed), {len(todo)} still to do -> {out_dir / 'raw'}")
    if not todo:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "settings_used.yaml").write_text(yaml.safe_dump(s, allow_unicode=True, sort_keys=False), encoding="utf-8")
    engine = StructureEngine(s)

    errors, t0 = 0, time.time()
    for n, r in enumerate(todo, 1):
        dst = raw_path(s, r)
        dst.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        try:
            width, height, blocks, regions = engine(str(root / r["file"]))
            data = {"page_id": r["page_id"], "file": r["file"], "width": width, "height": height,
                    "run_name": s["run_name"], "seconds": round(time.time() - t, 1), "blocks": blocks,
                    "regions_detected": regions}
            dst.with_suffix(".error.txt").unlink(missing_ok=True)
            write_json(dst, data)
            status = f"{len(blocks)} blocks, {len({b['article_region'] for b in blocks})} articles"
        except Exception:
            errors += 1
            dst.with_suffix(".error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            status = "ERROR (see .error.txt)"
        left = (time.time() - t0) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] {r['page_id']} {r['newspaper']} p{r['page']}: {status} "
              f"({time.time() - t:.0f}s, ~{left / 60:.0f} min left)", flush=True)
    print(f"done, {errors} errors -> {out_dir / 'raw'}")


if __name__ == "__main__":
    main()
