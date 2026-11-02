"""Stage 5 - pictures to check the result by eye: output/runs/<run_name>/check/<date>/<newspaper>/<page_id>.jpg
Each article gets its own colour and a big number; each text group a thin box with "article.group category".
Skipped blocks (images, page headers, ...) are grey. Needs stage 4 (the day tables) first.
    python s5_check.py
    python s5_check.py --set select.newspapers=[451]
"""

from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont, ImageOps

from common import load_settings, path_setting, read_csv, read_pages, run_dir, select_pages, stage_args

COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 190, 190),
          (240, 50, 230), (128, 128, 0), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128)]


def font(size):
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    s = load_settings(stage_args(__doc__).parse_args())
    out_dir = run_dir(s)
    root = path_setting(s["input_root"])
    groups, arts, skipped = defaultdict(list), defaultdict(dict), defaultdict(list)
    for r in (row for day in sorted((out_dir / "days").glob("*.csv")) for row in read_csv(day)):  # works while all_data.csv is open in Excel
        key = r["page_id"]
        if r["row_type"] == "skipped_block":
            skipped[key].append(r)
        elif r["row_type"] == "text_group":
            groups[key].append(r)
            box = [int(r[k]) for k in ("x1", "y1", "x2", "y2")]
            a = arts[key].setdefault(r["article"], box)  # article box = all its groups together
            arts[key][r["article"]] = [min(a[0], box[0]), min(a[1], box[1]), max(a[2], box[2]), max(a[3], box[3])]
    pages = [p for p in select_pages(read_pages(), s) if p["page_id"] in groups]
    print(f"{len(pages)} pages with results -> {out_dir / 'check'}")
    for n, p in enumerate(pages, 1):
        key = p["page_id"]
        dst = out_dir / "check" / p["date"] / key.split("-")[1] / f"{key}.jpg"
        if dst.exists() and not s["redo"]:
            continue
        img = ImageOps.exif_transpose(Image.open(root / p["file"])).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")
        lw = max(3, img.width // 500)
        small, big = font(5 * lw), font(18 * lw)
        for b in skipped[key]:
            d.rectangle([int(b[k]) for k in ("x1", "y1", "x2", "y2")], fill=(128, 128, 128, 60), outline=(128, 128, 128), width=lw)
        for art, box in arts[key].items():
            col = COLORS[(int(art) - 1) % len(COLORS)]
            d.rectangle(box, fill=col + (35,), outline=col, width=2 * lw)
            d.text((box[0] + 3 * lw, box[1] + 2 * lw), art, fill=col + (230,), font=big)
        for g in groups[key]:
            col = COLORS[(int(g["article"]) - 1) % len(COLORS)]
            box = [int(g[k]) for k in ("x1", "y1", "x2", "y2")]
            d.rectangle(box, outline=col, width=lw)
            tag = f"{g['article']}.{g['group']} {g['category']}"
            pos = (box[0] + lw, box[1] + lw)
            tb = d.textbbox(pos, tag, font=small)
            d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=col)
            d.text(pos, tag, fill="white", font=small)
        img.thumbnail((2000, 2000))
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=85)
        print(f"  [{n}/{len(pages)}] {dst.relative_to(out_dir)}", end="\r", flush=True)
    print()


if __name__ == "__main__":
    main()
