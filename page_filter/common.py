"""Shared helpers for the page filter (keep / mixed / trash). Self-contained: imports nothing from the rest of the project."""

import csv
import os
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RUNS = HERE / "runs"       # trained models
WORK = HERE / "work"       # generated datasets
OUTPUT = HERE / "output"   # predictions for all pages
MODEL = RUNS / "page_filter" / "weights" / "best.pt"


def load_config():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    for key in ("images", "labels", "pending"):
        cfg[key] = (HERE / cfg[key]).resolve()
    cfg["categories"] = [str(c) for c in cfg["categories"]]
    cfg["skip"] = [str(c) for c in cfg.get("skip") or []]
    return cfg


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def page_info(stem):
    """File names look like '<page number>_<id>'."""
    head, _, tail = stem.partition("_")
    return (int(head), tail) if head.isdigit() else ("", "")


# ---------------------------------------------------------------- labels: page -> category
def read_labels(cfg):
    p = Path(cfg["labels"])
    if not p.exists():
        return {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        return {r["page"]: r["category"] for r in csv.DictReader(f) if r.get("page")}


def write_labels(cfg, labels):
    with Path(cfg["labels"]).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page", "category"])
        for page in sorted(labels):
            w.writerow([page, labels[page]])


# ---------------------------------------------------------------- review list: page -> model confidence
def read_pending(cfg):
    p = Path(cfg["pending"])
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        stem, _, conf = line.strip().partition(",")
        if stem:
            out[stem] = float(conf) if conf else None
    return out


def write_pending(cfg, pending):
    Path(cfg["pending"]).write_text("".join(f"{s},{'' if c is None else round(c, 3)}\n"
                                            for s, c in sorted(pending.items())), encoding="utf-8")


def link_or_copy(src, dst):
    """Hard-link (no extra disk space) when possible, otherwise copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
