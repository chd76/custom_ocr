"""Page filter labeler (keep / mixed / trash): one key press per page.

Press the number of the page's category (see the list on the right). It is saved and the next
page opens. In review mode (label.py --review) the model's suggestion is shown; press its number
(or D) to accept it, or another number to correct it.

Keys:
    1-9 ......... set category and go to the next page
    D / Right ... next page (in review mode: accept the suggestion)     A / Left ... previous page
    N ........... next page without a category      Delete ... remove this page's category
    Mouse wheel = zoom, Space + drag or middle-drag = move around, F = whole page    H = help
"""

import argparse
import math
import sys
import tkinter as tk

from PIL import Image, ImageOps, ImageTk

from common import list_images, load_config, read_labels, read_pending, write_labels, write_pending

PALETTE = ["#ff3b3b", "#3b9bff", "#22c55e", "#f59e0b", "#c084fc", "#f472b6", "#2dd4bf", "#eab308", "#a3a3a3"]


class PageCategoryLabeler:
    def __init__(self, root, cfg, images, review=False, start=None):
        self.root, self.cfg, self.images, self.review = root, cfg, images, review
        self.cats = cfg["categories"]
        self.labels = read_labels(cfg)
        self.pending = read_pending(cfg)
        self.idx = 0
        self.img = self.tk_img = None
        self.zoom, self.scale, self.offset = 1.0, 1.0, (0, 0)
        self._key, self._pos, self._pan, self.space_down = None, (0, 0), None, False
        self.show_help = False
        self.note = ""

        root.title("Page Filter" + (" - REVIEW" if review else ""))
        root.geometry("1250x950")
        root.configure(bg="#1e1e1e")
        side = tk.Frame(root, bg="#252526", width=240)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(side, text="Page category", fg="#ddd", bg="#252526", font=("Segoe UI", 12, "bold")).pack(pady=(12, 6))
        self.rows = []
        for i, c in enumerate(self.cats[:9]):
            lab = tk.Label(side, text=f"{i + 1}  {c}", anchor="w", fg=PALETTE[i % 9], bg="#252526",
                           font=("Segoe UI", 12), padx=12)
            lab.pack(fill=tk.X, pady=1)
            self.rows.append(lab)
        self.counts = tk.Label(side, fg="#999", bg="#252526", font=("Segoe UI", 9), justify=tk.LEFT, anchor="nw")
        self.counts.pack(fill=tk.X, padx=12, pady=14)
        self.status = tk.Label(root, anchor="w", fg="#ddd", bg="#0e5a7a", font=("Segoe UI", 10), padx=8)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas = tk.Canvas(root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        cv = self.canvas
        cv.bind("<Configure>", lambda e: self.render())
        cv.bind("<MouseWheel>", self.on_wheel)
        cv.bind("<ButtonPress-1>", lambda e: self.space_down and self.pan_start(e))
        cv.bind("<B1-Motion>", self.pan_move)
        cv.bind("<ButtonRelease-1>", self.pan_end)
        cv.bind("<ButtonPress-2>", self.pan_start)
        cv.bind("<B2-Motion>", self.pan_move)
        cv.bind("<ButtonRelease-2>", self.pan_end)
        for k in range(1, min(9, len(self.cats)) + 1):
            root.bind(str(k), lambda e, k=k: self.set_category(self.cats[k - 1]))
        for seq in ("<d>", "<D>", "<Right>"):
            root.bind(seq, lambda e: self.accept_or_next())
        for seq in ("<a>", "<A>", "<Left>"):
            root.bind(seq, lambda e: self.go(-1))
        root.bind("<n>", lambda e: self.next_todo())
        root.bind("<N>", lambda e: self.next_todo())
        root.bind("<Delete>", lambda e: self.remove())
        root.bind("<h>", lambda e: self.toggle_help())
        root.bind("<KeyPress-space>", lambda e: setattr(self, "space_down", True))
        root.bind("<KeyRelease-space>", lambda e: setattr(self, "space_down", False))
        for seq in ("<f>", "<F>", "0"):
            root.bind(seq, lambda e: self.reset_zoom())

        stems = [p.stem for p in images]
        first = stems.index(start) if start in stems else (0 if review else self._next_todo(-1))
        self.load(first or 0)

    # ---------------------------------------------------------------- pages
    def _next_todo(self, after):
        n = len(self.images)
        return next(((after + k) % n for k in range(1, n + 1) if self.images[(after + k) % n].stem not in self.labels), None)

    def load(self, i):
        self.idx = i
        self.img = ImageOps.exif_transpose(Image.open(self.images[i])).convert("RGB")
        self.zoom, self.offset, self._key = 1.0, (0, 0), None
        self.render()

    def go(self, step):
        j = self.idx + step
        if 0 <= j < len(self.images):
            self.load(j)
        else:
            self.note = "this is the " + ("last" if step > 0 else "first") + " page"
            self.render()

    def next_todo(self):
        j = self._next_todo(self.idx)
        if j is None:
            self.note = "every page has a category"
            self.render()
        else:
            self.load(j)

    def save(self):
        write_labels(self.cfg, self.labels)
        write_pending(self.cfg, self.pending)

    def set_category(self, cat):
        stem = self.images[self.idx].stem
        had_category = stem in self.labels
        self.labels[stem] = cat
        self.pending.pop(stem, None)  # a category you pressed yourself counts as reviewed
        self.save()
        if self.review or had_category or self.idx + 1 >= len(self.images):
            self.go(1)  # correcting a page (or reviewing): just the next page, no jump
        else:
            j = self._next_todo(self.idx)  # a new page: on to the next page without a category
            self.load(j) if j is not None else self.go(1)

    def accept_or_next(self):
        stem = self.images[self.idx].stem
        if self.review and stem in self.pending and stem in self.labels:
            self.pending.pop(stem)  # D in review mode = the suggestion is right
            self.save()
        self.go(1)

    def remove(self):
        stem = self.images[self.idx].stem
        self.labels.pop(stem, None)
        self.pending.pop(stem, None)
        self.save()
        self.note = "category removed"
        self.render()

    def toggle_help(self):
        self.show_help = not self.show_help
        self.render()

    # ---------------------------------------------------------------- zoom
    def on_wheel(self, e):
        old = self.zoom
        self.zoom = min(20.0, max(1.0, self.zoom * (1.25 if e.delta > 0 else 0.8)))
        if self.zoom != old:
            px, py = (e.x - self.offset[0]) / self.scale, (e.y - self.offset[1]) / self.scale
            ns = self.scale * self.zoom / old
            self.offset = (e.x - px * ns, e.y - py * ns)
            self.render()

    def pan_start(self, e):
        self._pan = (e.x, e.y, *self.offset)

    def pan_move(self, e):
        if self._pan:
            sx, sy, ox, oy = self._pan
            self.offset = (ox + e.x - sx, oy + e.y - sy)
            self.render()

    def pan_end(self, e):
        self._pan = None

    def reset_zoom(self):
        self.zoom, self.offset = 1.0, (0, 0)
        self.render()

    # ---------------------------------------------------------------- drawing
    def render(self):
        if self.img is None:
            return
        cv = self.canvas
        cw, ch = max(cv.winfo_width(), 50), max(cv.winfo_height(), 50)
        w, h = self.img.size
        self.scale = min(cw / w, ch / h) * self.zoom
        iw, ih = w * self.scale, h * self.scale
        ox, oy = self.offset
        ox = (cw - iw) / 2 if iw <= cw else min(0.0, max(cw - iw, ox))
        oy = (ch - ih) / 2 if ih <= ch else min(0.0, max(ch - ih, oy))
        self.offset = (ox, oy)
        key = (self.idx, round(self.scale, 6), round(ox, 1), round(oy, 1), cw, ch)
        if self._key != key:  # only the visible part, from the full-resolution page
            x0, y0 = max(0, int(-ox / self.scale)), max(0, int(-oy / self.scale))
            x1, y1 = min(w, math.ceil((cw - ox) / self.scale)), min(h, math.ceil((ch - oy) / self.scale))
            size = (max(1, round((x1 - x0) * self.scale)), max(1, round((y1 - y0) * self.scale)))
            self.tk_img = ImageTk.PhotoImage(self.img.crop((x0, y0, x1, y1)).resize(size, Image.BILINEAR))
            self._pos, self._key = (x0 * self.scale + ox, y0 * self.scale + oy), key
        cv.delete("all")
        cv.create_image(*self._pos, image=self.tk_img, anchor="nw")
        stem = self.images[self.idx].stem
        cur = self.labels.get(stem)
        suggested = stem in self.pending
        for i, lab in enumerate(self.rows):
            on = self.cats[i] == cur
            lab.config(bg="#0e5a7a" if on else "#252526", font=("Segoe UI", 12, "bold" if on else "normal"))
        if cur:  # big tag on the page
            text = f"{cur}" + (f"  (model {self.pending[stem]:.0%} - check it)" if suggested and self.pending[stem] is not None
                               else "  (model suggestion - check it)" if suggested else "")
            t = cv.create_text(14, 14, anchor="nw", text=text, fill="#000", font=("Segoe UI", 14, "bold"))
            x1, y1, x2, y2 = cv.bbox(t)
            col = PALETTE[self.cats.index(cur) % 9] if cur in self.cats else "#a3a3a3"
            cv.tag_lower(cv.create_rectangle(x1 - 8, y1 - 6, x2 + 8, y2 + 6, fill=col, outline=""), t)
        if self.show_help:
            t = cv.create_text(24, 70, anchor="nw", fill="#eee", font=("Consolas", 10),
                               text="Keys:\n" + __doc__.split("Keys:")[1].strip())
            x1, y1, x2, y2 = cv.bbox(t)
            cv.tag_lower(cv.create_rectangle(x1 - 12, y1 - 12, x2 + 12, y2 + 12, fill="#111", outline="#555"), t)
        counts = {c: 0 for c in self.cats}
        for p in self.images:
            c = self.labels.get(p.stem)
            if c in counts:
                counts[c] += 1
        self.counts.config(text="pages per category:\n" + "\n".join(f"  {c}: {n}" for c, n in counts.items()))
        done = sum(1 for p in self.images if p.stem in self.labels and p.stem not in self.pending)
        msg = (f"[{self.idx + 1}/{len(self.images)}]  {self.images[self.idx].name}   category: {cur or '-'}"
               f"   checked pages: {done}/{len(self.images)}"
               + (f"   to review: {len(self.pending)}" if self.pending else "")
               + (f"   zoom {self.zoom * 100:.0f}%" if self.zoom > 1 else "")
               + (f"   â€” {self.note}" if self.note else ""))
        self.status.config(text=msg)
        self.note = ""


def main():
    ap = argparse.ArgumentParser(description="Give every page a category")
    ap.add_argument("--review", action="store_true", help="only pages the model categorized, to check them")
    ap.add_argument("--labeled", action="store_true", help="only pages that already have a category")
    ap.add_argument("--start", help="open this page first (file name without extension)")
    args = ap.parse_args()
    cfg = load_config()
    images = list_images(cfg["images"])
    if args.review:
        pending = read_pending(cfg)
        images = [p for p in images if p.stem in pending]
        if not images:
            print("Nothing to review.")
            sys.exit(0)
        print(f"{len(images)} pages to review: press the right number, or D if the suggestion is right.")
    elif args.labeled:
        labels = read_labels(cfg)
        images = [p for p in images if p.stem in labels]
        if not images:
            print("No categorized pages yet.")
            sys.exit(0)
        args.start = args.start or images[0].stem
    if not images:
        print(f"No images in {cfg['images']}")
        sys.exit(1)
    root = tk.Tk()
    PageCategoryLabeler(root, cfg, images, review=args.review, start=args.start)
    root.mainloop()


if __name__ == "__main__":
    main()
