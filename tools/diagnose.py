"""
tools/diagnose.py
================
Dump what the detector actually "sees" so detection can be tuned from real
numbers instead of guesswork.

For every slot it prints the top-3 template matches + scores (picks vs the
square library, bans vs the circular library) and saves each slot crop to
``diag/`` so you can eyeball alignment.

Run while the draft is on screen:

    python tools/diagnose.py                 # whole primary monitor
    python tools/diagnose.py --region 0,0,1366,614
    python tools/diagnose.py --image shot.png   # diagnose a saved screenshot

Send me the printout (and a couple of the diag/*.png crops) and I'll set the
thresholds / templates precisely.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from engine import HeroDB
from detector import ScreenCapturer, TemplateLibrary, DraftDetector

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = np = None


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose draft detection.")
    p.add_argument("--region", default=None, help="L,T,W,H (default: primary monitor)")
    p.add_argument("--image", default=None, help="diagnose a saved screenshot instead")
    p.add_argument("--layout", default=config.LAYOUT_FILE)
    p.add_argument("--out", default="diag")
    args = p.parse_args()
    if cv2 is None:
        sys.stderr.write("needs numpy + opencv-python\n"); return 2

    # frame + region/layout
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.stderr.write(f"cannot read {args.image}\n"); return 2
        h, w = frame.shape[:2]
        region = {"left": 0, "top": 0, "width": w, "height": h}
    else:
        if args.region:
            l, t, w, h = (int(v) for v in args.region.split(","))
            region = {"left": l, "top": t, "width": w, "height": h}
        else:
            region = ScreenCapturer.primary_monitor()
        frame = ScreenCapturer(region).grab(force=True)

    layout = config.load_layout(args.layout) if os.path.exists(args.layout) else None
    config.apply_region(region, layout)
    L = config.LAYOUT

    square = TemplateLibrary.from_dir(config.TEMPLATE_DIR)
    circle = TemplateLibrary.from_dir(config.TEMPLATE_CIRCLE_DIR, circular=True)
    print(f"region {region['width']}x{region['height']} | "
          f"{len(square)} square + {len(circle)} circular templates\n")
    os.makedirs(args.out, exist_ok=True)
    det = DraftDetector(HeroDB.load("heroes.json"))

    def dump(title, boxes, lib, thr):
        print(f"== {title} (threshold {thr}) ==")
        for i, b in enumerate(boxes):
            crop = det._crop(frame, b)
            cv2.imwrite(os.path.join(args.out, f"{title}_{i}.png"), crop)
            sat = float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
            empty = det._is_empty(crop)
            tops = lib.top_matches(crop, 3) if not empty else []
            shown = ", ".join(f"{n}:{s:.2f}" for n, s in tops) or "(empty)"
            flag = "" if (tops and tops[0][1] >= thr) else "  <-- below threshold"
            print(f"  [{i}] sat={sat:5.1f}  {shown}{flag}")
        print()

    dump("ally_pick", L.ally_picks, square, config.TEMPLATE_MATCH_THRESHOLD)
    dump("enemy_pick", L.enemy_picks, square, config.ENEMY_MATCH_THRESHOLD)
    dump("ally_ban", L.ally_bans, circle, config.BAN_MATCH_THRESHOLD)
    dump("enemy_ban", L.enemy_bans, circle, config.BAN_MATCH_THRESHOLD)
    print(f"slot crops saved to ./{args.out}/  - check a few for alignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
