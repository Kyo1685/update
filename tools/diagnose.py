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

    # Build the SAME ally / enemy / ban libraries the live app uses, so this
    # diagnostic reflects exactly what the overlay decides (ally picks =
    # circular un-flipped + overrides, enemy picks = square + overrides, bans =
    # circular un-flipped).
    ally_flip = not config.PACK_FACES_ALLY
    square = TemplateLibrary.from_dir(config.TEMPLATE_DIR)
    circle = TemplateLibrary.from_dir(config.TEMPLATE_CIRCLE_DIR,
                                      circular=True, flip=ally_flip)
    a_ovr = TemplateLibrary.from_dir(config.TEMPLATE_ALLY_DIR, circular=True)
    e_ovr = TemplateLibrary.from_dir(config.TEMPLATE_ENEMY_DIR)
    n_a = circle.overlay(a_ovr) if len(a_ovr) else 0
    n_e = square.overlay(e_ovr) if len(e_ovr) else 0
    # Highest-priority learned memory (same as main.start_live).
    la = TemplateLibrary.from_dir(config.TEMPLATE_LEARNED_DIR, circular=True)
    le = TemplateLibrary.from_dir(config.TEMPLATE_LEARNED_ENEMY_DIR)
    n_a += circle.overlay(la, learned=True) if len(la) else 0
    n_e += square.overlay(le, learned=True) if len(le) else 0
    print(f"region {region['width']}x{region['height']} | "
          f"ally/ban=circular:{len(circle)}(+{n_a} ovr)  "
          f"enemy=square:{len(square)}(+{n_e} ovr)  "
          f"PACK_FACES_ALLY={config.PACK_FACES_ALLY} histogram_fallback="
          f"{config.USE_HISTOGRAM_FALLBACK}\n")
    os.makedirs(args.out, exist_ok=True)
    det = DraftDetector(HeroDB.load("heroes.json"))

    def dump(title, boxes, lib, thr, picks=False):
        print(f"== {title} (threshold {thr}) ==")
        crops = [det._crop(frame, b) for b in boxes]
        live = [c for c in crops if not det._is_empty(c)]
        sref = max((det._mean_saturation(c) for c in live), default=0.0)
        vref = max((det._mean_value(c) for c in live), default=0.0)
        for i, (b, crop) in enumerate(zip(boxes, crops)):
            cv2.imwrite(os.path.join(args.out, f"{title}_{i}.png"), crop)
            sat = det._mean_saturation(crop)
            val = det._mean_value(crop)
            empty = det._is_empty(crop)
            pending = bool(picks and not empty and sref > 0 and vref > 0
                           and config.LOCKED_REL_SATURATION > 0
                           and config.LOCKED_REL_VALUE > 0
                           and sat < config.LOCKED_REL_SATURATION * sref
                           and val < config.LOCKED_REL_VALUE * vref)
            if empty:
                print(f"  [{i}] sat={sat:5.1f} val={val:5.1f}  (empty)")
                continue
            if pending:
                print(f"  [{i}] sat={sat:5.1f} val={val:5.1f}  NOT PICKED (grayed)")
                continue
            tops = lib.top_matches(crop, 3)
            # The ACTUAL decision the overlay would make (incl. histogram fallback).
            name, score, method = lib.match(crop, thr)
            decided = f"{name}:{score:.2f}({method})" if name else "(no match)"
            top3 = ", ".join(f"{n}:{s:.2f}" for n, s in tops)
            print(f"  [{i}] sat={sat:5.1f} val={val:5.1f}  ->{decided:24} | top3: {top3}")
        print()

    dump("ally_pick", L.ally_picks, circle, config.TEMPLATE_MATCH_THRESHOLD, picks=True)
    dump("enemy_pick", L.enemy_picks, square, config.ENEMY_MATCH_THRESHOLD, picks=True)
    dump("ally_ban", L.ally_bans, circle, config.BAN_MATCH_THRESHOLD)
    dump("enemy_ban", L.enemy_bans, circle, config.BAN_MATCH_THRESHOLD)
    print(f"slot crops saved to ./{args.out}/  - check a few for alignment, "
          f"and look for '(histogram)' decisions where a template should win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
