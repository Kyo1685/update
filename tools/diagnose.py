"""
tools/diagnose.py
================
Dump what the detector actually "sees" so detection can be tuned from real
numbers instead of guesswork.

For every slot it prints the decision the live overlay makes (DINOv2 + the
registered-art double-check when the AI stack is installed, else templates)
with its evidence, next to the old template matcher's decision and top-3, and
saves each slot crop to ``diag/`` so you can eyeball alignment.

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
    p.add_argument("--no-dino", action="store_true",
                   help="template matching only (skip the DINOv2 decision)")
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
    from icon_match import IconMatcher
    icons = IconMatcher.from_dirs()
    rec = None
    if config.USE_DINO and not args.no_dino:
        from recognizer import DinoRecognizer
        rec = DinoRecognizer.load(icons=icons)
    print(f"engine: {'dino (' + rec.label + ') + registered art' if rec else 'templates + registered art (bans)'}\n")

    # The decision the live overlay makes, with its evidence ([why]).
    det = DraftDetector(HeroDB.load("heroes.json"), ally_library=circle,
                        enemy_library=square, ban_library=circle,
                        recognizer=rec, icons=icons)
    state = det.detect(frame)

    def dump(title, prefix, boxes, names, pending, lib, thr):
        print(f"== {title} ==")
        for i, b in enumerate(boxes):
            crop = det._crop(frame, b)
            cv2.imwrite(os.path.join(args.out, f"{title}_{i}.png"), crop)
            sat, val = det._mean_saturation(crop), det._mean_value(crop)
            if pending and pending[i]:
                decided = "NOT PICKED (hover)"
            else:
                decided = names[i] or "-"
            why = det._detail.get(f"{prefix}{i}", "")
            head = f"  [{i}] sat={sat:5.1f} val={val:5.1f}  "
            print(f"{head}overlay  -> {decided:20} [{why}]")
            if det._is_empty(crop):
                continue
            # The old template matcher, for comparison.
            tops = lib.top_matches(crop, 3)
            name, score, method = lib.match(crop, thr)
            old = f"{name}:{score:.2f}({method})" if name and method != "low" else "-"
            print(f"{' ' * len(head)}template -> {old:20} top3: "
                  + ", ".join(f"{n}:{s:.2f}" for n, s in tops))
        print()

    dump("ally_pick", "ap", L.ally_picks, state.ally_picks, state.ally_pending,
         circle, config.TEMPLATE_MATCH_THRESHOLD)
    dump("enemy_pick", "ep", L.enemy_picks, state.enemy_picks, state.enemy_pending,
         square, config.ENEMY_MATCH_THRESHOLD)
    dump("ally_ban", "ab", L.ally_bans, state.ally_bans, None,
         circle, config.BAN_MATCH_THRESHOLD)
    dump("enemy_ban", "eb", L.enemy_bans, state.enemy_bans, None,
         circle, config.BAN_MATCH_THRESHOLD)
    print(f"slot crops saved to {os.path.join(args.out, '')}  - check a few for alignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
