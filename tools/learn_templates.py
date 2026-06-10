"""
tools/learn_templates.py
========================
Bootstrap hero templates from YOUR live draft.

The bundled portraits often don't closely match your screen, so detection
falls back to fuzzy COLOUR matching and confuses same-coloured look-alikes
(e.g. Helcurt -> Gord, both dark).  This grabs the current draft and saves each
slot's crop AS that hero's template, so it then self-matches at ~1.0 and the
colour guessing no longer matters.

Tell it who is in each slot, left->right / top->bottom; leave a slot BLANK to
skip it (so you can fix just the wrong ones):

    # fix everything this draft:
    python tools/learn_templates.py ^
        --ally  nana,melissa,lukas,,helcurt ^
        --enemy vexana,johnson,layla ^
        --ally-bans  sora,mathilda,zhuxin,hilda,minsithar ^
        --enemy-bans chou,selena,saber,hayabusa,miya

    # or fix ONLY the broken ones (Helcurt in ally slot 5, Sora in ally ban 1):
    python tools/learn_templates.py --ally ,,,,helcurt --ally-bans sora

Ally picks + ALL bans -> templates_ally/<hero>.png   (the circular library)
Enemy picks           -> templates_enemy/<hero>.png  (the square splash library)

Then re-run `python tools/diagnose.py`: those slots now score ~1.0 by template.
Uses your layout.json automatically, so calibrate first (main.py --calibrate).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None


def _names(s):
    return [x.strip() for x in s.split(",")] if s else []


def main() -> int:
    p = argparse.ArgumentParser(description="Save live draft crops as hero templates.")
    p.add_argument("--ally", default="", help="ally pick heroes, top->bottom, comma-sep")
    p.add_argument("--enemy", default="", help="enemy pick heroes, top->bottom")
    p.add_argument("--ally-bans", default="", help="ally ban heroes, left->right")
    p.add_argument("--enemy-bans", default="", help="enemy ban heroes, left->right")
    p.add_argument("--region", default=None, help="L,T,W,H (default: primary monitor)")
    p.add_argument("--image", default=None, help="learn from a saved screenshot instead")
    p.add_argument("--layout", default=config.LAYOUT_FILE)
    args = p.parse_args()
    if cv2 is None:
        sys.stderr.write("needs opencv-python (pip install opencv-python)\n")
        return 2

    # Same capture + layout path the detector/diagnose use, so crops line up.
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.stderr.write(f"cannot read {args.image}\n"); return 2
        h, w = frame.shape[:2]
        region = {"left": 0, "top": 0, "width": w, "height": h}
    else:
        from detector import ScreenCapturer
        if args.region:
            l, t, w, h = (int(v) for v in args.region.split(","))
            region = {"left": l, "top": t, "width": w, "height": h}
        else:
            region = ScreenCapturer.primary_monitor()
        frame = ScreenCapturer(region).grab(force=True)

    layout = config.load_layout(args.layout) if os.path.exists(args.layout) else None
    if layout is None:
        sys.stderr.write("[warn] no layout.json - run 'python main.py --calibrate' "
                         "first or crops won't line up.\n")
    config.apply_region(region, layout)
    L = config.LAYOUT

    saved = 0

    def learn(boxes, names, folder):
        nonlocal saved
        if not names:
            return
        os.makedirs(folder, exist_ok=True)
        for box, name in zip(boxes, names):
            if not name:
                continue
            crop = frame[box.y:box.y2, box.x:box.x2]
            out = os.path.join(folder, f"{name.strip().lower()}.png")
            cv2.imwrite(out, crop)
            print(f"  saved {out}  ({crop.shape[1]}x{crop.shape[0]})")
            saved += 1

    # Ally picks + BOTH ban rows share the circular library (templates_ally
    # overrides feed it); enemy picks use the square splash library.
    learn(L.ally_picks, _names(args.ally), config.TEMPLATE_ALLY_DIR)
    learn(L.enemy_picks, _names(args.enemy), config.TEMPLATE_ENEMY_DIR)
    learn(L.ally_bans, _names(args.ally_bans), config.TEMPLATE_ALLY_DIR)
    learn(L.enemy_bans, _names(args.enemy_bans), config.TEMPLATE_ALLY_DIR)

    if saved:
        print(f"\nLearned {saved} template(s) from your draft. "
              f"Re-run:  python tools/diagnose.py  (those slots now score ~1.0)")
    else:
        print("Nothing saved - pass --ally/--enemy/--ally-bans/--enemy-bans names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
