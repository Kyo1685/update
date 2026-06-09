"""
tools/grab_templates.py
======================
Capture EXACT hero templates straight from YOUR draft screen, so the templates
match your client 1:1 (no mismatched / mirrored / outdated art).

It crops every calibrated slot from a live capture (or a saved screenshot) and
saves the crops to ``captured/``.  Rename the ones you want to the hero name and
move them in:

    picks  (ally avatars + enemy splash) ->  templates/<hero>.png
    bans   (circular icons)              ->  templates_circle/<hero>.png

Matching is mirror-invariant, so orientation doesn't matter.

Run during a draft:
    python tools/grab_templates.py
    python tools/grab_templates.py --image shot.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from detector import ScreenCapturer

try:
    import cv2
except Exception:
    cv2 = None


def _crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1 = max(0, box.x), max(0, box.y)
    x2, y2 = min(w, box.x2), min(h, box.y2)
    return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def main() -> int:
    p = argparse.ArgumentParser(description="Capture templates from your draft.")
    p.add_argument("--out", default="captured")
    p.add_argument("--region", default=None, help="L,T,W,H (default: primary monitor)")
    p.add_argument("--image", default=None, help="use a saved screenshot instead")
    p.add_argument("--layout", default=config.LAYOUT_FILE)
    args = p.parse_args()
    if cv2 is None:
        sys.stderr.write("needs opencv-python\n"); return 2

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
    os.makedirs(args.out, exist_ok=True)

    groups = (("ally_pick", L.ally_picks), ("enemy_pick", L.enemy_picks),
              ("ally_ban", L.ally_bans), ("enemy_ban", L.enemy_bans))
    n = 0
    for gname, boxes in groups:
        for i, b in enumerate(boxes):
            crop = _crop(frame, b)
            if crop is None or crop.size == 0:
                continue
            cv2.imwrite(os.path.join(args.out, f"{gname}_{i}.png"), crop)
            n += 1

    print(f"saved {n} slot crops to ./{args.out}/")
    print("Rename the ones you want to the hero name, then move them:")
    print("  ally/enemy picks ->  templates/<hero>.png")
    print("  bans (circular)  ->  templates_circle/<hero>.png")
    print("(matching is mirror-invariant, so flipped art is fine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
