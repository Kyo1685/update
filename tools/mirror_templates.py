"""
tools/mirror_templates.py
=========================
Mirror every ally-side crop into a left-facing enemy-side crop.

The ally avatar faces one way; the enemy side is its horizontal mirror. This
flips each image in ``templates_ally/`` left<->right and writes it to
``templates_enemy/`` (same filename), so the enemy library matches the enemy
portraits directly (the enemy library applies no auto-flip). Alpha is preserved.

    python tools/mirror_templates.py                  # templates_ally -> templates_enemy
    python tools/mirror_templates.py --only helcurt   # just one hero
    python tools/mirror_templates.py --src A --dst B   # any pair of folders

Uses opencv (already a project dependency) - no extra install needed.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None

import config


def main() -> int:
    p = argparse.ArgumentParser(description="Mirror ally crops into enemy crops.")
    p.add_argument("--src", default=config.TEMPLATE_ALLY_DIR,
                   help="folder of ally-side crops (default: templates_ally)")
    p.add_argument("--dst", default=config.TEMPLATE_ENEMY_DIR,
                   help="folder to write enemy-side crops (default: templates_enemy)")
    p.add_argument("--only", default=None,
                   help="only this filename stem, e.g. 'helcurt'")
    args = p.parse_args()
    if cv2 is None:
        sys.stderr.write("needs opencv-python (pip install opencv-python)\n")
        return 2

    os.makedirs(args.dst, exist_ok=True)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    files = sorted(f for e in exts for f in glob.glob(os.path.join(args.src, e)))
    written = 0
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        if args.only and stem.lower() != args.only.lower():
            continue
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)        # keep alpha if present
        if img is None:
            continue
        out = os.path.join(args.dst, os.path.basename(fp))
        cv2.imwrite(out, cv2.flip(img, 1))                # 1 = horizontal mirror
        print(f"mirrored {fp}  ->  {out}")
        written += 1
    print(f"done: {written} file(s) mirrored {args.src} -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
