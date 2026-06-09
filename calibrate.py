"""
calibrate.py
===========
Click-to-calibrate the draft slot boxes for YOUR screen.

Why: the fractional defaults can't know where Scrcpy put the mirror or how it
scaled your 2712x1220 phone onto your (e.g.) 1366x768 PC.  This grabs your
screen, lets you click a few anchor points on the live draft, computes all 20
boxes and writes ``layout.json`` - which main.py then loads for pixel-perfect
placement at your real resolution.

It works in your DESKTOP coordinates (grabs the whole monitor), so it doesn't
matter where the Scrcpy window sits, as long as you don't move it afterwards.

Run
---
    1. Open Scrcpy and bring the MLBB draft on screen.
    2. python calibrate.py
    3. Click the 8 points it asks for (first & last slot of each group).
    4. Tune box size with '[' / ']' if needed, then press 's' to save.

Keys in preview:  s=save   r=restart   [=smaller  ]=bigger   q=quit
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import config
from config import Box, Layout

try:
    import numpy as np
    import cv2
    import mss
except Exception:                                   # pragma: no cover
    np = cv2 = mss = None

Point = Tuple[int, int]

# The 8 anchor prompts, in click order.
PROMPTS = [
    "ALLY PICK  #1  (top slot)    - click CENTER",
    "ALLY PICK  #5  (bottom slot) - click CENTER",
    "ENEMY PICK #1  (top slot)    - click CENTER",
    "ENEMY PICK #5  (bottom slot) - click CENTER",
    "ALLY BAN   #1  (left-most)   - click CENTER",
    "ALLY BAN   #5  (right-most)  - click CENTER",
    "ENEMY BAN  #1  (left-most)   - click CENTER",
    "ENEMY BAN  #5  (right-most)  - click CENTER",
]


# ---------------------------------------------------------------------------
#  PURE GEOMETRY  (unit-testable, no GUI)
# ---------------------------------------------------------------------------
def _interp(first: Point, last: Point, n: int = 5) -> List[Point]:
    (x0, y0), (x1, y1) = first, last
    return [(round(x0 + (x1 - x0) * i / (n - 1)),
             round(y0 + (y1 - y0) * i / (n - 1))) for i in range(n)]


def _pitch(first: Point, last: Point, n: int = 5) -> float:
    (x0, y0), (x1, y1) = first, last
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (n - 1)


def _boxes(first: Point, last: Point, factor: float) -> List[Box]:
    centers = _interp(first, last)
    side = max(8, round(_pitch(first, last) * factor))
    return [Box(cx - side // 2, cy - side // 2, side, side) for cx, cy in centers]


def layout_from_anchors(ally_pick: Tuple[Point, Point],
                        enemy_pick: Tuple[Point, Point],
                        ally_ban: Tuple[Point, Point],
                        enemy_ban: Tuple[Point, Point],
                        pick_factor: float = 0.82,
                        ban_factor: float = 0.82) -> Layout:
    """Build the full 20-box Layout from the first/last slot of each group.
    Box size is derived from the slot pitch (so you only click 2 points/group)."""
    return Layout(
        ally_picks=_boxes(*ally_pick, factor=pick_factor),
        enemy_picks=_boxes(*enemy_pick, factor=pick_factor),
        ally_bans=_boxes(*ally_ban, factor=ban_factor),
        enemy_bans=_boxes(*enemy_ban, factor=ban_factor),
    )


# ---------------------------------------------------------------------------
#  INTERACTIVE GUI
# ---------------------------------------------------------------------------
def _require_gui() -> None:
    if np is None or cv2 is None or mss is None:
        raise RuntimeError("calibrate needs numpy + opencv-python + mss: "
                           "pip install numpy opencv-python mss")


def _grab_monitor(index: int):
    factory = getattr(mss, "MSS", None) or getattr(mss, "mss")
    with factory() as sct:
        mon = sct.monitors[index]
        raw = np.asarray(sct.grab(mon))[:, :, :3]
        return np.ascontiguousarray(raw), mon


def _check_gui() -> None:
    """opencv-python-headless has no imshow; fail with a clear message."""
    try:
        cv2.namedWindow("__probe__")
        cv2.destroyWindow("__probe__")
    except cv2.error:
        raise RuntimeError(
            "Your OpenCV has no GUI (opencv-python-headless is installed).\n"
            "Either:  pip uninstall opencv-python-headless && pip install opencv-python\n"
            "Or just use the built-in calibrator:  python main.py --calibrate")


def run_gui(out_path: str, monitor: int = 1,
            max_w: int = 1280, max_h: int = 720) -> int:
    _require_gui()
    config.enable_dpi_awareness()
    _check_gui()
    frame, mon = _grab_monitor(monitor)
    H, W = frame.shape[:2]
    scale = min(max_w / W, max_h / H, 1.0)
    disp_base = cv2.resize(frame, (int(W * scale), int(H * scale)))

    clicks: List[Point] = []                        # stored in monitor-local px

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(PROMPTS):
            clicks.append((round(x / scale), round(y / scale)))

    win = "MLBB draft calibration"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    pick_factor = ban_factor = 0.82

    def render():
        img = disp_base.copy()
        # already-placed clicks
        for i, (mx, my) in enumerate(clicks):
            cv2.circle(img, (int(mx * scale), int(my * scale)), 5, (0, 255, 255), -1)
            cv2.putText(img, str(i + 1), (int(mx * scale) + 6, int(my * scale) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if len(clicks) < len(PROMPTS):
            msg = f"[{len(clicks)+1}/8] {PROMPTS[len(clicks)]}"
            cv2.putText(img, msg, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (60, 255, 170), 2, cv2.LINE_AA)
        else:
            layout = _current_layout()
            _draw_layout(img, layout, scale)
            cv2.putText(img, "s=save  r=restart  [ ]=size  q=quit",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (60, 255, 170), 2, cv2.LINE_AA)
        return img

    def _current_layout() -> Layout:
        return layout_from_anchors(
            (clicks[0], clicks[1]), (clicks[2], clicks[3]),
            (clicks[4], clicks[5]), (clicks[6], clicks[7]),
            pick_factor=pick_factor, ban_factor=ban_factor)

    while True:
        cv2.imshow(win, render())
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return 1
        if key == ord("r"):
            clicks.clear()
        if len(clicks) == len(PROMPTS):
            if key == ord("["):
                pick_factor = ban_factor = max(0.4, pick_factor - 0.04)
            elif key == ord("]"):
                pick_factor = ban_factor = min(1.2, pick_factor + 0.04)
            elif key == ord("s"):
                config.save_layout(_current_layout(), out_path)
                print(f"[calibrate] saved {out_path} "
                      f"(monitor {mon['width']}x{mon['height']})")
                cv2.destroyAllWindows()
                return 0


def _draw_layout(img, layout: Layout, scale: float) -> None:
    groups = ((layout.ally_picks, (60, 230, 120)),
              (layout.enemy_picks, (70, 70, 235)),
              (layout.ally_bans, (160, 160, 160)),
              (layout.enemy_bans, (160, 160, 160)))
    for boxes, color in groups:
        for b in boxes:
            cv2.rectangle(img, (int(b.x * scale), int(b.y * scale)),
                          (int(b.x2 * scale), int(b.y2 * scale)), color, 2)


def main() -> int:
    p = argparse.ArgumentParser(description="Click-to-calibrate draft boxes.")
    p.add_argument("--out", default=config.LAYOUT_FILE)
    p.add_argument("--monitor", type=int, default=1, help="mss monitor index")
    args = p.parse_args()
    try:
        return run_gui(args.out, args.monitor)
    except Exception as exc:
        sys.stderr.write(f"[fatal] {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
