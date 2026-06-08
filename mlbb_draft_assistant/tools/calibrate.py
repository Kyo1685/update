"""
Region calibration helper for the vision module.

The pixel rectangles in ``config.REGIONS`` must line up with the pick/ban slots
on *your* screen. This script helps you find them.

Modes
-----
    python tools/calibrate.py dump
        Capture the screen and write the full image plus each *current* slot
        crop to ./calibration_debug/ so you can see how far off the defaults are.

    python tools/calibrate.py select
        Open the captured screen and let you drag rectangles with the mouse
        (ENTER/SPACE to confirm each, ESC when done). The chosen
        [left, top, width, height] values are printed, ready to paste into
        ``config.REGIONS``.

Open the MLBB draft screen (or a screenshot of one) before running.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def dump() -> None:
    from src.vision import VisionModule

    vm = VisionModule()
    out = vm.debug_regions("calibration_debug")
    print(f"Wrote full screenshot + slot crops to: {out.resolve()}")
    print("Open the crops: each should contain exactly one portrait. "
          "If not, adjust config.REGIONS and re-run.")


def select() -> None:
    import cv2
    from src.vision import VisionModule

    vm = VisionModule()
    frame = vm.grab_monitor()
    print("Drag rectangles over the slots you care about.")
    print("ENTER/SPACE confirms a box, ESC finishes.")
    rois = cv2.selectROIs("Calibrate — drag boxes, ESC when done", frame,
                          showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    print("\nSelected regions as [left, top, width, height]:")
    for (x, y, w, h) in rois:
        print(f"    [{int(x)}, {int(y)}, {int(w)}, {int(h)}],")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
    if mode == "dump":
        dump()
    elif mode == "select":
        select()
    else:
        print(__doc__)
        sys.exit(1)
