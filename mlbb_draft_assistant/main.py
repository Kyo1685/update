"""
MLBB Draft Assistant — entry point
==================================

Wires the four modules together:

    Vision (scan screen) --> Scoring Engine (rank picks) --> Overlay (display)

A background ``DraftWorker`` thread scans + scores on an interval and pushes
results to the always-on-top overlay via a Qt signal, so the UI never blocks.

Usage
-----
    python main.py                 # live: capture screen + template match
    python main.py --mock          # screen-free demo with scripted draft data
    python main.py --rank all      # use a different rank's stats
    python main.py --interval 0.5  # scan twice a second

If the GUI / vision dependencies aren't installed the program prints how to fix
it instead of crashing.
"""

from __future__ import annotations

import argparse
import sys

import config
from src.data_manager import DataManager
from src.scoring_engine import ScoringEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLBB real-time draft assistant overlay")
    p.add_argument("--mock", action="store_true",
                   help="run without screen capture using scripted draft data")
    p.add_argument("--rank", default=config.DEFAULT_RANK, choices=config.RANK_TIERS,
                   help="rank tier whose stats to use")
    p.add_argument("--interval", type=float, default=config.SCAN_INTERVAL_S,
                   help="seconds between draft scans")
    p.add_argument("--lane-counter", action="store_true",
                   help="start with Lane Counter mode enabled")
    p.add_argument("--dark", action="store_true",
                   help="start with the joke Dark System enabled")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # GUI is required to actually run; import lazily for a friendly error.
    try:
        from PyQt5 import QtCore, QtWidgets
        from src.overlay import OverlayFlags, OverlayWindow
    except Exception as e:
        print("PyQt5 is required to run the overlay. Install dependencies with:")
        print("    pip install -r requirements.txt")
        print(f"(import error: {e})")
        return 1

    # --- data + engine (always available) ---------------------------------
    dm = DataManager(rank=args.rank).load()
    engine = ScoringEngine(dm)
    print(f"Loaded {len(dm.all_heroes())} heroes • rank={dm.rank} • patch {dm.patch}")

    # --- vision (live or mock) --------------------------------------------
    from src.vision import MockVision  # MockVision has no heavy deps

    if args.mock:
        vision = MockVision()
        print("Vision: MOCK mode (no screen capture).")
    else:
        try:
            from src.vision import VisionModule

            vision = VisionModule(valid_names=dm.names)
            if vision.template_count == 0:
                print("WARNING: no portrait templates in data/portraits/ — "
                      "the scanner will see an empty draft. Add <Hero>.png files "
                      "and calibrate config.REGIONS (see tools/calibrate.py).")
            else:
                print(f"Vision: LIVE • {vision.template_count} templates loaded.")
        except Exception as e:
            print(f"Vision unavailable ({e}); falling back to MOCK mode.")
            vision = MockVision()

    # --- background scan/score worker -------------------------------------
    class DraftWorker(QtCore.QThread):
        updated = QtCore.pyqtSignal(dict)

        def __init__(self, flags: OverlayFlags, interval_s: float):
            super().__init__()
            self.flags = flags
            self.interval_ms = max(100, int(interval_s * 1000))
            self._stop = False

        def run(self) -> None:
            while not self._stop:
                try:
                    draft = vision.scan()
                    recs = engine.recommend(
                        draft,
                        lane_counter=self.flags.lane_counter,
                        dark_system=self.flags.dark_system,
                    )
                    self.updated.emit(recs)
                except Exception as exc:  # keep the loop alive on transient errors
                    print(f"[worker] scan error: {exc}")
                self.msleep(self.interval_ms)

        def stop(self) -> None:
            self._stop = True

    app = QtWidgets.QApplication(sys.argv)
    flags = OverlayFlags(lane_counter=args.lane_counter, dark_system=args.dark)
    win = OverlayWindow(flags)
    win.set_rank_label(f"{dm.rank} • patch {dm.patch}")

    worker = DraftWorker(flags, args.interval)
    worker.updated.connect(win.update_recommendations)
    worker.start()

    win.show()
    print("Overlay running. Drag to move; toggle Lock for click-through.")
    code = app.exec_()

    worker.stop()
    worker.wait(2000)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
