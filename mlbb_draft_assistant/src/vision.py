"""
Module 2 — Vision (Screen Capture & Image Recognition)
======================================================

Continuously reads the draft screen and reports which heroes are picked / banned
by both teams, returning a :class:`~src.scoring_engine.DraftState`.

Pipeline
--------
1. **Capture** the whole monitor once per scan with ``mss`` (fast, no temp files).
2. **Crop** the fixed pick/ban *slot* rectangles defined in ``config.REGIONS``.
3. **Match** each slot against a library of hero portrait templates using
   OpenCV template matching (``cv2.matchTemplate`` / ``TM_CCOEFF_NORMED``); the
   best score above ``config.MATCH_THRESHOLD`` wins, otherwise the slot is empty.

Setup
-----
Drop one cropped portrait per hero into ``data/portraits/`` named after the
hero, e.g. ``Fanny.png``, ``Yu Zhong.png`` (underscores also accepted:
``Yu_Zhong.png``). Then calibrate ``config.REGIONS`` with ``tools/calibrate.py``.

The heavy dependencies (OpenCV / numpy / mss) are imported lazily so the rest of
the project can be imported and unit-tested without them. Use :class:`MockVision`
for a screen-free demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
from src.scoring_engine import DraftState

try:  # heavy / platform deps — optional
    import cv2
    import numpy as np
    import mss
    _DEPS_OK = True
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on environment
    _DEPS_OK = False
    _IMPORT_ERROR = exc


Slot = Tuple[int, int, int, int]  # (left, top, width, height)


class VisionModule:
    """Screen-capture + template-matching draft reader."""

    def __init__(
        self,
        regions: Optional[Dict[str, List[Slot]]] = None,
        portraits_dir: Path | str = config.PORTRAITS_DIR,
        threshold: float = config.MATCH_THRESHOLD,
        monitor: int = config.CAPTURE_MONITOR,
        valid_names: Optional[List[str]] = None,
    ) -> None:
        if not _DEPS_OK:
            raise RuntimeError(
                "Vision dependencies missing (cv2/numpy/mss). "
                f"Install requirements.txt or use MockVision. Original error: {_IMPORT_ERROR}"
            )
        self.regions = regions or config.REGIONS
        self.portraits_dir = Path(portraits_dir)
        self.threshold = threshold
        self.monitor = monitor
        self._valid = set(valid_names) if valid_names else None
        self.templates: Dict[str, "np.ndarray"] = self._load_templates()

    # ---- template library --------------------------------------------- #
    def _load_templates(self) -> Dict[str, "np.ndarray"]:
        templates: Dict[str, "np.ndarray"] = {}
        if not self.portraits_dir.exists():
            return templates
        for path in sorted(self.portraits_dir.glob("*.png")):
            name = path.stem.replace("_", " ")
            if self._valid is not None and name not in self._valid:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                templates[name] = img
        return templates

    @property
    def template_count(self) -> int:
        return len(self.templates)

    # ---- capture ------------------------------------------------------- #
    def grab_monitor(self) -> "np.ndarray":
        """Capture the configured monitor as a BGR ndarray."""
        with mss.mss() as sct:
            mon = sct.monitors[self.monitor]
            shot = sct.grab(mon)
            frame = np.asarray(shot)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    @staticmethod
    def _crop(frame: "np.ndarray", slot: Slot) -> "np.ndarray":
        left, top, w, h = slot
        return frame[top : top + h, left : left + w]

    # ---- matching ------------------------------------------------------ #
    def _match_slot(self, slot_img: "np.ndarray") -> Tuple[Optional[str], float]:
        """Return (best_hero_name, confidence) for one slot crop."""
        if slot_img.size == 0 or not self.templates:
            return None, 0.0
        h, w = slot_img.shape[:2]
        best_name, best_score = None, 0.0
        for name, template in self.templates.items():
            resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(slot_img, resized, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_name, best_score = name, score
        if best_score >= self.threshold:
            return best_name, best_score
        return None, best_score

    def _read_region(self, frame: "np.ndarray", region_name: str) -> List[str]:
        found: List[str] = []
        for slot in self.regions.get(region_name, []):
            name, _ = self._match_slot(self._crop(frame, slot))
            if name and name not in found:
                found.append(name)
        return found

    # ---- public API ---------------------------------------------------- #
    def scan(self) -> DraftState:
        """Capture once and return the detected draft state."""
        frame = self.grab_monitor()
        ally_picks = self._read_region(frame, "ally_picks")
        enemy_picks = self._read_region(frame, "enemy_picks")
        bans = self._read_region(frame, "ally_bans") + self._read_region(
            frame, "enemy_bans"
        )
        # De-dupe bans while preserving order.
        bans = list(dict.fromkeys(bans))
        return DraftState(
            ally_picks=ally_picks, enemy_picks=enemy_picks, bans=bans
        )

    # ---- calibration helpers ------------------------------------------ #
    def debug_regions(self, out_dir: Path | str = "calibration_debug") -> Path:
        """Save the current monitor and each slot crop for visual calibration."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        frame = self.grab_monitor()
        cv2.imwrite(str(out / "_full.png"), frame)
        for region_name, slots in self.regions.items():
            for i, slot in enumerate(slots):
                crop = self._crop(frame, slot)
                if crop.size:
                    cv2.imwrite(str(out / f"{region_name}_{i}.png"), crop)
        return out


class MockVision:
    """
    Screen-free stand-in for :class:`VisionModule`.

    Returns a fixed (or scripted) draft state so the engine + overlay can be
    demoed and tested without the game running. Pass a list of ``DraftState``
    objects to ``script`` to step through a fake draft on successive ``scan()``s.
    """

    def __init__(
        self,
        state: Optional[DraftState] = None,
        script: Optional[List[DraftState]] = None,
    ) -> None:
        self.template_count = 0
        self._state = state or DraftState(
            ally_picks=["Tigreal"],
            enemy_picks=["Layla", "Esmeralda"],
            bans=["Fanny", "Ling", "Valentina", "Joy"],
        )
        self._script = script or []
        self._step = 0

    def scan(self) -> DraftState:
        if self._script:
            state = self._script[min(self._step, len(self._script) - 1)]
            self._step += 1
            return state
        return self._state


# --------------------------------------------------------------------------- #
# Manual smoke test:  python -m src.vision
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if _DEPS_OK:
        try:
            vm = VisionModule()
            print(f"Vision ready. Loaded {vm.template_count} portrait templates.")
            if vm.template_count == 0:
                print(
                    "No templates found in data/portraits/. Add <Hero>.png files, "
                    "then run tools/calibrate.py to align config.REGIONS."
                )
            else:
                print("Scanning current screen...")
                print(vm.scan())
        except Exception as e:  # pragma: no cover
            print(f"Vision error: {e}")
    else:
        print(f"Vision deps unavailable ({_IMPORT_ERROR}); using MockVision.")
        print(MockVision().scan())
