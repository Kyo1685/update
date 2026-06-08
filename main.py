"""
main.py
=======
Entry point that wires the three subsystems together:

    ScreenCapturer + DraftDetector   (detector.py)   --> DraftState
    ScoringEngine                    (engine.py)     --> DraftResult
    DraftOverlay (canvas + dock)     (ui.py)         --> on-screen HUD

The capture+detect loop lives on a background QThread so the Qt event loop
(and therefore every animation / click) stays buttery smooth.  The worker only
emits when the board actually changes, and the detector caches unchanged slots,
so an idle draft costs almost nothing.

Usage
-----
    python main.py                      # live: capture the Scrcpy mirror
    python main.py --templates ./templates
    python main.py --accept-low         # trust low-confidence guesses too
    python main.py --mock               # demo with a scripted draft (no game)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

from PyQt5.QtCore import QThread, QTimer, pyqtSignal

import config
from engine import DraftState, HeroDB, ScoringEngine, Settings
from ui import DraftOverlay


# ---------------------------------------------------------------------------
# Background capture / detection worker
# ---------------------------------------------------------------------------
class DetectorWorker(QThread):
    """Runs capture->detect on its own thread; emits a DraftState only when
    the board changes (cheap idle, no UI stutter)."""
    state_ready = pyqtSignal(object)         # emits DraftState

    def __init__(self, detector, capturer, interval: float = config.DETECT_INTERVAL):
        super().__init__()
        self._detector = detector
        self._capturer = capturer
        self._interval = interval
        self._running = False
        self._last_key: Optional[Tuple] = None

    @staticmethod
    def _key(state: DraftState) -> Tuple:
        return (tuple(state.ally_picks), tuple(state.enemy_picks),
                tuple(state.ally_bans), tuple(state.enemy_bans))

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                frame = self._capturer.grab()
                state = self._detector.detect(frame)
                key = self._key(state)
                if key != self._last_key:
                    self._last_key = key
                    self.state_ready.emit(state)
            except Exception as exc:                      # never kill the thread
                # In production route this to logging; keep the loop alive.
                sys.stderr.write(f"[detector] {exc}\n")
            self.msleep(int(self._interval * 1000))

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


# ---------------------------------------------------------------------------
# Application controller
# ---------------------------------------------------------------------------
class DraftAssistant:
    def __init__(self, db: HeroDB):
        self.db = db
        self.engine = ScoringEngine(db)
        self.overlay = DraftOverlay(db)
        self.state = DraftState()
        self.worker: Optional[DetectorWorker] = None

        # Recompute whenever the board OR the toggles change.
        self.overlay.settingsChanged.connect(self._on_settings)
        self.overlay.closed.connect(self.shutdown)

    # ----- event handlers -------------------------------------------------
    def on_state(self, state: DraftState) -> None:
        self.state = state
        self.overlay.update_state(state)
        self._recompute()

    def _on_settings(self, _settings: Settings) -> None:
        self._recompute()

    def _recompute(self) -> None:
        result = self.engine.evaluate(self.state, self.overlay.settings)
        self.overlay.update_result(result)

    # ----- lifecycle ------------------------------------------------------
    def start_live(self, template_dir: str, accept_low: bool) -> None:
        # Imported lazily so --mock works on machines without the CV stack.
        from detector import ScreenCapturer, TemplateLibrary, DraftDetector

        library = TemplateLibrary.from_dir(template_dir)
        if len(library) == 0:
            sys.stderr.write(
                f"[warn] no templates found in '{template_dir}'. "
                "Detection will rely on the histogram fallback only - drop "
                "cropped hero avatars (e.g. guinevere.png) into that folder.\n")
        capturer = ScreenCapturer()
        detector = DraftDetector(self.db, library=library, accept_low=accept_low)

        self.worker = DetectorWorker(detector, capturer)
        self.worker.state_ready.connect(self.on_state)
        self.worker.start()

    def start_mock(self) -> None:
        """Step through a scripted draft so the HUD can be demoed live."""
        script: List[DraftState] = _mock_script()
        self._mock_idx = 0
        self._mock_timer = QTimer()
        self._mock_timer.timeout.connect(lambda: self._mock_step(script))
        self._mock_timer.start(1500)
        self._mock_step(script)

    def _mock_step(self, script: List[DraftState]) -> None:
        state = script[self._mock_idx % len(script)]
        self._mock_idx += 1
        self.on_state(state)

    def show(self) -> None:
        self.overlay.show()
        self._recompute()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.stop()


# ---------------------------------------------------------------------------
# Mock data (mirrors the reference screenshot, then keeps drafting)
# ---------------------------------------------------------------------------
def _mock_script() -> List[DraftState]:
    bans_a = ["Harley", "Karrie", "Sora", "Gloo", "Freya"]
    bans_e = ["Freya", "Gusion", "Yi Sun-shin", "Sora", "Zhuxin"]

    def st(ally, enemy):
        from detector import assign_lanes        # pure-python, no cv2 needed
        s = DraftState(ally_picks=ally + [None] * (5 - len(ally)),
                       enemy_picks=enemy + [None] * (5 - len(enemy)),
                       ally_bans=bans_a, enemy_bans=bans_e)
        s.ally_lanes = assign_lanes(s.ally_picks, _DB)
        s.enemy_lanes = assign_lanes(s.enemy_picks, _DB)
        return s

    return [
        st(["Guinevere"], ["Alice"]),
        st(["Guinevere", "Zetian"], ["Alice", "Khufra"]),
        st(["Guinevere", "Zetian", "Minsithar"], ["Alice", "Khufra", "Akai"]),
        st(["Guinevere", "Zetian", "Minsithar"], ["Alice", "Khufra", "Akai", "Gord"]),
    ]


_DB: Optional[HeroDB] = None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    global _DB
    parser = argparse.ArgumentParser(description="MLBB real-time draft overlay")
    parser.add_argument("--templates", default=config.TEMPLATE_DIR,
                        help="folder of cropped hero avatar PNGs")
    parser.add_argument("--heroes", default="heroes.json",
                        help="hero database json")
    parser.add_argument("--accept-low", action="store_true",
                        help="accept low-confidence template guesses")
    parser.add_argument("--mock", action="store_true",
                        help="run a scripted demo without screen capture")
    args = parser.parse_args()

    # QApplication must exist before any widget.
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    _DB = HeroDB.load(args.heroes)
    assistant = DraftAssistant(_DB)
    assistant.show()

    if args.mock:
        assistant.start_mock()
    else:
        try:
            assistant.start_live(args.templates, args.accept_low)
        except Exception as exc:
            sys.stderr.write(
                f"[fatal] could not start live capture: {exc}\n"
                "Tip: run 'python main.py --mock' to preview the HUD, or install "
                "the CV stack:  pip install numpy opencv-python mss\n")
            return 2

    exit_code = app.exec_()
    assistant.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
