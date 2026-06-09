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
import os
import sys
from typing import List, Optional, Tuple

from PyQt5.QtCore import QThread, QTimer, pyqtSignal

import config
from engine import DraftState, HeroDB, ScoringEngine, Settings
from ui import DraftOverlay


def _resolve_region(args) -> dict:
    """Capture region on the desktop.  Default = the whole primary monitor, so
    the overlay works at the user's real PC resolution regardless of how Scrcpy
    scaled the phone."""
    if args.region:
        l, t, w, h = (int(v) for v in args.region.split(","))
        return {"left": l, "top": t, "width": w, "height": h}
    try:
        from detector import ScreenCapturer
        return ScreenCapturer.primary_monitor()
    except Exception as exc:
        sys.stderr.write(f"[region] could not query monitor ({exc}); "
                         "falling back to reference frame.\n")
        return dict(config.ACTIVE_REGION)


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
        self.repo = None
        self._stats_timer: Optional[QTimer] = None

        # Recompute whenever the board OR the toggles change.
        self.overlay.settingsChanged.connect(self._on_settings)
        self.overlay.closed.connect(self.shutdown)

    # ----- live stats -----------------------------------------------------
    def enable_live_stats(self, repo, interval_sec: int) -> None:
        """Periodically refresh hero stats from the StatsRepository in place.
        The DB is mutated, so the engine immediately scores on fresh numbers."""
        self.repo = repo
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(max(30, interval_sec) * 1000)

    def _refresh_stats(self) -> None:
        if self.repo is None:
            return
        if self.repo.refresh(self.db) > 0:
            self._recompute()                     # re-score with new stats

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
    def start_live(self, template_dir: str, ban_dir: str, accept_low: bool) -> None:
        # Imported lazily so --mock works on machines without the CV stack.
        from detector import ScreenCapturer, TemplateLibrary, DraftDetector

        library = TemplateLibrary.from_dir(template_dir)
        if len(library) == 0:
            sys.stderr.write(
                f"[warn] no templates found in '{template_dir}'. "
                "Detection will rely on the histogram fallback only - drop "
                "cropped hero avatars (e.g. guinevere.png) into that folder.\n")
        # Circular ban-row templates (optional): better matches for the small
        # circular ban icons.  Falls back to the main library if absent.
        ban_library = TemplateLibrary.from_dir(ban_dir)
        if len(ban_library) == 0:
            ban_library = None
        else:
            # Backfill any hero the circular pack lacks (e.g. Sora) with the
            # square portrait so bans never silently match the wrong hero.
            if len(library):
                ban_library.merge_missing(library)
            print(f"[templates] {len(library)} pick + {len(ban_library)} ban templates")
        capturer = ScreenCapturer()
        detector = DraftDetector(self.db, library=library, ban_library=ban_library,
                                 accept_low=accept_low)

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
                        help="folder of cropped hero avatar PNGs (picks)")
    parser.add_argument("--ban-templates", default=config.TEMPLATE_BAN_DIR,
                        help="folder of circular ban-icon templates (ban row)")
    parser.add_argument("--heroes", default="heroes.json",
                        help="hero database json")
    parser.add_argument("--stats-url", default=config.STATS_URL,
                        help="JSON endpoint for live win/ban stats (overlays "
                             "heroes.json; cached + auto-refreshed)")
    parser.add_argument("--region", default=None,
                        help="capture region L,T,W,H (default: whole primary monitor)")
    parser.add_argument("--layout", default=config.LAYOUT_FILE,
                        help="calibration JSON produced by calibrate.py")
    parser.add_argument("--accept-low", action="store_true",
                        help="accept low-confidence template guesses")
    parser.add_argument("--mock", action="store_true",
                        help="run a scripted demo without screen capture")
    parser.add_argument("--calibrate", action="store_true",
                        help="click-to-align the boxes over the live game, "
                             "save layout.json, then exit")
    args = parser.parse_args()

    # DPI awareness BEFORE Qt so mss (physical px) and PyQt agree on coordinates
    # (Windows display scaling otherwise offsets/rescales the boxes).
    config.enable_dpi_awareness()
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt as _Qt
    try:
        QApplication.setAttribute(_Qt.AA_DisableHighDpiScaling, True)
    except Exception:
        pass
    app = QApplication(sys.argv)

    # Resolve capture region + per-device layout BEFORE building the overlay,
    # so the boxes are positioned for THIS screen (not the 2712x1220 phone).
    region = _resolve_region(args)
    layout = None
    if args.layout and os.path.exists(args.layout):
        try:
            layout = config.load_layout(args.layout)
            print(f"[layout] loaded calibration from {os.path.abspath(args.layout)}")
        except Exception as exc:
            sys.stderr.write(f"[layout] could not read {args.layout}: {exc}\n")
    if layout is None and not args.calibrate:
        sys.stderr.write("[layout] no calibration found - using scaled fallback. "
                         "Run 'python main.py --calibrate' for pixel-perfect boxes.\n")
    config.apply_region(region, layout)

    # Diagnostics: capture size vs Qt screen reveals any residual DPI mismatch.
    scr = app.primaryScreen()
    sw, sh = scr.size().width(), scr.size().height()
    print(f"[display] capture {region['width']}x{region['height']} | "
          f"Qt screen {sw}x{sh} dpr={scr.devicePixelRatio():.2f}")
    if (sw, sh) != (region["width"], region["height"]):
        sys.stderr.write("[display] WARNING: capture size != Qt screen size - "
                         "DPI scaling may still offset boxes.\n")

    # ---- calibration mode: align on the live game, save, exit ----
    if args.calibrate:
        from ui import CalibrationOverlay
        cal = CalibrationOverlay(config.ACTIVE_REGION, args.layout)
        cal.saved.connect(lambda pth: (
            print(f"[calibrate] saved {pth}\nNow run:  python main.py"), app.quit()))
        cal.cancelled.connect(lambda: (
            print("[calibrate] cancelled (nothing saved)"), app.quit()))
        cal.show(); cal.raise_(); cal.activateWindow(); cal.setFocus()
        return app.exec_()

    # Build the hero DB - seed from heroes.json, optionally overlay live stats.
    repo = None
    if args.stats_url:
        from stats_provider import (StatsRepository, HttpJsonStatsProvider,
                                    CachedStatsProvider)
        http = HttpJsonStatsProvider(
            args.stats_url,
            record_path=config.STATS_RECORD_PATH,
            name_key=config.STATS_NAME_KEY,
            field_map=config.STATS_FIELD_MAP,
        )
        provider = CachedStatsProvider(http, cache_path=config.STATS_CACHE_PATH,
                                       ttl=config.STATS_CACHE_TTL)
        repo = StatsRepository(args.heroes, provider)
        _DB = repo.build()
    else:
        _DB = HeroDB.load(args.heroes)

    assistant = DraftAssistant(_DB)
    if repo is not None:
        assistant.enable_live_stats(repo, config.STATS_REFRESH_SEC)
    assistant.show()

    if args.mock:
        assistant.start_mock()
    else:
        try:
            assistant.start_live(args.templates, args.ban_templates, args.accept_low)
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
