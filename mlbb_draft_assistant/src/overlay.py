"""
Module 4 — UI Overlay
=====================

A frameless, always-on-top, semi-transparent PyQt5 overlay that shows the top
recommended hero per lane (Gold / EXP / Mid / Jungle / Roam) and exposes the
draft controls.

Controls
--------
* **Next**          page through the ranked suggestions per lane (for when you
  don't own the #1 pick).
* **Lane Counter**  toggle the engine's lane-counter prediction mode.
* **Dark System**   joke toggle — recommend the statistically *worst* heroes.
* **Lock (click-through)**  make the overlay ignore the mouse so clicks pass to
  the game. On Windows this uses the layered/transparent window styles; unlock
  again with the global hotkey ``Ctrl+Alt+L`` (the lock button itself becomes
  click-through while locked).

Click-through note
------------------
True OS-level click-through with *still-clickable* buttons is platform specific.
The pragmatic, widely-used approach implemented here is a **lock toggle**:
unlocked = interactive (press buttons, drag to move); locked = fully
click-through (play the game, overlay just displays). Windows uses
``WS_EX_LAYERED | WS_EX_TRANSPARENT`` via ctypes; other platforms fall back to
Qt's ``WA_TransparentForMouseEvents`` where supported.

Unlike the logic modules, this one imports PyQt5 directly — ``main.py`` wraps the
import so a missing GUI dependency degrades gracefully.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

import config

# Optional global-hotkey support (lets you unlock while click-through is on).
try:
    import keyboard  # type: ignore
    _HOTKEYS_OK = True
except Exception:
    _HOTKEYS_OK = False

QT_AVAILABLE = True  # importing this module succeeded => Qt is present


@dataclass
class OverlayFlags:
    """Shared, mutable settings the worker thread reads each scan."""

    lane_counter: bool = False
    dark_system: bool = False
    locked: bool = False


class LaneCard(QtWidgets.QFrame):
    """One column of the overlay: a lane label + its current suggestion."""

    def __init__(self, lane: str, parent=None):
        super().__init__(parent)
        self.lane = lane
        self.setObjectName("laneCard")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        self.lane_lbl = QtWidgets.QLabel(config.LANE_LABELS[lane])
        self.lane_lbl.setObjectName("laneLabel")
        self.lane_lbl.setAlignment(Qt.AlignCenter)

        self.hero_lbl = QtWidgets.QLabel("—")
        self.hero_lbl.setObjectName("heroLabel")
        self.hero_lbl.setAlignment(Qt.AlignCenter)

        self.score_lbl = QtWidgets.QLabel("")
        self.score_lbl.setObjectName("scoreLabel")
        self.score_lbl.setAlignment(Qt.AlignCenter)

        self.next_lbl = QtWidgets.QLabel("")
        self.next_lbl.setObjectName("nextLabel")
        self.next_lbl.setAlignment(Qt.AlignCenter)
        self.next_lbl.setWordWrap(True)

        for w in (self.lane_lbl, self.hero_lbl, self.score_lbl, self.next_lbl):
            lay.addWidget(w)

    def update_card(self, suggestions: List, page: int) -> None:
        """Show suggestion at ``page`` (wrapping) plus a preview of what's next."""
        if not suggestions:
            self.hero_lbl.setText("—")
            self.score_lbl.setText("")
            self.next_lbl.setText("")
            self.setToolTip("")
            return
        idx = page % len(suggestions)
        cur = suggestions[idx]
        self.hero_lbl.setText(f"{idx + 1}. {cur.hero}")
        self.score_lbl.setText(f"{cur.total:.0f} pts")
        upcoming = [
            suggestions[(idx + k) % len(suggestions)].hero
            for k in range(1, min(3, len(suggestions)))
        ]
        self.next_lbl.setText("next: " + ", ".join(upcoming) if upcoming else "")
        # Full reasoning on hover.
        self.setToolTip(cur.explain() if hasattr(cur, "explain") else "")


class OverlayWindow(QtWidgets.QWidget):
    """The main always-on-top overlay window."""

    # Emitted from the (background) global-hotkey thread; handled on GUI thread.
    hotkey_lock = pyqtSignal()
    hotkey_next = pyqtSignal()
    hotkey_quit = pyqtSignal()

    def __init__(self, flags: Optional[OverlayFlags] = None, parent=None):
        super().__init__(parent)
        self.flags = flags or OverlayFlags()
        self.page = 0
        self._latest: Dict[str, List] = {ln: [] for ln in config.LANES}
        self._drag_pos: Optional[QtCore.QPoint] = None

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # keep out of the taskbar / alt-tab
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("MLBB Draft Assistant")

        self._build_ui()
        self._apply_style()
        self._wire_hotkeys()
        self.move(60, 60)

    # ---- UI construction ---------------------------------------------- #
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        # Header / control bar -----------------------------------------
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("MLBB Draft Assistant")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)

        self.rank_lbl = QtWidgets.QLabel("")
        self.rank_lbl.setObjectName("rankLabel")
        header.addWidget(self.rank_lbl)

        self.btn_next = QtWidgets.QPushButton("Next ▸")
        self.btn_next.clicked.connect(self.next_page)

        self.btn_lane = QtWidgets.QPushButton("Lane Counter")
        self.btn_lane.setCheckable(True)
        self.btn_lane.setChecked(self.flags.lane_counter)
        self.btn_lane.toggled.connect(self._on_lane_toggle)

        self.btn_dark = QtWidgets.QPushButton("Dark System")
        self.btn_dark.setCheckable(True)
        self.btn_dark.setChecked(self.flags.dark_system)
        self.btn_dark.toggled.connect(self._on_dark_toggle)

        self.btn_lock = QtWidgets.QPushButton("🔓 Lock")
        self.btn_lock.setCheckable(True)
        self.btn_lock.toggled.connect(self._on_lock_toggle)

        self.btn_quit = QtWidgets.QPushButton("✕")
        self.btn_quit.setObjectName("quitBtn")
        self.btn_quit.clicked.connect(QtWidgets.QApplication.quit)

        for b in (self.btn_next, self.btn_lane, self.btn_dark, self.btn_lock, self.btn_quit):
            header.addWidget(b)
        root.addLayout(header)

        # Lane cards ----------------------------------------------------
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(6)
        self.cards: Dict[str, LaneCard] = {}
        for lane in config.LANES:
            card = LaneCard(lane)
            self.cards[lane] = card
            body.addWidget(card)
        root.addLayout(body)

        # Footer hint ---------------------------------------------------
        hint = QtWidgets.QLabel(
            "Drag to move • Hover a pick for the reasoning • "
            + ("Ctrl+Alt+L unlock, Ctrl+Alt+N next" if _HOTKEYS_OK else "(install 'keyboard' for global hotkeys)")
        )
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        root.addWidget(hint)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { color: #f0f0f0; font-family: 'Segoe UI', Arial; }
            #title { font-size: 14px; font-weight: bold; color: #ffd54a; }
            #rankLabel { color: #9fd0ff; font-size: 11px; padding-right: 6px; }
            #hint { color: #9aa0a6; font-size: 10px; }
            QFrame#laneCard {
                background: rgba(20, 22, 30, 200);
                border: 1px solid rgba(120,130,160,90);
                border-radius: 8px;
                min-width: 120px;
            }
            #laneLabel { color: #8ab4f8; font-size: 11px; font-weight: bold; letter-spacing: 1px; }
            #heroLabel { color: #ffffff; font-size: 15px; font-weight: bold; }
            #scoreLabel { color: #7ee787; font-size: 11px; }
            #nextLabel { color: #9aa0a6; font-size: 9px; }
            QPushButton {
                background: rgba(40, 44, 60, 220);
                border: 1px solid rgba(120,130,160,120);
                border-radius: 6px; padding: 4px 8px; font-size: 11px;
            }
            QPushButton:hover { background: rgba(60, 66, 90, 230); }
            QPushButton:checked { background: #2d6cdf; border-color: #5b8def; }
            #quitBtn { color: #ff8a80; font-weight: bold; }
            """
        )

    # ---- public update slot ------------------------------------------- #
    @QtCore.pyqtSlot(dict)
    def update_recommendations(self, recs: Dict[str, List]) -> None:
        """Refresh the cards with a new ``{lane: [ScoreBreakdown, ...]}`` map."""
        self._latest = recs
        self._render()

    def set_rank_label(self, text: str) -> None:
        self.rank_lbl.setText(text)

    def _render(self) -> None:
        for lane, card in self.cards.items():
            card.update_card(self._latest.get(lane, []), self.page)

    # ---- control callbacks -------------------------------------------- #
    def next_page(self) -> None:
        self.page += 1
        self._render()

    def _on_lane_toggle(self, on: bool) -> None:
        self.flags.lane_counter = on

    def _on_dark_toggle(self, on: bool) -> None:
        self.flags.dark_system = on
        self.btn_dark.setText("😈 Dark System" if on else "Dark System")

    def _on_lock_toggle(self, on: bool) -> None:
        self.flags.locked = on
        self.btn_lock.setText("🔒 Locked" if on else "🔓 Lock")
        self._set_click_through(on)

    # ---- click-through ------------------------------------------------- #
    def _set_click_through(self, enable: bool) -> None:
        """Make the whole overlay transparent to mouse input when locked."""
        if sys.platform == "win32":
            try:
                import ctypes

                GWL_EXSTYLE = -20
                WS_EX_LAYERED = 0x00080000
                WS_EX_TRANSPARENT = 0x00000020
                hwnd = int(self.winId())
                user32 = ctypes.windll.user32
                style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                if enable:
                    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
                else:
                    style &= ~WS_EX_TRANSPARENT
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            except Exception as e:  # pragma: no cover
                print(f"[overlay] click-through unavailable: {e}")
        else:
            # Best-effort cross-platform fallback (works on some compositors).
            self.setAttribute(Qt.WA_TransparentForMouseEvents, enable)

    # ---- global hotkeys ----------------------------------------------- #
    def _wire_hotkeys(self) -> None:
        self.hotkey_lock.connect(lambda: self.btn_lock.toggle())
        self.hotkey_next.connect(self.next_page)
        self.hotkey_quit.connect(QtWidgets.QApplication.quit)
        if not _HOTKEYS_OK:
            return
        try:
            # keyboard callbacks fire on another thread -> emit signals (queued).
            keyboard.add_hotkey("ctrl+alt+l", lambda: self.hotkey_lock.emit())
            keyboard.add_hotkey("ctrl+alt+n", lambda: self.hotkey_next.emit())
            keyboard.add_hotkey("ctrl+alt+q", lambda: self.hotkey_quit.emit())
        except Exception as e:  # pragma: no cover
            print(f"[overlay] global hotkeys unavailable: {e}")

    # ---- dragging (when unlocked) ------------------------------------- #
    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == Qt.LeftButton and not self.flags.locked:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._drag_pos is not None and not self.flags.locked:
            self.move(e.globalPos() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        self._drag_pos = None


# --------------------------------------------------------------------------- #
# Standalone demo:  python -m src.overlay
# (Builds the real engine on mock draft data so you can see the overlay.)
# --------------------------------------------------------------------------- #
def _demo() -> int:
    from src.data_manager import DataManager
    from src.scoring_engine import ScoringEngine
    from src.vision import MockVision

    app = QtWidgets.QApplication(sys.argv)
    dm = DataManager().load()
    engine = ScoringEngine(dm)
    vision = MockVision()
    win = OverlayWindow()
    win.set_rank_label(f"{dm.rank} • patch {dm.patch}")

    def refresh():
        draft = vision.scan()
        recs = engine.recommend(
            draft,
            lane_counter=win.flags.lane_counter,
            dark_system=win.flags.dark_system,
        )
        win.update_recommendations(recs)

    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(int(config.SCAN_INTERVAL_S * 1000))
    refresh()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(_demo())
