"""
ui.py
=====
PyQt5 overlay that reproduces the "Otev-" AI LINEUP DRAFTER HUD.

Architecture note (important):
------------------------------
A single window cannot be BOTH globally click-through *and* host clickable
buttons - ``Qt.WindowTransparentForInput`` makes the whole surface ignore the
mouse, and ``setMask`` would clip the rendered boxes, not just the input.  The
production-correct solution is two cooperating top-level windows:

  * OverlayCanvas  - frameless, translucent, **WindowTransparentForInput**
                     (true click-through).  Pure painting: the bounding boxes
                     and the "NAME [ROLE]" tags drawn over the live portraits.
                     Never steals a click from the game.

  * ControlDock    - frameless, translucent, **interactive**.  The draggable
                     drafter panel: toggles, per-lane suggestions with chevron
                     paging, the win-probability gauge and the build-path feed.

``DraftOverlay`` wires the two together and exposes a tiny update API that
main.py drives from the detector/engine results.

Run ``python ui.py`` for a self-contained mock demo (no game / capture needed).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from PyQt5.QtCore import (Qt, QSize, QRectF, QPoint, pyqtSignal, pyqtProperty,
                          QPropertyAnimation, QEasingCurve)
from PyQt5.QtGui import (QPainter, QColor, QBrush, QPen, QFont, QFontMetrics,
                         QPainterPath)
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QAbstractButton,
                             QHBoxLayout, QVBoxLayout, QSizePolicy, QApplication)

import config
from config import THEME
from engine import DraftState, Settings, Suggestion, DraftResult
from detector import predicted_role_label


def qcolor(c) -> QColor:
    """RGBA tuple -> QColor."""
    return QColor(c[0], c[1], c[2], c[3] if len(c) > 3 else 255)


# ===========================================================================
#  CUSTOM WIDGET: slick animated toggle switch
# ===========================================================================
class ToggleSwitch(QAbstractButton):
    """A modern iOS-style switch.  Checkable; emits ``toggled(bool)``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(56, 28)
        self._knob = 0.0                                 # 0 = off, 1 = on
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate)

    # animated property ----------------------------------------------------
    def getKnob(self) -> float:
        return self._knob

    def setKnob(self, v: float) -> None:
        self._knob = v
        self.update()

    knob = pyqtProperty(float, fget=getKnob, fset=setKnob)

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def sizeHint(self) -> QSize:
        return QSize(56, 28)

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        radius = r.height() / 2.0

        off = QColor(46, 56, 52)
        on = qcolor(THEME.accent)
        # interpolate track colour by knob position
        t = self._knob
        track = QColor(
            int(off.red() + (on.red() - off.red()) * t),
            int(off.green() + (on.green() - off.green()) * t),
            int(off.blue() + (on.blue() - off.blue()) * t),
        )
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(r), radius, radius)

        # knob
        d = r.height() - 6
        x = r.left() + 3 + (r.width() - d - 6) * self._knob
        p.setBrush(QColor(245, 255, 250))
        p.drawEllipse(QRectF(x, r.top() + 3, d, d))


# ===========================================================================
#  CUSTOM WIDGET: win-probability gauge (green vs red)
# ===========================================================================
class ProbabilityBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._ally = 50.0
        self._enemy = 50.0

    def set_values(self, ally: float, enemy: float) -> None:
        self._ally, self._enemy = ally, enemy
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        total = max(self._ally + self._enemy, 1e-6)
        split = int(r.width() * (self._ally / total))

        path = QPainterPath()
        path.addRoundedRect(QRectF(r), 5, 5)
        p.setClipPath(path)

        p.fillRect(0, r.top(), split, r.height(), qcolor(THEME.prob_good))
        p.fillRect(split, r.top(), r.width() - split, r.height(),
                   qcolor(THEME.prob_bad))
        p.setClipping(False)
        p.setPen(QPen(qcolor(THEME.panel_border), 1))
        p.drawPath(path)


# ===========================================================================
#  CUSTOM WIDGET: one suggested-lane row with chevron paging
# ===========================================================================
class RoleRow(QWidget):
    """'Suggested MID:  4.Kadita 5.Pharsa 6.Lunox   < >'

    Chevrons slide a 3-wide window through the Top-N (default 10) list so a
    user with a limited hero pool can keep scrolling for an option they own.
    """
    PAGE_SIZE = 3
    MAX_RANK = 10

    def __init__(self, lane: str, parent=None):
        super().__init__(parent)
        self.lane = lane
        self._suggestions: List[Suggestion] = []
        self._page = 0

        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(6)

        self.tag = QLabel(f"{config.LANE_ABBR.get(lane, lane)}")
        self.tag.setFixedWidth(54)
        self.tag.setObjectName("laneTag")

        self.prev = QPushButton("‹")               # ‹
        self.next = QPushButton("›")               # ›
        for b in (self.prev, self.next):
            b.setObjectName("chevron")
            b.setFixedSize(26, 26)
            b.setCursor(Qt.PointingHandCursor)
        self.prev.clicked.connect(self._page_prev)
        self.next.clicked.connect(self._page_next)

        self.value = QLabel("—")
        self.value.setObjectName("suggestion")
        self.value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.value.setTextInteractionFlags(Qt.NoTextInteraction)

        row.addWidget(self.tag)
        row.addWidget(self.value, 1)
        row.addWidget(self.prev)
        row.addWidget(self.next)

    # --- data -------------------------------------------------------------
    def set_suggestions(self, suggestions: List[Suggestion]) -> None:
        self.prev.setVisible(True)
        self.next.setVisible(True)
        self.tag.setStyleSheet("")                     # revert to QSS accent
        self._suggestions = suggestions[: self.MAX_RANK]
        self._page = min(self._page, self._max_page())
        self._render()

    def set_filled(self, hero_name: str) -> None:
        """Lane already has an ally pick: gray it out, hide chevrons."""
        self._suggestions = []
        self.prev.setVisible(False)
        self.next.setVisible(False)
        self.tag.setStyleSheet(f"color:{THEME.rgba(THEME.text_dim)};")
        self.value.setText(
            f"<span style='color:{THEME.rgba(THEME.text_dim)}'>"
            f"&#10003; {hero_name} &nbsp;<i>picked</i></span>")

    def _max_page(self) -> int:
        return max(0, len(self._suggestions) - self.PAGE_SIZE)

    def _page_prev(self) -> None:
        self._page = max(0, self._page - 1)
        self._render()

    def _page_next(self) -> None:
        self._page = min(self._max_page(), self._page + 1)
        self._render()

    def _render(self) -> None:
        window = self._suggestions[self._page: self._page + self.PAGE_SIZE]
        if not window:
            self.value.setText("— no candidates —")
        else:
            parts = []
            for i, s in enumerate(window):
                rank = self._page + i + 1
                parts.append(
                    f"<span style='color:{THEME.rgba(THEME.accent)}'><b>{rank}.</b></span>"
                    f"&nbsp;<b style='color:{THEME.rgba(THEME.text)}'>{s.name}</b>"
                    f"<span style='color:{THEME.rgba(THEME.text_dim)}'>"
                    f"&nbsp;{s.score:g}</span>")
            self.value.setText("&nbsp;&nbsp;&nbsp;".join(parts))
        # chevron availability
        self.prev.setEnabled(self._page > 0)
        self.next.setEnabled(self._page < self._max_page())


# ===========================================================================
#  WINDOW 1: click-through painting canvas
# ===========================================================================
class OverlayCanvas(QWidget):
    """Transparent, click-through layer that paints boxes + tags over slots."""

    def __init__(self, db, layout: Optional[config.Layout] = None,
                 region: Optional[dict] = None):
        super().__init__()
        self.db = db
        # Resolve at construction time so a calibrated layout / region is used.
        self.layout = layout if layout is not None else config.LAYOUT
        region = region if region is not None else config.ACTIVE_REGION
        self._state = DraftState()

        self.setWindowFlags(Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint
                            | Qt.Tool
                            | Qt.WindowTransparentForInput)       # click-through
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # belt & braces
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setGeometry(region["left"], region["top"],
                         region["width"], region["height"])

        self._font = QFont(THEME.font_family, 11, QFont.Bold)
        self._font.setStyleHint(QFont.Monospace)

    def set_state(self, state: DraftState) -> None:
        self._state = state
        self.update()

    # --- painting ---------------------------------------------------------
    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setFont(self._font)

        self._paint_picks(p, self.layout.ally_picks, self._state.ally_picks,
                          self._state.ally_lanes, THEME.ally_box,
                          self._state.ally_pick_conf)
        self._paint_picks(p, self.layout.enemy_picks, self._state.enemy_picks,
                          self._state.enemy_lanes, THEME.enemy_box,
                          self._state.enemy_pick_conf)
        # Bans are listed inside the control dock (like the reference video),
        # so we don't draw anything over the game's tiny ban icons here.

    def _paint_picks(self, p, boxes, names, lanes, color, confs=None) -> None:
        pen = QPen(qcolor(color), THEME.box_thickness)
        confs = confs or [0.0] * len(boxes)
        for box, name, conf in zip(boxes, names, confs):
            if not name:
                continue
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(box.x, box.y, box.w, box.h)
            lane = next((ln for ln, hn in lanes.items() if hn == name), None)
            tag = predicted_role_label(name, lane, self.db)
            label = f"{name.upper()} [{tag}]"
            if config.SHOW_CONFIDENCE and conf:
                label += f" {int(round(conf * 100))}%"
            self._paint_label(p, box, label, color)

    def _paint_label(self, p, box, text, color, font=None) -> None:
        font = font or self._font
        p.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text) + 12
        th = fm.height() + 4
        lx = box.x
        ly = box.y - th - 4 if box.y - th - 4 > 0 else box.y2 + 4

        bg = QPainterPath()
        bg.addRoundedRect(QRectF(lx, ly, tw, th), 4, 4)
        p.fillPath(bg, qcolor(THEME.label_bg))
        p.setPen(QPen(qcolor(color), 1))
        p.drawPath(bg)
        p.setPen(QPen(qcolor(THEME.label_fg), 1))
        p.drawText(lx + 6, ly + fm.ascent() + 2, text)
        p.setFont(self._font)


# ===========================================================================
#  CALIBRATION OVERLAY: click anchors directly on the live game (WYSIWYG)
# ===========================================================================
class CalibrationOverlay(QWidget):
    """Transparent but INTERACTIVE full-screen overlay for click-to-calibrate.

    You click the 8 anchor points on the live draft showing through, so the
    coordinates are in the exact space the running overlay paints in (and, with
    DPI awareness on, the same space mss captures).  No separate cv2 window, no
    coordinate-space mismatch.
    """
    saved = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, region: Optional[dict] = None, out_path: Optional[str] = None):
        super().__init__()
        region = region if region is not None else config.ACTIVE_REGION
        self.out_path = out_path or config.LAYOUT_FILE
        self._clicks: List = []
        self._factor = 0.82

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setGeometry(region["left"], region["top"],
                         region["width"], region["height"])
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self._font = QFont(THEME.font_family, 15, QFont.Bold)
        self._font.setStyleHint(QFont.Monospace)

    # ----- input ----------------------------------------------------------
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and len(self._clicks) < 8:
            self._clicks.append((e.x(), e.y()))
            self.update()

    def keyPressEvent(self, e) -> None:
        k = e.key()
        if k == Qt.Key_Escape:
            self.cancelled.emit(); self.close(); return
        if k == Qt.Key_R:
            self._clicks = []; self.update(); return
        if len(self._clicks) == 8:
            if k == Qt.Key_S:
                config.save_layout(self._layout(), self.out_path)
                self.saved.emit(os.path.abspath(self.out_path))
                self.close()
            elif k == Qt.Key_BracketLeft:
                self._factor = max(0.40, self._factor - 0.04); self.update()
            elif k == Qt.Key_BracketRight:
                self._factor = min(1.20, self._factor + 0.04); self.update()

    # ----- geometry -------------------------------------------------------
    def _layout(self):
        from calibrate import layout_from_anchors
        c = self._clicks
        return layout_from_anchors((c[0], c[1]), (c[2], c[3]), (c[4], c[5]),
                                   (c[6], c[7]), self._factor, self._factor)

    # ----- painting -------------------------------------------------------
    def paintEvent(self, _) -> None:
        from calibrate import PROMPTS
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 55))     # dim => visibly active
        p.setFont(self._font)

        for i, (x, y) in enumerate(self._clicks):
            p.setPen(Qt.NoPen); p.setBrush(qcolor(THEME.accent))
            p.drawEllipse(QPoint(x, y), 7, 7)
            p.setPen(QPen(QColor(0, 0, 0)))
            p.drawText(x + 11, y - 9, str(i + 1))

        if len(self._clicks) < 8:
            self._banner(p, f"[{len(self._clicks)+1}/8]  {PROMPTS[len(self._clicks)]}",
                         "Click the slot CENTER   -   R restart   Esc cancel")
        else:
            p.setBrush(Qt.NoBrush)
            for boxes, color in ((self._layout().ally_picks, THEME.ally_box),
                                 (self._layout().enemy_picks, THEME.enemy_box),
                                 (self._layout().ally_bans, THEME.ban_box),
                                 (self._layout().enemy_bans, THEME.ban_box)):
                p.setPen(QPen(qcolor(color), 3))
                for b in boxes:
                    p.drawRect(b.x, b.y, b.w, b.h)
            self._banner(p, "Review the boxes",
                         "S save    [ / ] resize    R restart    Esc cancel")

    def _banner(self, p, line1: str, line2: str) -> None:
        w = self.width()
        p.setBrush(QColor(6, 14, 12, 235))
        p.setPen(QPen(qcolor(THEME.panel_border), 2))
        p.drawRoundedRect(int(w / 2 - 390), 22, 780, 74, 10, 10)
        p.setFont(self._font)
        p.setPen(QPen(qcolor(THEME.accent)))
        p.drawText(int(w / 2 - 366), 56, line1)
        p.setFont(QFont(THEME.font_family, 11))
        p.setPen(QPen(qcolor(THEME.text_dim)))
        p.drawText(int(w / 2 - 366), 84, line2)


# ===========================================================================
#  WINDOW 2: interactive control dock
# ===========================================================================
DOCK_QSS = f"""
QLabel {{ color: {THEME.rgba(THEME.text)}; }}
QLabel#title {{ color: {THEME.rgba(THEME.accent)};
               font-weight: bold; letter-spacing: 3px; }}
QLabel#section {{ color: {THEME.rgba(THEME.accent_dim)}; letter-spacing: 2px; }}
QLabel#laneTag {{ color: {THEME.rgba(THEME.accent)}; font-weight: bold;
                  font-size: 16px; }}
QLabel#suggestion {{ color: {THEME.rgba(THEME.text)}; font-size: 15px; }}
QLabel#section {{ font-size: 14px; }}
QLabel#build {{ color: {THEME.rgba(THEME.accent)};
               background: rgba(0,0,0,0.35); padding: 6px;
               border: 1px solid {THEME.rgba(THEME.accent_dim)};
               border-radius: 4px; }}
QLabel#toggleLabel {{ color: {THEME.rgba(THEME.text_dim)}; }}
QLabel#banTitleAlly {{ color: {THEME.rgba(THEME.accent)}; font-weight: bold; letter-spacing: 1px; }}
QLabel#banTitleEnemy {{ color: {THEME.rgba(THEME.warn)}; font-weight: bold; letter-spacing: 1px; }}
QLabel#banCellAlly {{ color: {THEME.rgba(THEME.text)}; font-size: 11px;
    border: 1px solid {THEME.rgba(THEME.accent_dim)}; border-radius: 3px;
    padding: 2px 1px; background: rgba(18,40,30,0.45); }}
QLabel#banCellEnemy {{ color: {THEME.rgba(THEME.text)}; font-size: 11px;
    border: 1px solid rgba(150,60,60,0.9); border-radius: 3px;
    padding: 2px 1px; background: rgba(44,20,22,0.45); }}
QLabel#banCellAlly[empty="true"], QLabel#banCellEnemy[empty="true"] {{
    color: #4a554f; background: rgba(255,255,255,0.02); }}
QPushButton#chevron {{
    color: {THEME.rgba(THEME.accent)};
    background: rgba(20,40,34,0.6);
    border: 1px solid {THEME.rgba(THEME.accent_dim)};
    border-radius: 5px; font-weight: bold; font-size: 15px;
}}
QPushButton#chevron:hover {{ background: rgba(40,90,70,0.8); }}
QPushButton#chevron:disabled {{ color: #44514c; border-color: #2a322f; }}
QPushButton#closeBtn {{
    color: {THEME.rgba(THEME.warn)}; background: transparent;
    border: none; font-weight: bold; font-size: 16px;
}}
QPushButton#closeBtn:hover {{ color: #ffffff; }}
"""


class ControlDock(QWidget):
    settingsChanged = pyqtSignal(object)        # emits Settings
    closed = pyqtSignal()

    def __init__(self, db, geometry=None):
        super().__init__()
        self.db = db
        self.settings = Settings()
        self._drag_pos: Optional[QPoint] = None
        geometry = geometry if geometry is not None else config.DOCK_GEOMETRY

        self.setWindowFlags(Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint
                            | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setGeometry(*geometry)
        self.setStyleSheet(DOCK_QSS)
        base_font = QFont(THEME.font_family, 11)
        base_font.setStyleHint(QFont.Monospace)
        self.setFont(base_font)

        self._build_ui()

    # ----- construction ---------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 16)
        root.setSpacing(10)

        # --- title bar ---
        title_row = QHBoxLayout()
        title = QLabel("[ AI LINEUP DRAFTER ]")
        title.setObjectName("title")
        f = title.font(); f.setPointSize(14); title.setFont(f)
        close = QPushButton("✕")
        close.setObjectName("closeBtn")
        close.setFixedSize(26, 26)
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self._on_close)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(close)
        root.addLayout(title_row)

        # --- feature toggles ---
        self.tg_lane = ToggleSwitch()
        self.tg_dark = ToggleSwitch()
        self.tg_pool = ToggleSwitch()
        for tg in (self.tg_lane, self.tg_dark, self.tg_pool):
            tg.toggled.connect(self._emit_settings)
        root.addLayout(self._toggle_row("LANE COUNTER MODE", self.tg_lane))
        root.addLayout(self._toggle_row("DARK SYSTEM MODE", self.tg_dark))
        root.addLayout(self._toggle_row("HERO POOL FILTER", self.tg_pool))

        # --- ban rows (like the reference: names listed inside the panel) ---
        self.ally_ban_cells = self._ban_row(root, "ALLY BANS", "Ally")
        self.enemy_ban_cells = self._ban_row(root, "ENEMY BANS", "Enemy")

        root.addWidget(self._section(">> OPTIMAL PICK VECTORS <<"))

        # --- per-lane suggestion rows ---
        self.rows: Dict[str, RoleRow] = {}
        for lane in config.LANES:
            row = RoleRow(lane)
            self.rows[lane] = row
            root.addWidget(row)

        # --- matchup probability gauge ---
        root.addWidget(self._section(">> MATCHUP PROBABILITY <<"))
        gauge_row = QHBoxLayout()
        self.lbl_ally = QLabel("50.0%")
        self.lbl_ally.setStyleSheet(f"color:{THEME.rgba(THEME.prob_good)};font-weight:bold;")
        self.lbl_enemy = QLabel("50.0%")
        self.lbl_enemy.setStyleSheet(f"color:{THEME.rgba(THEME.prob_bad)};font-weight:bold;")
        self.lbl_enemy.setAlignment(Qt.AlignRight)
        self.prob = ProbabilityBar()
        gauge_row.addWidget(self.lbl_ally)
        gauge_row.addWidget(self.prob, 1)
        gauge_row.addWidget(self.lbl_enemy)
        root.addLayout(gauge_row)

        # --- build path feed ---
        self.build = QLabel(">> AWAITING ENEMY PICKS <<")
        self.build.setObjectName("build")
        self.build.setWordWrap(True)
        root.addWidget(self.build)
        root.addStretch(1)

    def _toggle_row(self, text: str, widget: ToggleSwitch) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(text)
        lbl.setObjectName("toggleLabel")
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(widget)
        return row

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section")
        lbl.setAlignment(Qt.AlignCenter)
        return lbl

    def _ban_row(self, root, title: str, side: str) -> List[QLabel]:
        """A 'ALLY BANS'/'ENEMY BANS' label + five name cells, like the video."""
        row = QHBoxLayout()
        row.setSpacing(4)
        t = QLabel(title)
        t.setObjectName(f"banTitle{side}")
        t.setFixedWidth(134)
        row.addWidget(t)
        cells: List[QLabel] = []
        for _ in range(5):
            c = QLabel("—")
            c.setObjectName(f"banCell{side}")
            c.setAlignment(Qt.AlignCenter)
            c.setProperty("empty", "true")
            cells.append(c)
            row.addWidget(c, 1)
        root.addLayout(row)
        return cells

    # ----- behaviour ------------------------------------------------------
    def _emit_settings(self) -> None:
        self.settings = Settings(
            lane_counter_mode=self.tg_lane.isChecked(),
            dark_mode=self.tg_dark.isChecked(),
            hero_pool_filter=self.tg_pool.isChecked(),
        )
        self.settingsChanged.emit(self.settings)

    def _on_close(self) -> None:
        self.closed.emit()
        self.close()

    def apply_result(self, result: DraftResult) -> None:
        for lane, row in self.rows.items():
            if lane in result.filled_lanes:
                row.set_filled(result.filled_lanes[lane])     # picked -> grayed
            else:
                row.set_suggestions(result.suggestions.get(lane, []))
        self._fill_bans(self.ally_ban_cells, result.ally_bans)
        self._fill_bans(self.enemy_ban_cells, result.enemy_bans)
        self.prob.set_values(result.ally_win_pct, result.enemy_win_pct)
        self.lbl_ally.setText(f"{result.ally_win_pct:.1f}%")
        self.lbl_enemy.setText(f"{result.enemy_win_pct:.1f}%")
        self.build.setText(result.build_path)

    @staticmethod
    def _fill_bans(cells: List[QLabel], names: List[Optional[str]]) -> None:
        for i, cell in enumerate(cells):
            name = names[i] if i < len(names) else None
            cell.setText((name or "—").upper())
            cell.setProperty("empty", "false" if name else "true")
            cell.style().unpolish(cell)        # re-apply the [empty] QSS rule
            cell.style().polish(cell)

    # ----- frameless window: paint background + allow dragging ------------
    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(QRectF(r), 12, 12)
        p.fillPath(path, qcolor(THEME.panel_bg))
        p.setPen(QPen(qcolor(THEME.panel_border), 2))
        p.drawPath(path)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e) -> None:
        if self._drag_pos is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPos() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, _) -> None:
        self._drag_pos = None

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape:
            self._on_close()


# ===========================================================================
#  FACADE: ties the two windows together
# ===========================================================================
class DraftOverlay:
    """Convenience wrapper main.py talks to."""

    def __init__(self, db):
        self.db = db
        self.canvas = OverlayCanvas(db)
        self.dock = ControlDock(db)

    @property
    def settingsChanged(self):
        return self.dock.settingsChanged

    @property
    def closed(self):
        return self.dock.closed

    @property
    def settings(self) -> Settings:
        return self.dock.settings

    def show(self) -> None:
        self.canvas.show()
        self.dock.show()
        self.dock.raise_()

    def update_state(self, state: DraftState) -> None:
        self.canvas.set_state(state)

    def update_result(self, result: DraftResult) -> None:
        self.dock.apply_result(result)


# ===========================================================================
#  MOCK DEMO  (no game / capture required)
# ===========================================================================
def _demo():
    import sys
    from engine import HeroDB, ScoringEngine

    app = QApplication(sys.argv)
    db = HeroDB.load("heroes.json")
    engine = ScoringEngine(db)

    state = DraftState(
        ally_picks=["Guinevere", "Zetian", "Minsithar", None, None],
        enemy_picks=["Alice", "Khufra", "Akai", "Gord", None],
        ally_bans=["Harley", "Karrie", "Sora", "Gloo", "Freya"],
        enemy_bans=["Freya", "Gusion", "Yi Sun-shin", "Sora", "Zhuxin"],
        ally_lanes={"EXP": "Guinevere", "MID": "Zetian", "JUNGLE": "Minsithar"},
        enemy_lanes={"JUNGLE": "Alice", "ROAM": "Akai", "EXP": "Khufra", "MID": "Gord"},
    )

    overlay = DraftOverlay(db)

    def recompute(settings: Settings):
        overlay.update_result(engine.evaluate(state, settings))

    overlay.settingsChanged.connect(recompute)
    overlay.closed.connect(app.quit)
    overlay.update_state(state)
    recompute(overlay.settings)
    overlay.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    _demo()
