"""
config.py
=========
Single source of truth for the MLBB drafting overlay.

Everything that is "tunable" lives here:

  * The target capture resolution (2712 x 1220 - the Scrcpy mirror window).
  * The PIXEL-PERFECT bounding boxes for every draft slot (10 picks + 10 bans).
  * Lane <-> base_role mapping used by the role-prediction optimiser.
  * The numeric weights that drive engine.py's scoring maths.
  * The neon colour theme used by ui.py so the overlay matches the
    "Otev-" AI LINEUP DRAFTER aesthetic seen in the reference footage.

The slot boxes are derived from fractional anchors so the layout can be
re-targeted to a different resolution by changing ``RES_W`` / ``RES_H`` only.
Explicit integer boxes for 2712x1220 are exposed via ``build_layout()`` and the
pre-computed ``LAYOUT`` constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. CAPTURE TARGET
# ---------------------------------------------------------------------------
# The Scrcpy mirror is assumed to be a borderless window of EXACTLY this size,
# pinned to the top-left of the primary monitor.  If your mirror lives at a
# different origin, set CAPTURE_ORIGIN to its top-left (x, y) on the desktop.
RES_W: int = 2712
RES_H: int = 1220
CAPTURE_ORIGIN: Tuple[int, int] = (0, 0)

# Capture / detection throttling (seconds).  Drafts move slowly, so ~12 FPS of
# detection is plenty and keeps CPU + GPU cool.  The capturer also caches the
# last frame so the UI thread never blocks on a grab.
CAPTURE_MIN_INTERVAL: float = 0.08          # ~12.5 Hz hard ceiling on grabs
DETECT_INTERVAL: float = 0.10               # detector worker tick

# ---------------------------------------------------------------------------
# 2. LANES & ROLES
# ---------------------------------------------------------------------------
# Canonical draft lanes (the five MLBB positions).
LANES: List[str] = ["EXP", "JUNGLE", "MID", "GOLD", "ROAM"]

# Short labels rendered on the bounding-box tags, e.g. "GUINEVERE [ROA]".
LANE_ABBR: Dict[str, str] = {
    "EXP": "EXP",
    "JUNGLE": "JUG",
    "MID": "MID",
    "GOLD": "GLD",
    "ROAM": "ROA",
}

# base_role -> ordered lane preference.  Index 0 is the most natural lane.
# The role-prediction optimiser (detector.assign_lanes) turns this into a
# numeric preference matrix and finds the globally optimal 1-to-1 assignment.
ROLE_LANE_PRIORITY: Dict[str, List[str]] = {
    "Fighter":  ["EXP", "JUNGLE", "ROAM"],
    "Assassin": ["JUNGLE", "MID", "EXP"],
    "Mage":     ["MID", "ROAM", "GOLD"],
    "Marksman": ["GOLD", "MID"],
    "Tank":     ["ROAM", "EXP"],
    "Support":  ["ROAM", "MID"],
}

# Roles that count as "frontline" for the Team Composition Balancer.
FRONTLINE_ROLES = frozenset({"Tank", "Fighter"})

# ---------------------------------------------------------------------------
# 3. SCORING WEIGHTS  (consumed by engine.py)
# ---------------------------------------------------------------------------
BASELINE_SCORE: float = 10.0        # every hero starts here

W_WINRATE: float = 0.22             # points per win-rate %  above/below 50
W_BANRATE: float = 0.06             # points per ban-rate %  (contested == strong)

W_SYNERGY: float = 2.6              # per allied synergy hit
W_COUNTER: float = 3.1              # per enemy this hero hard-counters
W_COUNTERED: float = 3.6            # penalty per enemy that counters this hero

# "Lane Counter Mode": triple the weight of the DIRECT lane match-up and drop
# broad team-comp synergies entirely.
LANE_COUNTER_MULT: float = 3.0

# "Team Composition Balancer": multiplicative bonus applied to heroes that fill
# a missing need (frontline or magic damage).
COMP_MULTIPLIER: float = 1.25
COMP_MIN_FRONTLINE: int = 1         # want at least this many frontliners
COMP_MIN_MAGIC: int = 1            # want at least this much magic damage

# Logistic steepness for the win-probability bar (the green/red gauge).
MATCHUP_K: float = 0.055

# ---------------------------------------------------------------------------
# 4. DETECTION TUNING (consumed by detector.py)
# ---------------------------------------------------------------------------
TEMPLATE_DIR: str = "templates"             # cropped hero avatars live here
TEMPLATE_MATCH_THRESHOLD: float = 0.62      # TM_CCOEFF_NORMED accept threshold
HISTOGRAM_MATCH_THRESHOLD: float = 0.55     # fallback correlation accept
EMPTY_SLOT_STDDEV: float = 11.0             # below this a slot is treated empty
SIGNATURE_DELTA: int = 6                    # per-slot change threshold for cache

# ---------------------------------------------------------------------------
# 4b. LIVE STATS SOURCE (consumed by stats_provider.py / main.py)
# ---------------------------------------------------------------------------
# Leave STATS_URL = None to run purely from heroes.json.  Set it to a JSON
# endpoint and describe its shape with STATS_RECORD_PATH + STATS_FIELD_MAP to
# overlay live win/ban rates etc.  Nothing site-specific is assumed.
STATS_URL: Optional[str] = None
STATS_RECORD_PATH: Optional[str] = None             # e.g. "data.heroes"
STATS_NAME_KEY: str = "name"
STATS_FIELD_MAP: Dict[str, str] = {}                # our_field -> source_key
STATS_CACHE_PATH: str = ".stats_cache.json"
STATS_CACHE_TTL: float = 3600.0                     # serve cache for 1h
STATS_REFRESH_SEC: int = 900                        # auto-refresh every 15 min

# ---------------------------------------------------------------------------
# 5. PIXEL-PERFECT SLOT LAYOUT
# ---------------------------------------------------------------------------
# A draft slot box: integer pixel rectangle on the capture surface.
@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass(frozen=True)
class Layout:
    """All draft slot boxes for one resolution."""
    ally_picks: List[Box]
    enemy_picks: List[Box]
    ally_bans: List[Box]
    enemy_bans: List[Box]

    @property
    def all_picks(self) -> List[Box]:
        return self.ally_picks + self.enemy_picks

    @property
    def all_bans(self) -> List[Box]:
        return self.ally_bans + self.enemy_bans


def build_layout(res_w: int = RES_W, res_h: int = RES_H) -> Layout:
    """
    Derive every slot rectangle from fractional anchors so the same proportions
    re-scale to any resolution.  Values below were calibrated against the
    2712x1220 MLBB draft screen (allied picks down the LEFT margin, enemy picks
    down the RIGHT margin, all ten ban portraits along the TOP margin).

    NOTE on calibration: ship ``tools/calibrate.py`` (or the in-app F2 grid)
    to nudge these anchors per-device.  Template matching is forgiving of a few
    px of drift, but the tighter the box the higher the confidence score.
    """
    # ---- vertical pick column geometry (fractions of height) --------------
    pick_w = round(0.0560 * res_w)          # ~152 px portrait at 2712
    pick_h = pick_w                          # portraits are square
    pick_y0 = round(0.0820 * res_h)          # first slot top  (~100 px)
    pick_pitch = round(0.1605 * res_h)       # vertical stride (~196 px)

    ally_x = round(0.0130 * res_w)           # left margin     (~35 px)
    enemy_x = res_w - ally_x - pick_w        # mirrored to right margin

    ally_picks = [Box(ally_x, pick_y0 + i * pick_pitch, pick_w, pick_h)
                  for i in range(5)]
    enemy_picks = [Box(enemy_x, pick_y0 + i * pick_pitch, pick_w, pick_h)
                   for i in range(5)]

    # ---- horizontal ban row geometry (fractions of width) -----------------
    ban_w = round(0.0288 * res_w)            # ~78 px ban portrait
    ban_h = ban_w
    ban_y = round(0.0180 * res_h)            # top margin (~22 px)
    ban_pitch = round(0.0553 * res_w)        # horizontal stride (~150 px)

    ally_ban_x0 = round(0.1730 * res_w)      # first allied ban (~469 px)
    ally_bans = [Box(ally_ban_x0 + i * ban_pitch, ban_y, ban_w, ban_h)
                 for i in range(5)]
    # Enemy bans mirror the allied row across the screen's vertical centre.
    enemy_bans = [Box(res_w - (ally_ban_x0 + i * ban_pitch) - ban_w,
                      ban_y, ban_w, ban_h)
                  for i in range(5)]

    return Layout(ally_picks, enemy_picks, ally_bans, enemy_bans)


# Pre-computed layout for the default 2712x1220 target.
LAYOUT: Layout = build_layout()

# ---------------------------------------------------------------------------
# 6. UI THEME  (consumed by ui.py) -- neon "terminal" aesthetic
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    font_family: str = "Consolas"           # falls back to any monospace
    font_px: int = 22

    # RGBA tuples
    ally_box: Tuple[int, int, int, int] = (60, 230, 120, 235)      # green
    enemy_box: Tuple[int, int, int, int] = (235, 70, 70, 235)      # red
    ban_box: Tuple[int, int, int, int] = (150, 150, 160, 150)      # grey
    label_bg: Tuple[int, int, int, int] = (8, 16, 14, 205)
    label_fg: Tuple[int, int, int, int] = (180, 255, 210, 255)

    panel_bg: Tuple[int, int, int, int] = (6, 14, 12, 222)
    panel_border: Tuple[int, int, int, int] = (40, 230, 150, 255)
    accent: Tuple[int, int, int, int] = (60, 255, 170, 255)
    accent_dim: Tuple[int, int, int, int] = (40, 150, 110, 255)
    text: Tuple[int, int, int, int] = (200, 255, 225, 255)
    text_dim: Tuple[int, int, int, int] = (120, 170, 150, 255)
    warn: Tuple[int, int, int, int] = (255, 90, 90, 255)

    prob_good: Tuple[int, int, int, int] = (60, 230, 120, 255)
    prob_bad: Tuple[int, int, int, int] = (235, 70, 70, 255)

    box_thickness: int = 3
    line_thickness: int = 2

    @staticmethod
    def rgba(c: Tuple[int, int, int, int]) -> str:
        return f"rgba({c[0]},{c[1]},{c[2]},{c[3] / 255:.3f})"


THEME = Theme()

# Where the interactive control dock parks itself (centred panel like the
# reference video).  Coordinates are on the desktop, not the capture surface.
DOCK_GEOMETRY: Tuple[int, int, int, int] = (
    round(RES_W * 0.30) + CAPTURE_ORIGIN[0],   # x
    round(RES_H * 0.06) + CAPTURE_ORIGIN[1],   # y
    round(RES_W * 0.40),                        # w
    round(RES_H * 0.66),                        # h
)


if __name__ == "__main__":
    # Quick visual sanity dump of the computed layout.
    L = LAYOUT
    print(f"Resolution {RES_W}x{RES_H}")
    for name, boxes in (("ALLY PICK", L.ally_picks), ("ENEMY PICK", L.enemy_picks),
                        ("ALLY BAN", L.ally_bans), ("ENEMY BAN", L.enemy_bans)):
        for i, b in enumerate(boxes):
            print(f"  {name:10} {i}: {b.as_tuple()}")
