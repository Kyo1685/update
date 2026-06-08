"""
Central configuration for the MLBB Draft Assistant.

Everything tunable lives here: file paths, the five draft lanes, the default
rank filter, the scoring weights for the logic engine, and the screen-capture
regions for the vision module.

IMPORTANT: the pixel coordinates in ``REGIONS`` are placeholders for a
1920x1080 display. They *must* be calibrated to your own resolution and the
MLBB draft-screen layout before the vision module will work. Use
``tools/calibrate.py`` to find the right values.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HERO_STATS_PATH = DATA_DIR / "hero_stats.json"
PORTRAITS_DIR = DATA_DIR / "portraits"

# --------------------------------------------------------------------------- #
# The five MLBB draft lanes / role buckets
# --------------------------------------------------------------------------- #
LANES = ["gold", "exp", "mid", "jungle", "roam"]
LANE_LABELS = {
    "gold": "Gold",
    "exp": "EXP",
    "mid": "Mid",
    "jungle": "Jungle",
    "roam": "Roam",
}

# Standard 1-of-each composition target. Used by the draft-balancing logic to
# decide which lanes are already "filled".
IDEAL_COMP = {lane: 1 for lane in LANES}

# --------------------------------------------------------------------------- #
# Rank filtering
# --------------------------------------------------------------------------- #
# Ranks understood by the data manager. A hero may only carry stats for a
# subset of these; missing ranks fall back to "all".
RANK_TIERS = ["all", "epic", "legend", "mythic", "mythical_glory"]
DEFAULT_RANK = "mythical_glory"

# --------------------------------------------------------------------------- #
# Scoring weights (Module 3) — tweak freely to change the engine's "taste".
# --------------------------------------------------------------------------- #
SCORING = {
    "baseline": 10.0,            # every hero starts here
    # Meta strength
    "win_rate_weight": 40.0,     # multiplied by (win_rate - 0.50)
    "ban_rate_weight": 8.0,      # multiplied by ban_rate (0-1)
    "pick_rate_weight": 4.0,     # multiplied by pick_rate (0-1)
    # Counter relationships
    "counter_neutralized": 3.0,  # a hero that beats us is banned / on our team
    "countered_by_enemy": 6.0,   # the enemy already has a hero that beats us
    "we_counter_enemy": 5.0,     # we are strong against an enemy pick
    "synergy": 2.5,              # an ally pick synergises with us
    # Draft balancing
    "role_saturation": 7.0,      # our team already has this lane filled
    "damage_imbalance": 6.0,     # bonus for the damage type the team lacks
    # Lane-counter mode
    "lane_counter_bonus": 12.0,  # direct same-lane hard counter (mode only)
}

# How many suggestions to keep per lane (the overlay's "Next" button pages
# through this list).
SUGGESTIONS_PER_LANE = 8

# --------------------------------------------------------------------------- #
# Vision / screen capture (Module 2)
# --------------------------------------------------------------------------- #
MATCH_THRESHOLD = 0.70   # template-match confidence cutoff, 0.0-1.0
CAPTURE_MONITOR = 1      # mss monitor index (1 = primary screen)
SCAN_INTERVAL_S = 1.0    # seconds between draft scans in the live loop

# Slot rectangles as [left, top, width, height] in screen pixels.
# PLACEHOLDERS for 1920x1080 — calibrate with tools/calibrate.py.
#
# Typical MLBB draft layout:
#   - bans run along the very top (allies left, enemies right)
#   - your team's picks stack down the left edge
#   - the enemy team's picks stack down the right edge
def _row(left, top, w, h, count, dx=0, dy=0):
    """Helper: generate ``count`` slot rects stepping by (dx, dy)."""
    return [[left + dx * i, top + dy * i, w, h] for i in range(count)]


REGIONS = {
    # 5 ally picks down the left side
    "ally_picks": _row(40, 230, 90, 90, 5, dy=150),
    # 5 enemy picks down the right side
    "enemy_picks": _row(1790, 230, 90, 90, 5, dy=150),
    # 5 ally bans along the top-left
    "ally_bans": _row(360, 70, 60, 60, 5, dx=80),
    # 5 enemy bans along the top-right
    "enemy_bans": _row(1080, 70, 60, 60, 5, dx=80),
}
