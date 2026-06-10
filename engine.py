"""
engine.py
=========
The drafting brain.  Pure, side-effect-free maths so it is trivial to unit
test and cheap to re-run every time the detector reports a new board state.

Pipeline
--------
    HeroDB.load("heroes.json")          # static data
    state  = DraftState(...)            # what the detector saw
    set    = Settings(...)              # toggles from the UI
    engine = ScoringEngine(db)
    result = engine.evaluate(state, set)
    #   result.suggestions[lane] -> [Suggestion, ...]   (ranked, best first)
    #   result.comp                                      (CompAnalysis)
    #   result.ally_win_pct / result.enemy_win_pct       (the gauge)
    #   result.build_path                                (the bottom feed)

Scoring formula (per candidate hero, for a given lane)
------------------------------------------------------
    score  = BASELINE                                            (10)
    score += W_WINRATE * (win_rate - 50)                         global form
    score += W_BANRATE * ban_rate                                contested == strong
    score += W_SYNERGY * <#allied synergy hits>                  team chemistry
    score += W_COUNTER  * <#enemies hard-countered>              offence
    score -= W_COUNTERED * <#enemies that counter us>            defence
    if comp lacks frontline and hero is frontline:  score *= COMP_MULTIPLIER
    if comp lacks magic     and hero is Magical:     score *= COMP_MULTIPLIER

Toggles
-------
    Lane Counter Mode : synergy term dropped; only the DIRECT lane opponent's
                        counter relationship counts, multiplied by 3x.
    Dark System Mode  : score = -score  (surfaces the statistically WORST picks).
    Hero Pool Filter  : candidate set restricted to heroes flagged owned == True.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import config


# ===========================================================================
#  DATA MODEL
# ===========================================================================
@dataclass(frozen=True)
class Hero:
    name: str
    base_role: str
    win_rate: float
    ban_rate: float
    counters: Tuple[str, ...]          # heroes THIS hero beats
    countered_by: Tuple[str, ...]      # heroes that beat THIS hero
    synergies: Tuple[str, ...]
    damage_type: str                   # Physical | Magical | True
    archetypes: Tuple[str, ...]        # subset of Sustain | Burst | Utility
    owned: bool = True
    lanes: Tuple[str, ...] = ()        # actual lanes this hero plays (data-driven)
    roles: Tuple[str, ...] = ()        # all roles (dual-role heroes keep both)

    @property
    def is_frontline(self) -> bool:
        roles = self.roles or (self.base_role,)
        return any(r in config.FRONTLINE_ROLES for r in roles)

    @property
    def is_magic(self) -> bool:
        return self.damage_type == "Magical"


class HeroDB:
    """Case-insensitive hero registry loaded from heroes.json."""

    def __init__(self, heroes: Sequence[Hero]):
        self._by_key: Dict[str, Hero] = {self._key(h.name): h for h in heroes}

    # ----- construction ----------------------------------------------------
    @classmethod
    def load(cls, path: str = "heroes.json") -> "HeroDB":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        raw = payload["heroes"] if isinstance(payload, dict) else payload
        heroes = []
        for d in raw:
            heroes.append(Hero(
                name=d["name"],
                base_role=d.get("base_role", "Fighter"),
                win_rate=float(d.get("win_rate", 50.0)),
                ban_rate=float(d.get("ban_rate", 0.0)),
                counters=tuple(d.get("counters", [])),
                countered_by=tuple(d.get("countered_by", [])),
                synergies=tuple(d.get("synergies", [])),
                damage_type=d.get("damage_type", "Physical"),
                archetypes=tuple(d.get("archetypes", [])),
                owned=bool(d.get("owned", True)),
                lanes=tuple(d.get("lanes", [])),
                roles=tuple(d.get("roles") or ([d["base_role"]] if d.get("base_role") else [])),
            ))
        return cls(heroes)

    # ----- lookups ---------------------------------------------------------
    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower()

    def get(self, name: Optional[str]) -> Optional[Hero]:
        if not name:
            return None
        return self._by_key.get(self._key(name))

    def all(self) -> List[Hero]:
        return list(self._by_key.values())

    def names(self) -> List[str]:
        return [h.name for h in self._by_key.values()]

    def candidates_for_lane(self, lane: str, owned_only: bool = False) -> List[Hero]:
        out = []
        for h in self._by_key.values():
            # Prefer the hero's actual lanes; fall back to role-based priority.
            lanes = h.lanes or tuple(config.ROLE_LANE_PRIORITY.get(h.base_role, []))
            if lane in lanes:
                if owned_only and not h.owned:
                    continue
                out.append(h)
        return out

    # ----- live overlay ----------------------------------------------------
    _SCALAR_FIELDS = ("base_role", "damage_type")
    _NUMERIC_FIELDS = ("win_rate", "ban_rate")
    _LIST_FIELDS = ("counters", "countered_by", "synergies", "archetypes")

    def apply_updates(self, updates: Dict[str, dict]) -> int:
        """Overlay partial stat updates (e.g. from a live StatsProvider) onto
        existing heroes, keyed by name (case-insensitive).

        ``Hero`` is a frozen dataclass, so each touched hero is rebuilt via
        ``dataclasses.replace`` *in place* inside this DB - every object that
        already holds a reference to this HeroDB therefore sees the new numbers
        without needing to be re-wired.  Unknown heroes and unknown fields are
        ignored.  Returns the number of heroes actually changed.
        """
        updated = 0
        for raw_name, fields in (updates or {}).items():
            key = self._key(raw_name)
            hero = self._by_key.get(key)
            if hero is None or not isinstance(fields, dict):
                continue
            clean: Dict[str, object] = {}
            for f in self._NUMERIC_FIELDS:
                if fields.get(f) is not None:
                    try:
                        clean[f] = float(fields[f])
                    except (TypeError, ValueError):
                        pass
            for f in self._SCALAR_FIELDS:
                if fields.get(f):
                    clean[f] = str(fields[f])
            if "owned" in fields:
                clean["owned"] = bool(fields["owned"])
            for f in self._LIST_FIELDS:
                if fields.get(f) is not None:
                    clean[f] = tuple(fields[f])
            if clean:
                self._by_key[key] = replace(hero, **clean)
                updated += 1
        return updated


@dataclass
class DraftState:
    """Snapshot of the board as understood by the detector.

    Lists are length-5; ``None`` marks an empty / undetected slot.
    ``ally_lanes`` / ``enemy_lanes`` map a lane -> hero name once the
    role-prediction optimiser has run (filled in by detector.assign_lanes).
    """
    ally_picks: List[Optional[str]] = field(default_factory=lambda: [None] * 5)
    enemy_picks: List[Optional[str]] = field(default_factory=lambda: [None] * 5)
    ally_bans: List[Optional[str]] = field(default_factory=lambda: [None] * 5)
    enemy_bans: List[Optional[str]] = field(default_factory=lambda: [None] * 5)
    ally_lanes: Dict[str, str] = field(default_factory=dict)
    enemy_lanes: Dict[str, str] = field(default_factory=dict)
    # Per-slot match confidence (0..1) parallel to the pick/ban lists - shown on
    # the overlay so detection quality is visible for tuning.
    ally_pick_conf: List[float] = field(default_factory=lambda: [0.0] * 5)
    enemy_pick_conf: List[float] = field(default_factory=lambda: [0.0] * 5)
    ally_ban_conf: List[float] = field(default_factory=lambda: [0.0] * 5)
    enemy_ban_conf: List[float] = field(default_factory=lambda: [0.0] * 5)
    # Per-slot "occupied but not locked in yet" flags (grayed/hovered portraits).
    # The detector sets these so the overlay can label the slot "NOT PICKED"
    # instead of guessing a hero from a desaturated avatar.
    ally_pending: List[bool] = field(default_factory=lambda: [False] * 5)
    enemy_pending: List[bool] = field(default_factory=lambda: [False] * 5)

    def ally_names(self) -> List[str]:
        return [n for n in self.ally_picks if n]

    def enemy_names(self) -> List[str]:
        return [n for n in self.enemy_picks if n]

    def taken(self) -> set:
        """All hero names already unavailable (picked or banned, both teams)."""
        out = set()
        for lst in (self.ally_picks, self.enemy_picks, self.ally_bans, self.enemy_bans):
            out.update(self._key(n) for n in lst if n)
        return out

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower()

    def open_lanes(self) -> List[str]:
        """Ally lanes that have not been filled yet."""
        return [ln for ln in config.LANES if ln not in self.ally_lanes]


@dataclass
class Settings:
    lane_counter_mode: bool = False
    dark_mode: bool = False
    hero_pool_filter: bool = False


# ===========================================================================
#  ANALYSIS RESULTS
# ===========================================================================
@dataclass
class Suggestion:
    hero: Hero
    score: float
    reasons: List[str] = field(default_factory=list)   # human-readable "why"

    @property
    def name(self) -> str:
        return self.hero.name


@dataclass
class CompAnalysis:
    ally_frontline: int
    ally_magic: int
    needs_frontline: bool
    needs_magic: bool
    enemy_phys: int
    enemy_mag: int
    enemy_true: int
    enemy_frontline: int
    enemy_burst: int


@dataclass
class DraftResult:
    suggestions: Dict[str, List[Suggestion]]
    comp: CompAnalysis
    ally_win_pct: float
    enemy_win_pct: float
    build_path: str
    filled_lanes: Dict[str, str] = field(default_factory=dict)   # lane -> ally hero
    ally_bans: List[Optional[str]] = field(default_factory=list)
    enemy_bans: List[Optional[str]] = field(default_factory=list)


# ===========================================================================
#  ENGINE
# ===========================================================================
class ScoringEngine:
    def __init__(self, db: HeroDB):
        self.db = db

    # -------------------------------------------------- composition analysis
    def analyze_comp(self, state: DraftState) -> CompAnalysis:
        ally = [self.db.get(n) for n in state.ally_names()]
        ally = [h for h in ally if h]
        enemy = [self.db.get(n) for n in state.enemy_names()]
        enemy = [h for h in enemy if h]

        ally_frontline = sum(1 for h in ally if h.is_frontline)
        ally_magic = sum(1 for h in ally if h.is_magic)

        return CompAnalysis(
            ally_frontline=ally_frontline,
            ally_magic=ally_magic,
            needs_frontline=ally_frontline < config.COMP_MIN_FRONTLINE,
            needs_magic=ally_magic < config.COMP_MIN_MAGIC,
            enemy_phys=sum(1 for h in enemy if h.damage_type == "Physical"),
            enemy_mag=sum(1 for h in enemy if h.damage_type == "Magical"),
            enemy_true=sum(1 for h in enemy if h.damage_type == "True"),
            enemy_frontline=sum(1 for h in enemy if h.is_frontline),
            enemy_burst=sum(1 for h in enemy if "Burst" in h.archetypes),
        )

    # ------------------------------------------------------------ scoring
    def score_hero(self, hero: Hero, lane: str, state: DraftState,
                   settings: Settings, comp: CompAnalysis) -> Suggestion:
        """Compute a single hero's score for a single lane.  Returns a
        Suggestion carrying the score plus a short reason trail for the UI."""
        reasons: List[str] = []
        score = config.BASELINE_SCORE

        # --- global form -------------------------------------------------
        wr_term = config.W_WINRATE * (hero.win_rate - 50.0)
        score += wr_term
        if abs(wr_term) >= 0.5:
            reasons.append(f"WR {hero.win_rate:.1f}%")
        score += config.W_BANRATE * hero.ban_rate

        ally_names = state.ally_names()
        enemy_names = state.enemy_names()

        # --- synergy (suppressed in Lane Counter Mode) -------------------
        if not settings.lane_counter_mode:
            syn_hits = 0
            for an in ally_names:
                ah = self.db.get(an)
                if ah and (an in hero.synergies or hero.name in ah.synergies):
                    syn_hits += 1
            if syn_hits:
                score += config.W_SYNERGY * syn_hits
                reasons.append(f"+{syn_hits} synergy")

        # --- counters / countered-by ------------------------------------
        offence = defence = 0
        for en in enemy_names:
            eh = self.db.get(en)
            if not eh:
                continue
            mult = 1.0
            if settings.lane_counter_mode:
                # Only the direct opponent in THIS lane matters; broad picks
                # are ignored, and the relationship is amplified 3x.
                if state.enemy_lanes.get(lane) != en:
                    continue
                mult = config.LANE_COUNTER_MULT
            if en in hero.counters:
                score += config.W_COUNTER * mult
                offence += 1
            if en in hero.countered_by:
                score -= config.W_COUNTERED * mult
                defence += 1
        if offence:
            reasons.append(f"counters {offence}")
        if defence:
            reasons.append(f"weak vs {defence}")

        # --- team-composition balancer (multiplicative) -----------------
        if comp.needs_frontline and hero.is_frontline:
            score *= config.COMP_MULTIPLIER
            reasons.append("fills frontline")
        if comp.needs_magic and hero.is_magic:
            score *= config.COMP_MULTIPLIER
            reasons.append("fills magic dmg")

        # --- ban relief: a hero that COUNTERS us being banned is a threat
        #     removed, so this candidate gets safer (uses both teams' bans).
        banned = {DraftState._key(b) for lst in (state.ally_bans, state.enemy_bans)
                  for b in lst if b}
        if banned:
            relief = sum(1 for c in hero.countered_by if DraftState._key(c) in banned)
            if relief:
                score += config.W_BAN_RELIEF * relief
                reasons.append(f"{relief} threat banned")

        # --- Dark System Mode: invert the universe ----------------------
        if settings.dark_mode:
            score = -score

        return Suggestion(hero=hero, score=round(score, 2), reasons=reasons)

    # ------------------------------------------------------ per-lane ranking
    def suggest_for_lane(self, lane: str, state: DraftState,
                         settings: Settings, comp: CompAnalysis) -> List[Suggestion]:
        taken = state.taken()
        out: List[Suggestion] = []
        for hero in self.db.candidates_for_lane(lane, owned_only=settings.hero_pool_filter):
            if DraftState._key(hero.name) in taken:
                continue
            out.append(self.score_hero(hero, lane, state, settings, comp))
        out.sort(key=lambda s: s.score, reverse=True)
        return out

    # ---------------------------------------------------------- win gauge
    def matchup_probability(self, state: DraftState) -> Tuple[float, float]:
        """Logistic estimate of ally win % from net stat + counter advantage."""
        def side_strength(picks: List[str], foes: List[str]) -> float:
            adv = 0.0
            for n in picks:
                h = self.db.get(n)
                if not h:
                    continue
                adv += (h.win_rate - 50.0)
                for f in foes:
                    if f in h.counters:
                        adv += 4.0
                    if f in h.countered_by:
                        adv -= 4.0
            return adv

        ally = state.ally_names()
        enemy = state.enemy_names()
        if not ally and not enemy:
            return 50.0, 50.0
        diff = side_strength(ally, enemy) - side_strength(enemy, ally)
        ally_pct = 100.0 / (1.0 + math.exp(-config.MATCHUP_K * diff))
        ally_pct = max(1.0, min(99.0, ally_pct))
        return round(ally_pct, 1), round(100.0 - ally_pct, 1)

    # ---------------------------------------------------------- build feed
    def build_path(self, comp: CompAnalysis, enemy_count: int) -> str:
        if enemy_count == 0:
            return ">> AWAITING ENEMY PICKS <<"

        if comp.enemy_frontline >= 3:
            head = "GO FULL DAMAGE"          # break a beefy frontline
        elif comp.enemy_burst >= 3:
            head = "GO FULL TANK"            # survive the burst
        else:
            head = "GO HYBRID"

        split = f"ENEMY DMG {comp.enemy_mag}M / {comp.enemy_phys}P / {comp.enemy_true}T"
        if comp.enemy_mag > comp.enemy_phys:
            hint = "stack MAGIC DEF"
        elif comp.enemy_phys > comp.enemy_mag:
            hint = "stack PHYS ARMOR"
        else:
            hint = "mixed resist"
        return f"{head}  |  {split}  ->  {hint}"

    # ---------------------------------------------------------- top-level
    def evaluate(self, state: DraftState, settings: Settings) -> DraftResult:
        comp = self.analyze_comp(state)
        # Lanes already taken by an ally pick are "filled" - we don't recommend
        # for them (only open lanes get optimal-pick vectors).
        filled = dict(state.ally_lanes)
        suggestions = {}
        for lane in config.LANES:
            if lane in filled:
                suggestions[lane] = []                  # picked -> no suggestions
            else:
                suggestions[lane] = self.suggest_for_lane(lane, state, settings, comp)
        ally_pct, enemy_pct = self.matchup_probability(state)
        return DraftResult(
            suggestions=suggestions,
            comp=comp,
            ally_win_pct=ally_pct,
            enemy_win_pct=enemy_pct,
            build_path=self.build_path(comp, len(state.enemy_names())),
            filled_lanes=filled,
            ally_bans=list(state.ally_bans),
            enemy_bans=list(state.enemy_bans),
        )


# ===========================================================================
#  SELF-TEST  (reproduces the reference screenshot scenario)
# ===========================================================================
if __name__ == "__main__":
    db = HeroDB.load("heroes.json")
    eng = ScoringEngine(db)

    # Ally: Guinevere (EXP), Zetian (MID), Minsithar (EXP).  Enemy: Alice,
    # Khufra, Akai, Gord.  Bans roughly per the screenshot.
    state = DraftState(
        ally_picks=["Guinevere", "Zetian", "Minsithar", None, None],
        enemy_picks=["Alice", "Khufra", "Akai", "Gord", None],
        ally_bans=["Harley", "Karrie", "Sora", "Gloo", "Freya"],
        enemy_bans=["Freya", "Gusion", "Yi Sun-shin", "Sora", "Zhuxin"],
        ally_lanes={"EXP": "Guinevere", "JUNGLE": "Minsithar"},   # MID left open
        enemy_lanes={"ROAM": "Khufra", "MID": "Gord", "EXP": "Akai", "JUNGLE": "Alice"},
    )

    for mode_name, st in (("DEFAULT", Settings()),
                          ("LANE COUNTER", Settings(lane_counter_mode=True)),
                          ("DARK SYSTEM", Settings(dark_mode=True)),
                          ("POOL FILTER", Settings(hero_pool_filter=True))):
        res = eng.evaluate(state, st)
        print(f"\n===== {mode_name} =====")
        print(f"Gauge: ALLY {res.ally_win_pct}%  /  ENEMY {res.enemy_win_pct}%")
        print(f"Build: {res.build_path}")
        print(f"Comp : front={res.comp.ally_frontline} magic={res.comp.ally_magic} "
              f"needs_front={res.comp.needs_frontline} needs_magic={res.comp.needs_magic}")
        for lane in config.LANES:
            top = res.suggestions[lane][:3]
            shown = ", ".join(f"{s.name}({s.score})" for s in top)
            print(f"  {lane:7}: {shown}")
