"""
Module 3 — Logic & Scoring Engine
=================================

Turns the current draft state into a ranked set of hero recommendations,
broken down by lane (Gold / EXP / Mid / Jungle / Roam).

Scoring recipe (all weights configurable in ``config.SCORING``)
---------------------------------------------------------------
Every available hero starts at a baseline (10 pts) and is then adjusted:

* **Meta strength**   +/- based on win rate (vs 50%), ban rate and pick rate.
* **Counters**
    * + if a hero that *beats us* is already banned or on our team
      (our counter is neutralised).
    * - if the enemy already picked a hero that *beats us*.
    * + if *we beat* a hero the enemy already picked.
* **Synergy**         + for each ally pick we pair well with.
* **Draft balance**
    * - if our team already has this lane filled (role saturation), which
      naturally boosts the lanes/damage-dealers we still need.
    * + for the damage type (physical/magic) our team currently lacks.
* **Lane-counter mode** (optional toggle)
    * predicts each enemy hero's lane and *heavily* boosts candidates that are
      a direct, same-lane hard counter to an enemy laner.

Every recommendation carries a full ``ScoreBreakdown`` so the overlay (or a
curious human) can see *why* a hero was suggested.

Pure standard-library — safe to import without OpenCV / PyQt5.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from src.data_manager import DataManager, Hero


# --------------------------------------------------------------------------- #
# Draft state (shared with the vision module)
# --------------------------------------------------------------------------- #
@dataclass
class DraftState:
    """A snapshot of the draft at a moment in time."""

    ally_picks: List[str] = field(default_factory=list)
    enemy_picks: List[str] = field(default_factory=list)
    bans: List[str] = field(default_factory=list)
    # Optional manual lane overrides for enemy heroes, e.g. {"Fanny": "jungle"}.
    # When empty, the engine predicts lanes from each hero's primary lane.
    enemy_lane_overrides: Dict[str, str] = field(default_factory=dict)

    def taken(self) -> set:
        """Every hero that is no longer pickable."""
        return set(self.ally_picks) | set(self.enemy_picks) | set(self.bans)

    def neutralised(self) -> set:
        """Heroes removed from the enemy's reach (banned or on our team)."""
        return set(self.ally_picks) | set(self.bans)


# --------------------------------------------------------------------------- #
# Score breakdown
# --------------------------------------------------------------------------- #
@dataclass
class ScoreBreakdown:
    """The final score for one hero plus a labelled component breakdown."""

    hero: str
    lane: str
    total: float
    components: Dict[str, float] = field(default_factory=dict)

    def add(self, label: str, value: float) -> None:
        if value:
            self.components[label] = round(self.components.get(label, 0.0) + value, 2)
            self.total = round(self.total + value, 2)

    def explain(self) -> str:
        parts = [f"{k}: {v:+.1f}" for k, v in self.components.items()]
        return f"{self.hero} ({self.lane}) = {self.total:.1f}  [" + ", ".join(parts) + "]"


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class ScoringEngine:
    def __init__(
        self,
        data_manager: DataManager,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.dm = data_manager
        self.w = dict(config.SCORING)
        if weights:
            self.w.update(weights)

    # ---- individual scoring components -------------------------------- #
    def _meta_component(self, hero: Hero, bd: ScoreBreakdown) -> None:
        bd.add("win_rate", (hero.win_rate - 0.50) * self.w["win_rate_weight"])
        bd.add("ban_rate", hero.ban_rate * self.w["ban_rate_weight"])
        bd.add("pick_rate", hero.pick_rate * self.w["pick_rate_weight"])

    def _counter_component(
        self, hero: Hero, draft: DraftState, bd: ScoreBreakdown
    ) -> None:
        neutralised = draft.neutralised()
        enemy = set(draft.enemy_picks)

        # Our threats that are banned or already on our team.
        for threat in hero.weak_against:
            if threat in neutralised:
                bd.add("counter_neutralised", self.w["counter_neutralized"])

        # Enemy already fielded a hero we are weak against.
        for threat in hero.weak_against:
            if threat in enemy:
                bd.add("countered_by_enemy", -self.w["countered_by_enemy"])

        # We are strong against a hero the enemy already picked.
        for victim in hero.strong_against:
            if victim in enemy:
                bd.add("we_counter_enemy", self.w["we_counter_enemy"])

    def _synergy_component(
        self, hero: Hero, draft: DraftState, bd: ScoreBreakdown
    ) -> None:
        allies = set(draft.ally_picks)
        for partner in hero.synergies:
            if partner in allies:
                bd.add("synergy", self.w["synergy"])

    def _balance_component(
        self, hero: Hero, draft: DraftState, bd: ScoreBreakdown
    ) -> None:
        ally_heroes = [self.dm.get_hero(n) for n in draft.ally_picks]
        ally_heroes = [h for h in ally_heroes if h is not None]

        # --- Role / lane saturation -------------------------------------
        filled_lanes = Counter(h.primary_lane for h in ally_heroes)
        if filled_lanes.get(hero.primary_lane, 0) >= config.IDEAL_COMP.get(
            hero.primary_lane, 1
        ):
            # This lane is already covered: penalise piling on more of it.
            bd.add("role_saturation", -self.w["role_saturation"])

        # --- Physical / magic damage balance ----------------------------
        if ally_heroes:
            phys = magic = 0.0
            for h in ally_heroes:
                if h.damage_type == "physical":
                    phys += 1
                elif h.damage_type == "magic":
                    magic += 1
                else:  # mixed counts toward both
                    phys += 0.5
                    magic += 0.5

            if hero.damage_type in ("physical", "magic"):
                lacking = "magic" if magic < phys else "physical"
                surplus_gap = abs(phys - magic)
                if hero.damage_type == lacking and surplus_gap >= 1:
                    bd.add("damage_balance", self.w["damage_imbalance"])
                elif hero.damage_type != lacking and surplus_gap >= 2:
                    # Team is already heavily skewed toward this type.
                    bd.add("damage_balance", -self.w["damage_imbalance"] * 0.5)

    def _predict_enemy_lanes(self, draft: DraftState) -> Dict[str, str]:
        """Map each enemy hero -> the lane we expect them in."""
        lanes: Dict[str, str] = {}
        used: set = set()
        for name in draft.enemy_picks:
            if name in draft.enemy_lane_overrides:
                lanes[name] = draft.enemy_lane_overrides[name]
                used.add(lanes[name])
                continue
            hero = self.dm.get_hero(name)
            if not hero:
                continue
            # Prefer an unclaimed lane this hero can play, else its primary.
            chosen = next((ln for ln in hero.lanes if ln not in used), hero.primary_lane)
            lanes[name] = chosen
            used.add(chosen)
        return lanes

    def _lane_counter_component(
        self,
        hero: Hero,
        lane: str,
        enemy_lanes: Dict[str, str],
        bd: ScoreBreakdown,
    ) -> None:
        """Boost candidates that hard-counter the enemy in the *same* lane."""
        for enemy_name, enemy_lane in enemy_lanes.items():
            if enemy_lane == lane and enemy_name in hero.strong_against:
                bd.add("lane_counter", self.w["lane_counter_bonus"])

    # ---- public API ---------------------------------------------------- #
    def score_hero(
        self,
        hero: Hero,
        draft: DraftState,
        lane: Optional[str] = None,
        lane_counter: bool = False,
        enemy_lanes: Optional[Dict[str, str]] = None,
    ) -> ScoreBreakdown:
        """Score a single hero for a given lane (defaults to its primary)."""
        lane = lane or hero.primary_lane
        bd = ScoreBreakdown(hero=hero.name, lane=lane, total=0.0)
        bd.add("baseline", self.w["baseline"])
        self._meta_component(hero, bd)
        self._counter_component(hero, draft, bd)
        self._synergy_component(hero, draft, bd)
        self._balance_component(hero, draft, bd)
        if lane_counter:
            if enemy_lanes is None:
                enemy_lanes = self._predict_enemy_lanes(draft)
            self._lane_counter_component(hero, lane, enemy_lanes, bd)
        return bd

    def recommend(
        self,
        draft: DraftState,
        lane_counter: bool = False,
        dark_system: bool = False,
        per_lane: int = config.SUGGESTIONS_PER_LANE,
    ) -> Dict[str, List[ScoreBreakdown]]:
        """
        Produce ranked recommendations for every lane.

        Returns ``{lane: [ScoreBreakdown, ...]}`` sorted best-first
        (or *worst*-first when ``dark_system`` is on — the joke "Dark System"
        that recommends the statistically worst available heroes).
        """
        available = self.dm.available_heroes(draft.taken())
        enemy_lanes = self._predict_enemy_lanes(draft) if lane_counter else {}

        buckets: Dict[str, List[ScoreBreakdown]] = {ln: [] for ln in config.LANES}
        for hero in available:
            # A flex hero appears in every lane it can play, scored per lane
            # (the lane affects the lane-counter bonus).
            for lane in hero.lanes:
                if lane not in buckets:
                    continue
                bd = self.score_hero(
                    hero,
                    draft,
                    lane=lane,
                    lane_counter=lane_counter,
                    enemy_lanes=enemy_lanes,
                )
                buckets[lane].append(bd)

        for lane, scores in buckets.items():
            scores.sort(key=lambda b: b.total, reverse=not dark_system)
            buckets[lane] = scores[:per_lane]
        return buckets


# --------------------------------------------------------------------------- #
# Manual smoke test:  python -m src.scoring_engine
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    dm = DataManager().load()
    engine = ScoringEngine(dm)

    draft = DraftState(
        ally_picks=["Tigreal"],            # we already have a roam/tank
        enemy_picks=["Layla", "Esmeralda"],  # enemy gold + exp
        bans=["Fanny", "Ling", "Valentina", "Joy"],
    )

    print("=== Normal recommendations (Mythical Glory) ===")
    recs = engine.recommend(draft, lane_counter=False)
    for lane in config.LANES:
        top = recs[lane][:3]
        line = ", ".join(f"{b.hero}({b.total:.0f})" for b in top)
        print(f"  {config.LANE_LABELS[lane]:<7}: {line}")

    print("\n=== Lane-Counter mode ON ===")
    recs = engine.recommend(draft, lane_counter=True)
    for lane in config.LANES:
        top = recs[lane][:3]
        line = ", ".join(f"{b.hero}({b.total:.0f})" for b in top)
        print(f"  {config.LANE_LABELS[lane]:<7}: {line}")

    print("\n=== DARK SYSTEM (worst picks) ===")
    recs = engine.recommend(draft, dark_system=True)
    for lane in config.LANES:
        top = recs[lane][:3]
        line = ", ".join(f"{b.hero}({b.total:.0f})" for b in top)
        print(f"  {config.LANE_LABELS[lane]:<7}: {line}")

    print("\nExample breakdown for top Gold pick:")
    print("  " + recs["gold"][0].explain())
