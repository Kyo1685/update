"""
Module 1 — Data Scraper & Manager
=================================

Loads, structures and serves MLBB hero meta data (win/ban/pick rates plus
counter/synergy relationships) and exposes it as convenient ``Hero`` objects.

Key features
------------
* Reads from a local JSON file (``data/hero_stats.json``).
* Resolves stats for a *specific rank* (e.g. ``mythical_glory``) with graceful
  fallback to the ``all`` bucket when a hero has no data for that rank.
* ``scrape_web()`` is a drop-in mock of a real web scraper — it documents
  exactly where you'd plug in ``requests`` + ``BeautifulSoup`` and, by default,
  just returns the bundled dataset so the rest of the app keeps working offline.

This module only uses the standard library, so it can be imported and unit
tested without OpenCV / PyQt5 installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import config


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Hero:
    """A single hero with stats already resolved for one rank."""

    name: str
    roles: List[str]
    lanes: List[str]
    damage_type: str            # 'physical' | 'magic' | 'mixed'
    win_rate: float
    ban_rate: float
    pick_rate: float
    strong_against: List[str]   # heroes this hero beats
    weak_against: List[str]     # heroes that beat this hero
    synergies: List[str]        # heroes that pair well with this hero

    @property
    def primary_lane(self) -> str:
        """The lane the hero is most commonly played in."""
        return self.lanes[0] if self.lanes else "roam"

    @property
    def meta_strength(self) -> float:
        """A quick 0-1-ish blend of win and ban rate; handy for sorting."""
        return (self.win_rate - 0.5) + self.ban_rate * 0.5


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class DataManager:
    """Loads the raw stats file and serves rank-resolved ``Hero`` objects."""

    def __init__(
        self,
        stats_path: Path | str = config.HERO_STATS_PATH,
        rank: str = config.DEFAULT_RANK,
    ) -> None:
        self.stats_path = Path(stats_path)
        self.rank = rank
        self._raw: Dict = {}
        self._heroes: Dict[str, Hero] = {}

    # -- loading ---------------------------------------------------------- #
    def load(self) -> "DataManager":
        """Read the JSON file from disk and build the hero index."""
        with self.stats_path.open("r", encoding="utf-8") as fh:
            self._raw = json.load(fh)
        self._rebuild()
        return self

    def load_from_dict(self, data: Dict) -> "DataManager":
        """Load already-parsed data (used by scrapers / tests)."""
        self._raw = data
        self._rebuild()
        return self

    def set_rank(self, rank: str) -> "DataManager":
        """Switch the active rank filter and re-resolve every hero's stats."""
        if rank not in config.RANK_TIERS:
            raise ValueError(
                f"Unknown rank {rank!r}; expected one of {config.RANK_TIERS}"
            )
        self.rank = rank
        self._rebuild()
        return self

    def _rebuild(self) -> None:
        """(Re)build the ``name -> Hero`` index for the active rank."""
        self._heroes = {}
        for name, blob in self._raw.get("heroes", {}).items():
            stats = self._resolve_stats(blob.get("stats_by_rank", {}))
            self._heroes[name] = Hero(
                name=name,
                roles=list(blob.get("roles", [])),
                lanes=list(blob.get("lanes", [])),
                damage_type=blob.get("damage_type", "physical"),
                win_rate=float(stats.get("win_rate", 0.5)),
                ban_rate=float(stats.get("ban_rate", 0.0)),
                pick_rate=float(stats.get("pick_rate", 0.0)),
                strong_against=list(blob.get("strong_against", [])),
                weak_against=list(blob.get("weak_against", [])),
                synergies=list(blob.get("synergies", [])),
            )

    def _resolve_stats(self, stats_by_rank: Dict) -> Dict:
        """Return the stat block for ``self.rank``, falling back to ``all``."""
        if self.rank in stats_by_rank:
            return stats_by_rank[self.rank]
        if "all" in stats_by_rank:
            return stats_by_rank["all"]
        # No data at all — return neutral values so scoring still works.
        return {"win_rate": 0.5, "ban_rate": 0.0, "pick_rate": 0.0}

    # -- queries ---------------------------------------------------------- #
    def get_hero(self, name: str) -> Optional[Hero]:
        return self._heroes.get(name)

    def all_heroes(self) -> List[Hero]:
        return list(self._heroes.values())

    @property
    def names(self) -> List[str]:
        return list(self._heroes.keys())

    def heroes_by_lane(self, lane: str) -> List[Hero]:
        return [h for h in self._heroes.values() if lane in h.lanes]

    def available_heroes(self, taken: Iterable[str]) -> List[Hero]:
        """All heroes that have not been picked or banned yet."""
        taken_set = set(taken)
        return [h for h in self._heroes.values() if h.name not in taken_set]

    @property
    def patch(self) -> str:
        return self._raw.get("meta", {}).get("patch", "unknown")

    # -- (mock) web scraper ---------------------------------------------- #
    @staticmethod
    def scrape_web(
        url: str = "https://example-mlbb-stats.invalid/heroes",
        rank: str = config.DEFAULT_RANK,
        save_to: Optional[Path | str] = None,
    ) -> Dict:
        """
        Mock web scraper.

        A real implementation would look roughly like this::

            import requests
            from bs4 import BeautifulSoup

            resp = requests.get(url, params={"rank": rank}, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            heroes = {}
            for row in soup.select("table.hero-stats tbody tr"):
                name = row.select_one(".hero-name").get_text(strip=True)
                heroes[name] = {
                    "win_rate": _pct(row.select_one(".win-rate").text),
                    "ban_rate": _pct(row.select_one(".ban-rate").text),
                    "pick_rate": _pct(row.select_one(".pick-rate").text),
                    ...  # counters / synergies from a detail page
                }
            data = {"meta": {...}, "heroes": heroes}

        To keep the project fully runnable offline, this mock simply loads and
        returns the bundled JSON dataset (optionally writing a copy to
        ``save_to``). Swap in the body above to scrape a live source.
        """
        with config.HERO_STATS_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("meta", {})["scraped_from"] = url
        data["meta"]["scraped_rank"] = rank
        if save_to is not None:
            with Path(save_to).open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        return data


# --------------------------------------------------------------------------- #
# Manual smoke test:  python -m src.data_manager
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    dm = DataManager().load()
    print(f"Loaded {len(dm.all_heroes())} heroes from patch {dm.patch}")
    print(f"Active rank: {dm.rank}")
    top = sorted(dm.all_heroes(), key=lambda h: h.meta_strength, reverse=True)[:5]
    print("\nTop 5 by meta strength (Mythical Glory):")
    for h in top:
        print(
            f"  {h.name:<12} {','.join(h.lanes):<14} "
            f"WR {h.win_rate:.0%}  BR {h.ban_rate:.0%}"
        )

    dm.set_rank("all")
    print("\nSame heroes at 'all' rank:")
    for h in top:
        hh = dm.get_hero(h.name)
        print(f"  {hh.name:<12} WR {hh.win_rate:.0%}  BR {hh.ban_rate:.0%}")
