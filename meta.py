"""
meta.py
=======
The live MLBB meta from the official source: Moonton's own hero-rank data (the
numbers behind the Hero Rank page on the official site), no login needed.

    win / pick / ban rate    per hero, for a time window (past 1/3/7/15/30 days)
                             and rank group (all ranks ... Mythical Glory)
    counters                 the 5 enemies this hero's win rate rises most against
    countered_by             the 5 enemies it falls most against
    synergies                the 5 teammates it rises most with
    roster                   every hero's roles, lanes and official portrait -
                             so a newly released hero can be added in one go

Two ways in (both keep heroes.json as the offline fallback):

  * live:     ``MoontonStatsProvider`` plugs into stats_provider's cache +
              repository, so the running app refreshes the numbers itself.
  * one-shot: ``tools/update_meta.py`` writes them into heroes.json and adds
              new heroes (with their portraits, so the detector knows them).

Only the standard library is needed; portraits additionally need opencv.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import json
import os
import re
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import config
from stats_provider import StatsProvider

# Moonton's hero-rank sources, one per time window (past N days).
RANK_SOURCES = {1: 2756567, 3: 2756568, 7: 2756569, 15: 2756565, 30: 2756570}
HERO_LIST_SOURCE = 2756564
# Rank groups ("bigrank").
RANKS = {"all": "101", "epic": "5", "legend": "6", "mythic": "7",
         "honor": "8", "glory": "9"}
_LANES = {"exp lane": "EXP", "gold lane": "GOLD", "jungle": "JUNGLE",
          "mid lane": "MID", "roam": "ROAM"}
_UA = {"User-Agent": "Mozilla/5.0 (mlbb-draft-overlay)",
       "Content-Type": "application/json"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class NameMap:
    """Official hero names -> this app's spellings (heroes.json / template
    files), e.g. Moonton's "Minsitthar" -> "Minsithar".  Unknown heroes keep
    the official name."""

    def __init__(self, known: Iterable[str], cutoff: float = 0.85):
        self._known = {_norm(n): n for n in known}
        self._cutoff = cutoff

    def __call__(self, official: str) -> str:
        key = _norm(official)
        if key in self._known:
            return self._known[key]
        near = difflib.get_close_matches(key, list(self._known), n=1, cutoff=self._cutoff)
        return self._known[near[0]] if near else official

    def knows(self, name: str) -> bool:
        return _norm(name) in self._known


class MoontonClient:
    """Read-only client for Moonton's public hero data."""

    def __init__(self, base: Optional[str] = None, timeout: float = 30.0):
        self.base = (base or config.META_API).rstrip("/") + "/"
        self.timeout = timeout

    def _post(self, source: int, body: dict) -> List[dict]:
        req = urllib.request.Request(f"{self.base}{source}", headers=_UA,
                                     data=json.dumps(body).encode("utf-8"))
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("code") != 0:
            raise RuntimeError(f"Moonton API error {payload.get('code')}: "
                               f"{payload.get('message')}")
        return [r.get("data") or {} for r in payload["data"]["records"]]

    def hero_rank(self, days: int = 7, rank: str = "all", teammates: bool = False
                  ) -> List[dict]:
        """One record per hero: rates + the 5 best / worst matchups (vs enemies,
        or with teammates when ``teammates``)."""
        if days not in RANK_SOURCES:
            raise ValueError(f"days must be one of {sorted(RANK_SOURCES)}")
        if rank not in RANKS:
            raise ValueError(f"rank must be one of {sorted(RANKS)}")
        filters = [{"field": "bigrank", "operator": "eq", "value": RANKS[rank]},
                   {"field": "match_type", "operator": "eq",
                    "value": "1" if teammates else "0"}]
        fields = ["main_heroid", "main_hero.data.name", "main_hero_win_rate",
                  "main_hero_appearance_rate", "main_hero_ban_rate",
                  "sub_hero.heroid", "sub_hero.increase_win_rate",
                  "sub_hero_last.heroid", "sub_hero_last.increase_win_rate"]
        return self._post(RANK_SOURCES[days], {"pageSize": 500, "pageIndex": 1,
                                               "filters": filters, "fields": fields})

    def heroes(self) -> List[dict]:
        """The roster: [{id, name, roles, lanes, portrait}]."""
        rows = self._post(HERO_LIST_SOURCE, {
            "pageSize": 500, "pageIndex": 1,
            "fields": ["hero_id", "hero.data.name", "hero.data.head",
                       "hero.data.sortid", "hero.data.roadsort"]})
        out = []
        for r in rows:
            h = (r.get("hero") or {}).get("data") or {}
            if not h.get("name"):
                continue
            roles = [s["data"]["sort_title"].strip().title()
                     for s in h.get("sortid") or []
                     if isinstance(s, dict) and s.get("data", {}).get("sort_title")]
            lanes = []
            for s in h.get("roadsort") or []:
                if isinstance(s, dict):
                    lane = _LANES.get(str(s.get("data", {}).get("road_sort_title", "")).lower())
                    if lane and lane not in lanes:
                        lanes.append(lane)
            out.append({"id": r.get("hero_id"), "name": h["name"].strip(),
                        "roles": roles, "lanes": lanes, "portrait": h.get("head")})
        return out


def fetch_meta(client: Optional[MoontonClient] = None, days: Optional[int] = None,
               rank: Optional[str] = None, names: Optional[NameMap] = None,
               top: Optional[int] = None) -> Dict[str, dict]:
    """{hero: {win_rate, ban_rate, pick_rate, counters, countered_by,
    synergies}} with percentages like heroes.json and names mapped to this
    app's spellings."""
    client = client or MoontonClient()
    days = config.META_DAYS if days is None else days
    rank = config.META_RANK if rank is None else rank
    top = config.META_TOP if top is None else top
    names = names or NameMap(())
    versus = client.hero_rank(days, rank)
    together = client.hero_rank(days, rank, teammates=True)
    by_id = {r["main_heroid"]: names(r["main_hero"]["data"]["name"])
             for r in versus if r.get("main_hero", {}).get("data", {}).get("name")}

    def heroes_of(entries) -> List[str]:
        return [by_id[e["heroid"]] for e in (entries or [])[:top]
                if e.get("heroid") in by_id]

    meta: Dict[str, dict] = {}
    for r in versus:
        name = by_id.get(r.get("main_heroid"))
        if not name:
            continue
        meta[name] = {
            "win_rate": round(100.0 * float(r.get("main_hero_win_rate") or 0), 2),
            "ban_rate": round(100.0 * float(r.get("main_hero_ban_rate") or 0), 2),
            "pick_rate": round(100.0 * float(r.get("main_hero_appearance_rate") or 0), 3),
            "counters": heroes_of(r.get("sub_hero")),
            "countered_by": heroes_of(r.get("sub_hero_last")),
        }
    for r in together:
        name = by_id.get(r.get("main_heroid"))
        if name in meta:
            meta[name]["synergies"] = heroes_of(r.get("sub_hero"))
    return meta


class MoontonStatsProvider(StatsProvider):
    """Live overlay for StatsRepository (wrap it in CachedStatsProvider).
    ``new_heroes`` lists official heroes this app doesn't know yet."""

    def __init__(self, known_names: Iterable[str], days: Optional[int] = None,
                 rank: Optional[str] = None, client: Optional[MoontonClient] = None):
        self.names = NameMap(list(known_names))
        self.days, self.rank = days, rank
        self.client = client or MoontonClient()
        self.new_heroes: List[str] = []

    def fetch(self, force: bool = False) -> Dict[str, dict]:
        meta = fetch_meta(self.client, self.days, self.rank, self.names)
        self.new_heroes = sorted(n for n in meta if not self.names.knows(n))
        return meta

    def describe(self) -> str:
        days = config.META_DAYS if self.days is None else self.days
        rank = config.META_RANK if self.rank is None else self.rank
        return f"official Moonton hero rank, past {days} days, {rank} ranks"


# ===========================================================================
#  heroes.json update (tools/update_meta.py)
# ===========================================================================
_STAT_FIELDS = ("win_rate", "ban_rate", "pick_rate", "counters", "countered_by",
                "synergies")


def _damage_guess(roles: Sequence[str]) -> str:
    return "Magical" if roles and roles[0] == "Mage" else "Physical"


def update_heroes(payload: dict, meta: Dict[str, dict], roster: Sequence[dict],
                  names: NameMap, lanes: bool = True,
                  source: str = "") -> Tuple[dict, List[str], List[str]]:
    """Apply fresh meta (+ roster) to a heroes.json payload.  Returns
    (new payload, changes, added heroes).  Manual fields - owned,
    archetypes, damage type - are never touched for known heroes."""
    heroes = [dict(h) for h in payload.get("heroes", [])]
    by_key = {_norm(h["name"]): h for h in heroes}
    roster_by = {_norm(names(r["name"])): r for r in roster}
    changes: List[str] = []
    added: List[str] = []
    for name, stats in meta.items():
        key = _norm(name)
        hero = by_key.get(key)
        info = roster_by.get(key, {})
        fresh = hero is None
        if fresh:
            roles = info.get("roles") or ["Fighter"]
            hero = {"name": name, "base_role": roles[0], "win_rate": 50.0,
                    "ban_rate": 0.0, "counters": [], "countered_by": [],
                    "synergies": [], "damage_type": _damage_guess(roles),
                    "archetypes": [], "owned": False,
                    "lanes": info.get("lanes") or [], "roles": roles}
            heroes.append(hero)
            by_key[key] = hero
            added.append(name)
        old_wr, old_br = hero.get("win_rate"), hero.get("ban_rate")
        for f in _STAT_FIELDS:
            if f in stats:
                hero[f] = stats[f]
        if fresh:
            continue
        if old_wr is not None and abs(float(old_wr) - hero["win_rate"]) >= 2.0:
            changes.append(f"{hero['name']}: win rate {old_wr} -> {hero['win_rate']}")
        if old_br is not None and abs(float(old_br) - hero["ban_rate"]) >= 10.0:
            changes.append(f"{hero['name']}: ban rate {old_br} -> {hero['ban_rate']}")
        if lanes and info.get("lanes") and set(info["lanes"]) != set(hero.get("lanes", [])):
            changes.append(f"{hero['name']}: lanes {hero.get('lanes')} -> {info['lanes']}")
            hero["lanes"] = list(info["lanes"])
        if lanes and info.get("roles") and hero.get("roles") != info["roles"]:
            hero["roles"] = list(info["roles"])
            hero["base_role"] = info["roles"][0]
    out = dict(payload)
    out["heroes"] = heroes
    out["meta"] = {"source": source or "Moonton hero rank",
                   "updated": _dt.date.today().isoformat()}
    return out, changes, added


def save_portrait(url: str, name: str, dirs: Optional[Dict[str, bool]] = None,
                  timeout: float = 30.0) -> List[str]:
    """Download an official portrait and save it into every template folder
    the way tools/rebuild_templates.py does (dark-background composite, 160 px
    square; the enemy set mirrored).  Returns the files written."""
    import numpy as np
    import cv2
    dirs = dirs or {config.TEMPLATE_DIR: False, config.TEMPLATE_CIRCLE_DIR: False,
                    config.TEMPLATE_ALLY_DIR: False, config.TEMPLATE_ENEMY_DIR: True}
    req = urllib.request.Request(url, headers={"User-Agent": _UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode the portrait for {name}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:                               # flatten alpha onto dark
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 18.0 * (1.0 - a)).astype(np.uint8)
    h, w = img.shape[:2]
    m = min(h, w)
    img = cv2.resize(img[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m],
                     (160, 160), interpolation=cv2.INTER_AREA)
    fname = re.sub(r'[<>:"/\\|?*]', "", name).strip().lower() + ".png"
    written = []
    for d, mirror in dirs.items():
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, fname)
        cv2.imwrite(path, cv2.flip(img, 1) if mirror else img)
        written.append(path)
    return written
