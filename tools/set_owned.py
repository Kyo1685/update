"""
tools/set_owned.py
==================
Mark which heroes YOU own, so the overlay's HERO POOL FILTER only ever
suggests heroes from your roster.

Pass your roster (comma/space/newline separated, spelling-forgiving - fuzzy
matched to heroes.json); every listed hero becomes ``owned: true`` and every
other hero becomes ``owned: false``.  Re-run any time you buy a new hero.

    python tools/set_owned.py --names "miya, balmond, layla, ..."
    python tools/set_owned.py --file my_heroes.txt
    python tools/set_owned.py --add "aamon, cici"     # keep current, add these
    python tools/set_owned.py --all                   # own everything (default seed)

Then in the overlay turn HERO POOL FILTER on.  The script prints what matched
and - importantly - any names it could NOT resolve, so nothing is silently
dropped.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys

HEROES = "heroes.json"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _split(raw: str):
    # accept commas, newlines or runs of spaces as separators
    return [p.strip() for p in re.split(r"[,\n]+", raw) if p.strip()]


def resolve(name: str, index: dict, cutoff: float = 0.82):
    key = _norm(name)
    if key in index:
        return index[key]
    near = difflib.get_close_matches(key, list(index), n=1, cutoff=cutoff)
    return index[near[0]] if near else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Flag owned heroes for the pool filter.")
    ap.add_argument("--names", default="", help="owned roster (comma separated)")
    ap.add_argument("--file", default=None, help="read roster from a text file")
    ap.add_argument("--add", default="", help="ADD these to the current owned set")
    ap.add_argument("--all", action="store_true", help="mark every hero owned")
    ap.add_argument("--heroes", default=HEROES)
    args = ap.parse_args()

    with open(args.heroes, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    heroes = payload["heroes"] if isinstance(payload, dict) else payload
    index = {_norm(h["name"]): h["name"] for h in heroes}

    if args.all:
        for h in heroes:
            h["owned"] = True
        _save(payload, args.heroes)
        print(f"All {len(heroes)} heroes marked owned.")
        return 0

    raw = args.names or args.add
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            raw = (raw + "\n" + fh.read()) if raw else fh.read()
    wanted = _split(raw)
    if not wanted:
        sys.stderr.write("nothing to do: pass --names / --file / --add / --all\n")
        return 2

    resolved, unmatched = {}, []
    for n in wanted:
        canon = resolve(n, index)
        (resolved.setdefault(canon, n) if canon else unmatched.append(n))

    owned_names = set(resolved)
    if args.add:                                  # union with the current set
        owned_names |= {h["name"] for h in heroes if h.get("owned")}

    for h in heroes:
        h["owned"] = h["name"] in owned_names
    _save(payload, args.heroes)

    n_owned = sum(1 for h in heroes if h["owned"])
    print(f"Owned: {n_owned}/{len(heroes)} heroes  (HERO POOL FILTER now suggests "
          f"only these).")
    print("  " + ", ".join(sorted(owned_names)))
    if unmatched:
        print(f"\n!! {len(unmatched)} name(s) NOT recognised (fix spelling & re-run): "
              + ", ".join(unmatched))
        print("   valid names: see heroes.json")
    return 0


def _save(payload, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
