"""
tools/update_meta.py
====================
Bring heroes.json up to the CURRENT meta from the official source - Moonton's
own hero-rank data (the numbers behind the Hero Rank page on the official
site): win / ban / pick rates, counters, countered-by, synergies, recommended
lanes and roles.  Newly released heroes are added too, with their official
portraits in every template folder, so the overlay both recommends AND
recognises them.

    python tools/update_meta.py                      # past 7 days, all ranks
    python tools/update_meta.py --days 30 --rank mythic
    python tools/update_meta.py --dry-run            # show changes, write nothing
    python tools/update_meta.py --keep-lanes         # leave lanes/roles alone

heroes.json is backed up to heroes.json.bak first.  Your own fields - owned
heroes, archetypes, damage type - are kept (new heroes start as not owned:
add them with tools/set_owned.py once you buy them).

The running app refreshes the same numbers by itself (config.META_LIVE); this
tool makes them permanent and is the one to run when a new hero comes out.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config
from meta import (RANK_SOURCES, RANKS, MoontonClient, NameMap, fetch_meta,
                  save_portrait, update_heroes)


def _art_path(name: str) -> str:
    return os.path.join(config.TEMPLATE_CIRCLE_DIR, f"{name.strip().lower()}.png")


def main() -> int:
    p = argparse.ArgumentParser(description="Update heroes.json to the current "
                                            "meta from the official hero rank.")
    p.add_argument("--heroes", default="heroes.json")
    p.add_argument("--days", type=int, default=config.META_DAYS, choices=sorted(RANK_SOURCES),
                   help="time window: past N days (default %(default)s)")
    p.add_argument("--rank", default=config.META_RANK, choices=sorted(RANKS),
                   help="rank group (default %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="show changes, write nothing")
    p.add_argument("--keep-lanes", action="store_true",
                   help="don't update lanes/roles from the official roster")
    p.add_argument("--no-portraits", action="store_true",
                   help="don't download portraits for heroes missing one")
    args = p.parse_args()

    with open(args.heroes, encoding="utf-8") as fh:
        payload = json.load(fh)
    names = NameMap([h["name"] for h in payload.get("heroes", [])])
    client = MoontonClient()
    source = f"Moonton hero rank, past {args.days} days, {args.rank} ranks"
    print(f"[meta] fetching the official hero rank ({source})...")
    try:
        meta = fetch_meta(client, args.days, args.rank, names)
        roster = client.heroes()
    except Exception as exc:
        sys.stderr.write(f"[meta] could not reach the official source: {exc}\n"
                         "Nothing was changed.\n")
        return 1
    if not meta:
        sys.stderr.write("[meta] the source returned no heroes - nothing changed.\n")
        return 1

    updated, changes, added = update_heroes(payload, meta, roster, names,
                                            lanes=not args.keep_lanes, source=source)
    print(f"[meta] {len(meta)} heroes in the official data; "
          f"{len(updated['heroes'])} in heroes.json after the update")
    top_bans = sorted(meta.items(), key=lambda kv: -kv[1]["ban_rate"])[:8]
    top_wins = sorted(meta.items(), key=lambda kv: -kv[1]["win_rate"])[:8]
    print("  most banned : " + ", ".join(f"{n} {s['ban_rate']:.0f}%" for n, s in top_bans))
    print("  best win    : " + ", ".join(f"{n} {s['win_rate']:.1f}%" for n, s in top_wins))
    if added:
        print(f"  NEW heroes  : {', '.join(added)}")
    if changes:
        print(f"  big changes ({len(changes)}):")
        for c in changes[:25]:
            print(f"    {c}")
        if len(changes) > 25:
            print(f"    ... and {len(changes) - 25} more")

    # Portraits for every hero the detector has no art for (new releases).
    roster_by = {names(r["name"]).lower(): r for r in roster}
    need_art = [h["name"] for h in updated["heroes"] if not os.path.exists(_art_path(h["name"]))]
    if need_art:
        print(f"  portraits missing: {', '.join(need_art)}")

    if args.dry_run:
        print("[meta] dry run - nothing written.")
        return 0

    shutil.copyfile(args.heroes, args.heroes + ".bak")
    with open(args.heroes, "w", encoding="utf-8") as fh:
        json.dump(updated, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"[meta] wrote {args.heroes} (backup: {args.heroes}.bak)")

    if need_art and not args.no_portraits:
        for name in need_art:
            r = roster_by.get(name.lower())
            if not r or not r.get("portrait"):
                print(f"  (no official portrait for {name})")
                continue
            try:
                save_portrait(r["portrait"], name)
                print(f"  + portrait for {name} (all template folders)")
            except Exception as exc:
                print(f"  (portrait for {name} failed: {exc})")
    if added:
        print("New heroes start as NOT owned - add the ones you have with "
              "tools/set_owned.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
