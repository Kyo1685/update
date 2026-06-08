# MLBB Draft Assistant 🛡️

A real-time AI **drafting assistant + overlay** for *Mobile Legends: Bang Bang*.
It reads the draft screen, scores every available hero against the live
pick/ban state and current meta, and shows the statistically optimal picks per
lane in a transparent, always-on-top overlay.

Inspired by the draft-helper concept popularised by YouTuber **Otev-**.

> ⚠️ **Use responsibly.** This is an educational computer-vision / decision-engine
> project. It only reads pixels that are already visible to you on your own
> screen and gives *advice* — it does not touch the game's memory, files, or
> network, and it does not automate inputs. Even so, third-party overlays may be
> against a game's Terms of Service; check the rules and use at your own risk,
> ideally for solo study, VOD review, and learning drafts.

---

## Architecture

```
        ┌──────────────────────┐
        │  Data Scraper/Manager │  hero_stats.json  (win/ban/pick, counters,
        │   (Module 1)          │                    synergies, by rank)
        └──────────┬───────────┘
                   │ Hero objects (rank-resolved)
                   ▼
┌─────────────┐   DraftState   ┌──────────────────────┐
│   Vision    │ ─────────────▶ │   Scoring Engine      │
│ (Module 2)  │  picks/bans    │   (Module 3)          │
│ mss+OpenCV  │                │ baseline + meta +     │
└─────────────┘                │ counters + balance +  │
   ▲ screen                    │ lane-counter / dark   │
   │                           └──────────┬───────────┘
   │ scan() every N s                     │ {lane: [ScoreBreakdown...]}
   │  (background QThread)                ▼
   │                           ┌──────────────────────┐
   └───────────────────────────│   UI Overlay          │
                               │   (Module 4) PyQt5    │
                               │ top pick per lane,    │
                               │ Next / Lane Counter / │
                               │ Dark System / Lock    │
                               └──────────────────────┘
```

**Data flow:** a background worker thread calls `Vision.scan()` → builds a
`DraftState` → `ScoringEngine.recommend()` ranks heroes per lane → the result is
pushed to the overlay via a Qt signal (the UI never blocks on capture/scoring).

---

## Project structure

```
mlbb_draft_assistant/
├── main.py                  # Entry point: wires modules + background worker
├── config.py                # Paths, lanes, scoring weights, capture regions
├── requirements.txt
├── data/
│   ├── hero_stats.json      # Mock meta data (editable / scrapeable)
│   └── portraits/           # Hero portrait templates (you add these)
├── src/
│   ├── data_manager.py      # Module 1 — load/filter stats, mock scraper
│   ├── scoring_engine.py    # Module 3 — DraftState + scoring logic
│   ├── vision.py            # Module 2 — mss capture + OpenCV matching
│   └── overlay.py           # Module 4 — PyQt5 transparent overlay
├── tools/
│   └── calibrate.py         # Find/verify screen regions for your resolution
└── tests/
    └── test_scoring_engine.py
```

---

## Quick start

```bash
cd mlbb_draft_assistant
python -m pip install -r requirements.txt

# 1) Try everything WITHOUT the game or templates (scripted draft):
python main.py --mock

# 2) Verify the pure logic:
python -m unittest discover -s tests -v

# 3) Go live (after adding portraits + calibrating, see below):
python main.py
```

### Live setup (3 steps)

1. **Add portraits** — drop one `*.png` per hero into `data/portraits/`, named to
   match `data/hero_stats.json` (e.g. `Fanny.png`, `Yu Zhong.png`). See
   `data/portraits/README.md`.
2. **Calibrate regions** — open the draft screen and run:
   ```bash
   python tools/calibrate.py dump      # dumps slot crops to ./calibration_debug
   python tools/calibrate.py select    # drag boxes, prints [left,top,w,h]
   ```
   Paste the coordinates into `REGIONS` in `config.py`.
3. **Run** — `python main.py`. Drag the overlay where you want it.

---

## The four modules

### 1. Data Scraper & Manager — `src/data_manager.py`
- Loads `hero_stats.json` into rank-resolved `Hero` objects.
- `set_rank("mythical_glory" | "all" | ...)` re-resolves every hero's
  win/ban/pick rate, falling back to `all` when a rank is missing.
- `scrape_web()` is a documented **mock** of a `requests`+`BeautifulSoup`
  scraper; by default it returns the bundled data so the app runs offline.

### 2. Vision — `src/vision.py`
- `mss` grabs the monitor once per scan (fast, no temp files).
- Each pick/ban **slot** rectangle is cropped and matched against the portrait
  templates with OpenCV `matchTemplate` (`TM_CCOEFF_NORMED`); best score above
  `MATCH_THRESHOLD` wins.
- Returns a `DraftState(ally_picks, enemy_picks, bans)`.
- `MockVision` provides a screen-free draft for demos/tests.

### 3. Scoring Engine — `src/scoring_engine.py`
Baseline **10 pts**, then:
| Factor | Effect |
|---|---|
| Win rate vs 50%, ban rate, pick rate | meta strength + |
| Our counter is banned / on our team | + (`counter_neutralised`) |
| Enemy already has a hero that beats us | − (`countered_by_enemy`) |
| We beat a hero the enemy picked | + (`we_counter_enemy`) |
| Ally synergy partner present | + (`synergy`) |
| Our team already filled this lane | − (`role_saturation`) |
| Damage type our team lacks | + (`damage_balance`) |
| **Lane Counter mode**: same-lane hard counter | + (`lane_counter_bonus`) |

Every suggestion carries a `ScoreBreakdown` (`.explain()`), and **Dark System**
simply reverses the sort to surface the statistically *worst* heroes.

### 4. UI Overlay — `src/overlay.py`
- Frameless, always-on-top, translucent; top pick per lane with score and a
  "next:" preview; hover any pick to see the scoring breakdown.
- **Next** pages through suggestions (own a different hero? skip to it).
- **Lane Counter** / **Dark System** toggles.
- **Lock** = OS-level click-through (Windows `WS_EX_TRANSPARENT`; other
  platforms best-effort). While locked the buttons are click-through too, so
  unlock with the global hotkey **Ctrl+Alt+L** (needs the optional `keyboard`
  package). Other hotkeys: **Ctrl+Alt+N** next, **Ctrl+Alt+Q** quit.

---

## Customization

- **Tune the engine** — edit weights in `config.SCORING`.
- **Update the meta** — edit `data/hero_stats.json` or wire a real scraper into
  `DataManager.scrape_web()`.
- **Change scan speed** — `config.SCAN_INTERVAL_S` or `--interval`.
- **Match sensitivity** — `config.MATCH_THRESHOLD`.

## Tested

`python -m unittest discover -s tests -v` covers rank filtering, every scoring
component, lane-counter boosting, role/damage balancing, and Dark System
inversion. These run with **no GUI/OpenCV** needed.
