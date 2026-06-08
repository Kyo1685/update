# Hero portrait templates

Put one cropped portrait image per hero in this folder. The vision module
(`src/vision.py`) loads every `*.png` here and uses it as a template-matching
reference to detect picks and bans on screen.

## Naming

The file name (minus extension) is the hero name and **must match the keys in
`data/hero_stats.json`**. Spaces or underscores both work:

```
Fanny.png
Lancelot.png
Yu Zhong.png        # or  Yu_Zhong.png
```

## How to make good templates

1. Open the MLBB draft screen (or a clear screenshot).
2. Crop each hero's portrait *tightly* — just the face/icon as it appears in the
   pick/ban slots, with as little border as possible.
3. Save as PNG here, named after the hero.
4. Keep them roughly the same aspect ratio as the on-screen slots. The matcher
   resizes each template to the slot size automatically, so exact dimensions
   don't matter, but consistent framing improves accuracy.

## Tips

- Use the **picked/locked-in** art style (the small square portraits), not the
  full splash art.
- If detection is flaky, lower `MATCH_THRESHOLD` in `config.py` slightly
  (e.g. 0.70 → 0.62) or improve the crops.
- Run `python tools/calibrate.py dump` to see exactly what the scanner is
  cropping for each slot, and adjust `config.REGIONS` until each crop holds one
  portrait.

> This folder ships empty (only this README). Detection stays inactive until you
> add templates — the app still runs, it just reports an empty draft. Use
> `python main.py --mock` to try everything without templates.
