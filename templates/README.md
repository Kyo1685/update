# Hero avatar templates

One portrait per hero lives here. The detector matches these against the live
draft slots with `cv2.matchTemplate` (with an HSV colour-histogram fallback).

## Fastest way: auto-download the whole roster

```bash
python fetch_templates.py            # ~130 portraits, named correctly
python fetch_templates.py --only-db  # just heroes in heroes.json
```

That's it — skip the rest unless you want to hand-tune a few.

## Naming (if you add files manually)
The filename (minus extension) becomes the hero name, resolved against
`heroes.json` case-insensitively:

```
guinevere.png        -> "Guinevere"
yi_sun-shin.png      -> "Yi Sun-shin"
x.borg.png           -> "X.Borg"
```

Underscores become spaces. Accepted extensions: `.png .jpg .jpeg .webp`.

## Hand-cropping for max accuracy
Auto-downloaded portraits match well via the histogram fallback (run the app
with `--accept-low`). For a tighter, higher-confidence match on a specific
hero, replace its file with a **square crop of the locked portrait** taken from
a screenshot of your own 2712×1220 draft (≈152×152). Tight, centred crops score
highest.

> Templates are personal assets — none are committed to this repo.
