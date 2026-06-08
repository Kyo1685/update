# Hero avatar templates

Drop one cropped **locked-portrait** image per hero here. The detector matches
these against the live draft slots with `cv2.matchTemplate` (with an HSV
colour-histogram fallback).

## Naming
The filename (minus extension) becomes the hero name, so it must resolve
against `heroes.json` (matching is case-insensitive):

```
guinevere.png        -> "Guinevere"
yi_sun-shin.png      -> "Yi Sun-Shin"   (resolves to "Yi Sun-shin")
x.borg.png           -> "X.Borg"
```

Underscores become spaces. Accepted extensions: `.png .jpg .jpeg .webp`.

## How to crop good templates
1. Run a real draft (or open a replay) on the 2712x1220 mirror.
2. `python main.py --mock` is *not* needed; instead use the calibration grid:
   - `python -c "import config; print(config.LAYOUT)"` shows the exact pixel
     boxes, or
   - capture one frame and slice it with the boxes from `config.LAYOUT`.
3. Save each hero's **square portrait** region (roughly 152x152 at this
   resolution). Tight, centred crops score highest.

A handful of templates is enough to start; the histogram fallback covers the
rest at lower confidence (`--accept-low` to trust those too).

> Templates are personal assets — none are shipped in this repo.
