# `templates_enemy/` — enemy-side pick override crops

Drop a hero portrait **cropped from the ENEMY (right) side of the draft** here as
`<hero>.png` (lower-case, e.g. `helcurt.png`).

Each file here **overrides** the auto-oriented base template for that hero on the
**enemy side only** — it does not affect the ally side or the ban row. Add only
the heroes the base set misreads; the rest keep using the bundled square splash
art.

The enemy picks are **square splash** art (not inscribed) and are matched as-is
with **no** auto-flip — crop them exactly as they appear on your screen.

The ally counterpart lives in [`../templates_ally/`](../templates_ally).

Seeded with `helcurt.png` — the ally crop mirrored to face **left** — as a
worked example.
