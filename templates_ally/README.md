# `templates_ally/` — ally-side pick override crops

Drop a hero portrait **cropped from the ALLY (left) side of the draft** here as
`<hero>.png` (lower-case, spaces → keep, e.g. `helcurt.png`, `yi sun-shin.png`).

Each file here **overrides** the auto-oriented base template for that hero on the
**ally side only** — it does not affect the enemy side or the ban row. You only
need to add the few heroes the base set misreads; everything else keeps using
the bundled circular avatars.

The ally avatars are **circular** and face the opposite way to the enemy splash,
so these crops are matched on the inscribed face (`circular=True`) and are **not**
auto-flipped — crop them exactly as they appear on your screen.

The enemy counterpart lives in [`../templates_enemy/`](../templates_enemy).

Seeded with `helcurt.png` (the circular ally avatar) as a worked example.
