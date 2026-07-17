# visualizer/ — CLAUDE.md

Godot 4 playback viewer. **Reads the root `CLAUDE.md` first** — this file only
adds viewer-local notes.

## Hard rules (repeated because they're load-bearing)

- **Knows ONLY the cache format.** Never import or reference the solver,
  Python, or Taichi (root §2). The only input is a cache directory.
- **Read the attribute layout from the manifest — never hardcode column
  offsets** (root §4, CACHE_FORMAT §2). The solver can append attributes and
  the viewer must keep working.
- **Develop against the fixture.** `fixtures/tiny_golden_cache/` lets you build
  and run the viewer with no solver present. `particle_view.gd` defaults to it.

## Layout

| File | Role |
|---|---|
| `project.godot` | Godot 4 project; main scene = `main.tscn` |
| `main.tscn` | Minimal scene: one `MultiMeshInstance2D` + `particle_view.gd` |
| `scripts/cache_loader.gd` | Parses manifest + streams `frames.bin` (the contract reader) |
| `scripts/particle_view.gd` | `MultiMeshInstance2D` playback + per-instance color/heat, HUD (scenario caption, speed slider+field), glow env |
| `shaders/particle.gdshader` | Soft-disc particle shader; per-instance "heat" drives size + HDR glow |
| `fixtures/tiny_golden_cache/` | Canonical 3-frame cache (regenerate via `tools/make_golden_cache.py`) |

## Why MultiMeshInstance2D

One draw call for the whole point cloud, per-instance color, trivial `float32`
reads — the viewer is "just a player." Unlikely to change (root §5).

## Known limit: `_show_frame` is the smoothness ceiling

**Playback smoothness is bound by the viewer, not by the cache.** `_show_frame`
runs a per-particle GDScript loop making three `set_instance_*` calls each, plus a
`frames.bin` read. Measured cost:

| deck | particles | `_show_frame` | max sustainable fps |
|---|---|---|---|
| `apfsds_vs_rha` | 137k | ~100 ms | ~10 |
| `heat_vs_composite` | 179k | ~135 ms | ~7 |
| `apfsds_vs_era_oblique` | 287k | ~220 ms | ~4.5 |

Above that rate `_process` skips baked frames — which **throws away exactly the
resolution the uniform `frame_dt = 2.0e-7` was baked for**. Hence
`frames_per_second` defaults to 10, not 24: it is better to play every frame
slowly than to skip. The HUD speed slider (or `↑`/`↓`) retunes per deck.

The slider spans 0 (hard stop) and 1..240 fps, mapped **logarithmically** — a
linear travel would spend most of its length above the sustainable rate. It runs
well past that ceiling on purpose: skipping frames to scrub through a deck's tail
is a legitimate want, and "too slow" is a real complaint on a 700-frame deck at
10 fps (70 s of wall clock for a ~0.14 ms event). Smooth is the *default*, not a
cage.

**The field beside it is not a duplicate of the slider.** 100 log-spaced notches
cannot land on a round number, so the slider can *reach* 24 fps but cannot be
*told* 24 — and a rate you can read but not enter is a readout, not a control. The
field is the readout too; `_update_hud` deliberately no longer prints fps, because
two live displays of one variable is one more than can be kept honest.

Three things about the field are load-bearing, all verified rather than assumed:

- **It reverts garbage instead of obeying it.** `"abc".to_float()` is `0.0`, and
  `0.0` here means *hard stop* — so parsing without `is_valid_float()` turns a
  typo into a frozen viewer that reads as a playback bug.
- **Submitting releases focus.** Unlike the slider (which is `FOCUS_NONE` exactly
  so it cannot swallow `←`/`→`), a field must hold the caret to be typed into, so
  while focused it eats `SPACE` and `←`/`→`. Enter hands them back.
- **`_apply_fps` is the only writer of `frames_per_second`**, and both widgets are
  mirrored from it with set-without-signal. Nothing between 0 and `MIN_FPS` is
  representable on the slider, so a value in that gap snaps up rather than letting
  the two widgets show different speeds.

**Pause and speed are deliberately independent.** `SPACE` owns `_playing`; the
slider and field own `frames_per_second`, and 0 fps freezes `_process` on its own.
Because pausing never zeroes the speed, resuming needs no saved-speed variable —
it just picks the slider back up. Don't "simplify" these into one state; that's
what forces a `_last_speed` and three coupled handlers.

## The caption: the viewer can finally say what it is drawing

Schema v3 added the **scenario block** (`CACHE_FORMAT` §2.1) for this half of the
repo specifically. The viewer knows only the cache format, so before it, playback
was an unlabelled point cloud — and every scalar color mode dropped even the
material *names*, because the legend became a ramp.

- **The material list is always on**, in every color mode. It is only a *color
  key* when `material_id` is the active mode; in a scalar mode the swatches go
  neutral gray and the caption says so, because nothing on screen is
  tungsten-gold then and a gold swatch would be a legend for a color that is not
  there.
- **List the ids present in the DATA, never the manifest's `materials` keys.** The
  solver emits its whole library, so those are seven ids on every deck (§2.1 says
  this outright). `_collect_present_materials` reads frame 0 — which is not a
  sample: particle count is fixed and a particle never changes material, so t=0 is
  the whole bake.
- **It is provenance, not data.** `projectile.velocity` is the seeded tip speed at
  t=0. Nothing computes from these; they are strings on their way to a `Label`.
  The live velocity is the `vel_mag` column.
- The panel is dense by design and covers part of the field, so **`H` hides it**.
  Note the wheel guard in `_unhandled_input` tests `_hud_panel.visible` — a hidden
  panel still has a rect, and without that check `H` leaves an invisible dead zone
  that eats scroll.

*(Those figures come from the `--shots` path, which reads two frames per call
because its frames are non-sequential — sequential playback reads one, so the true
cost is somewhat lower. The ordering and the conclusion hold.)*

**The real fix, when this matters: `MultiMesh.set_buffer()`** — one bulk upload of
a prebuilt `PackedFloat32Array` (8 transform + 4 color + 4 custom floats per
instance) instead of ~3×N engine calls per frame. Not done yet; it is the highest-
value viewer work outstanding. Until then, particle count is a direct tax on
playback, so weigh it when sizing a deck's domain.

**Gotcha:** `--shots` sets `set_process(false)` and routes no input, so it verifies
*rendering* only — never playback timing, zoom, pan, or any key. Don't mistake a
clean capture for a working interactive viewer.
