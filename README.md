# armor_pen

A **2D offline simulation of terminal ballistics** — a shell (kinetic-energy
penetrator or shaped-charge jet) striking tank armor (RHA, composite,
spaced/NERA, reactive) — rendered as a deformation simulation with spall and
fragmentation.

> **Educational / game-grade, not engineering-grade.** This project is built
> entirely on **public-domain, textbook-level physics** (Tate–Alekseevskii
> hydrodynamic penetration, standard Material Point Method, von Mises
> plasticity). The fidelity bar is **plausible, not validated**: we want
> results that *look and behave* like real penetration events and respond
> sensibly to real-ish parameters. We do **not** validate against, reproduce,
> or tune toward real experimental performance, real munition/armor
> specifications, or anything meant to optimize lethality. Material parameters
> are representative order-of-magnitude values from public literature. See
> [`CLAUDE.md` §10](CLAUDE.md) and [`docs/PHYSICS.md`](docs/PHYSICS.md) for the
> full scope guard and references.

## Architecture in one sentence

> **The solver and the visualizer are two independent programs that share
> exactly one thing: the on-disk cache format. Nothing else.**

The solver (Python + NVIDIA Warp, MLS-MPM) *bakes* a scenario offline to a cache
on disk. The visualizer (Godot 4) plays that cache back. Neither imports the
other; the only bridge is the language-neutral cache format specified in
[`docs/CACHE_FORMAT.md`](docs/CACHE_FORMAT.md). This decoupling is deliberate —
the solver is disposable and may be rewritten (Warp → CUDA/Rust) without
touching a line of visualizer code. (It already paid off: the original Taichi
plan was swapped for Warp with zero viewer changes.)

## Layout

```
docs/         CACHE_FORMAT.md (the contract) + PHYSICS.md
solver/       standalone MLS-MPM solver; knows nothing about Godot
visualizer/   Godot 4 player; knows only how to read the cache
tools/        language-neutral cache helpers (validate, inspect); depend on neither side
caches/       gitignored bake outputs (large)
```

## Status

All six solver milestones are done, and the Godot viewer plays real bakes back in
motion. Every KE deck bakes on the RTX 5090 (NVIDIA Warp, sm_120) and passes
`validate_cache`.

1. **Elasticity** — fixed-corotated MLS-MPM; elastic impact, no perforation.
2. **von Mises plasticity** — radial return in log-strain space; the rod mushrooms
   and the plate craters.
3. **Damage / spall** — a plastic-strain threshold detaches particles into free
   fragments: a penetration channel lined with spall, plus a crater-lip spray.
4. **Multi-material armor stack** — bonded and spaced decks, plus brittle
   (stress-triggered) fracture so ceramics shatter with ~zero plastic flow.
5. **Reactive ERA/NERA layer** — filler ignites on the impact shock and releases a
   detonation overpressure through the ordinary grid, flinging the sandwich plates
   apart (emergent, not a scripted rod kick).
6. **Oblique reactive armor** — the rod strikes nose-first at angle.

Every deck runs long enough to resolve **perforate-or-stop**, in a domain sized so
the armor spans the full field height: the target is a plate that continues past
the frame (armor on a vehicle), not a block floating in vacuum.

**Headline result (verified, `docs/PHYSICS.md` §3.1–3.2).** Measured against an
equal-areal-mass *inert* twin, at 55° obliquity the reactive layer measurably
protects the backing plate — main-plate spall ≈21% lower, the gap growing
monotonically over the event — where the same A/B at 0° is an honest null. But the
tungsten rod itself is **not** cut or deflected; it is degraded only modestly (~9%
residual velocity), and the protection arrives mainly through the backing plate
being shoved forward. The "flyer sweep erodes the rod" expectation did not hold,
and is reported as it came out rather than tuned toward.

*Read that 21% with its error bar attached.* On the earlier, smaller-domain
geometry the same A/B read ≈40%. The **sign and the monotonic growth are robust;
the magnitude is not** — and the 0° arm's margin actually flipped sign between
geometries, which is the clearest evidence available of what counts as noise here.
Plausibility, not prediction (see the scope note above).

**Next:** the shaped-charge (HEAT) jet is still a tungsten-rod stand-in — a real
jet model is the open capability gap. See the per-directory `CLAUDE.md` files for
the build order.

## Quick start

```bash
# --- Solver (Python) ---
cd solver && pip install -e .
python -m ballistics_solver.run scenarios/apfsds_vs_rha.yaml --out ../caches/apfsds_vs_rha

# --- Validate a cache against the format spec (no Godot needed) ---
python tools/validate_cache.py caches/apfsds_vs_rha

# --- Quick visual sanity check (matplotlib, no Godot needed) ---
python tools/inspect_cache.py caches/apfsds_vs_rha

# --- Visualize ---
# Open visualizer/ in Godot 4, point it at the cache dir, play.
# Or develop the viewer against visualizer/fixtures/tiny_golden_cache/ with no solver present.
```

On Windows, **`play_viewer.bat`** is the one to double-click: it lists every deck
and lets you pick a color mode (material / velocity / damage / stress) before
launching. Each deck also has its own direct launcher (`play_apfsds_vs_rha.bat`,
`play_apfsds_vs_era_oblique.bat`, …) if you'd rather skip the menu. The `_inert`
twins are the equal-areal-mass controls: play a deck against its twin to see what
the *reactive* layer actually contributes.

Viewer controls: `space` play/pause, `←/→` step, `↑/↓` speed, `C` cycle color
mode, **mouse wheel or `+`/`-` to zoom (about the cursor), drag with middle/right
button to pan, `F` to fit the domain again**, `R` restart, `Esc` quit.

## License

Boyko Non-Commercial License v1.0 (BNCL-1.0) — non-commercial use only; see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Commercial use requires a separate
license from the copyright holder.
