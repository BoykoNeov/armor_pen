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

The repository structure, the cache-format contract, and a tiny golden fixture
are in place. The solver runs **milestone 1: elastic MLS-MPM** (NVIDIA Warp) —
`apfsds_vs_rha` bakes end-to-end on the RTX 5090 and passes `validate_cache`,
showing an elastic impact (rod decelerates and rebounds, plate bulges; no
perforation yet). Next: von Mises plasticity, then damage/spall. See the
per-directory `CLAUDE.md` files for the build order.

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

## License

Boyko Non-Commercial License v1.0 (BNCL-1.0) — non-commercial use only; see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Commercial use requires a separate
license from the copyright holder.
