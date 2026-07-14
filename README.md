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

The solver (Python + Taichi, MLS-MPM) *bakes* a scenario offline to a cache on
disk. The visualizer (Godot 4) plays that cache back. Neither imports the
other; the only bridge is the language-neutral cache format specified in
[`docs/CACHE_FORMAT.md`](docs/CACHE_FORMAT.md). This decoupling is deliberate —
the solver is disposable and may be rewritten (Taichi → Warp → CUDA/Rust)
without touching a line of visualizer code.

## Layout

```
docs/         CACHE_FORMAT.md (the contract) + PHYSICS.md
solver/       standalone MLS-MPM solver; knows nothing about Godot
visualizer/   Godot 4 player; knows only how to read the cache
tools/        language-neutral cache helpers (validate, inspect); depend on neither side
caches/       gitignored bake outputs (large)
```

## Status

Early scaffold. The repository structure, the cache-format contract, and a
tiny golden fixture are in place. The solver kernels are stubs — see the
per-directory `CLAUDE.md` files and issue tracker for what's next.

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

MIT — see [`LICENSE`](LICENSE).
