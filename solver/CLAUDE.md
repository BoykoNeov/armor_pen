# solver/ — CLAUDE.md

Standalone MLS-MPM terminal-ballistics solver. **Reads the root `CLAUDE.md`
first** — this file only adds solver-local notes.

## Hard rules (repeated because they're load-bearing)

- **Never import or reference the visualizer / Godot.** The only output is a
  cache directory per `docs/CACHE_FORMAT.md`. (root §2)
- **Assert the GPU.** After `wp.init()`, call `mpm.assert_gpu(device)`. A
  silent CPU fallback is the #1 gotcha (root §11).
- **One unit system: mm-ms-g.** All constants live in `materials.py`. Never
  put raw SI in a kernel. (root §7)
- **Scenarios are data.** New setups are YAML in `scenarios/`, parsed by
  `config.py`. Don't hardcode scenario specifics in kernels. (root §9)

## Layout

| File | Role | Status |
|---|---|---|
| `config.py` | Scenario schema (dataclasses), YAML loader | working |
| `materials.py` | Material library, all constants in mm-ms-g | data (used by mpm; plasticity/damage fields unused until those milestones) |
| `cache_writer.py` | Writes manifest.json + frames.bin (the contract) | working |
| `mpm.py` | MLS-MPM transfer kernels + substep loop (Warp) | **elastic (milestone 1)** — see below |
| `run.py` | CLI: scenario.yaml → cache dir | working (Warp init + GPU assert + bake) |

## Build order (root §9) — where we are

Grow the reference MLS-MPM incrementally, validating visually with
`tools/inspect_cache.py` at each step:

1. **elasticity** — ✅ done. Fixed-corotated MLS-MPM in Warp; rod + plate seeded
   from the deck, elastic *impact* (rod decelerates and rebounds, plate bulges;
   **no** perforation — correct for elastic-only). `apfsds_vs_rha` bakes clean
   on the RTX 5090 and passes `validate_cache`.
2. **von Mises plasticity** — next. Radial-return in the P2G stress; this is
   what lets the metal flow/mushroom instead of bouncing.
3. **damage/spall** — detach particles past `damage_threshold` (fixed particle
   count; flag via the `damage` attribute).
4. **multi-material armor stack** — layered/spaced/NERA targets.

Don't rewrite from scratch.

## Commands

```bash
cd solver && pip install -e ".[dev]"     # installs warp-lang; or PYTHONPATH=src for a no-install run
pytest                                   # schema smoke tests (no GPU)
python -m ballistics_solver.run scenarios/apfsds_vs_rha.yaml --out ../caches/apfsds_vs_rha
python -m ballistics_solver.run <deck> --out <dir> --cpu   # CPU fallback for debugging
```

## Toolchain: Warp, not Taichi (decided & migrated 2026-07-14)

**Taichi has no wheel for the machine's Python 3.14** (`pip install taichi` → no
matching distribution), so the Taichi-default stack never installed here.
**Warp (`warp-lang` 1.15.0)** is the engine: it detects the RTX 5090 as
`cuda:0` / **sm_120** and the MLS-MPM kernels compile and run on the GPU
(`assert_gpu` guards against a silent CPU fallback). This is exactly the swap
root §5 anticipates, and it cost **zero** visualizer code — the cache is the
only bridge. `pyproject.toml` depends on `warp-lang`; `mpm.py`/`run.py` use
Warp. Milestone-1 elastic bake verified end-to-end on the real GPU.

## If Taichi fights Blackwell (sm_120)

Don't sink time into it — switch to NVIDIA Warp (root §5). The architecture is
built so that's a contained change: the visualizer never sees the solver.
