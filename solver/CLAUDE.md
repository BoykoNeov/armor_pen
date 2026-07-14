# solver/ — CLAUDE.md

Standalone MLS-MPM terminal-ballistics solver. **Reads the root `CLAUDE.md`
first** — this file only adds solver-local notes.

## Hard rules (repeated because they're load-bearing)

- **Never import or reference the visualizer / Godot.** The only output is a
  cache directory per `docs/CACHE_FORMAT.md`. (root §2)
- **Assert the GPU.** After `ti.init(arch=ti.cuda)`, call `mpm.assert_gpu()`.
  A silent CPU fallback is the #1 gotcha (root §11).
- **One unit system: mm-ms-g.** All constants live in `materials.py`. Never
  put raw SI in a kernel. (root §7)
- **Scenarios are data.** New setups are YAML in `scenarios/`, parsed by
  `config.py`. Don't hardcode scenario specifics in kernels. (root §9)

## Layout

| File | Role | Status |
|---|---|---|
| `config.py` | Scenario schema (dataclasses), YAML loader | working |
| `materials.py` | Material library, all constants in mm-ms-g | data stub |
| `cache_writer.py` | Writes manifest.json + frames.bin (the contract) | working |
| `mpm.py` | MLS-MPM transfer kernels + substep loop | **stub** |
| `run.py` | CLI: scenario.yaml → cache dir | plumbing works, bake stubbed |

## Build order (root §9)

Grow the reference MLS-MPM incrementally, validating visually with
`tools/inspect_cache.py` at each step: elasticity → von Mises plasticity →
damage/spall → multi-material armor stack. Don't rewrite from scratch.

## Commands

```bash
cd solver && pip install -e ".[dev]"
pytest                                   # schema smoke tests (no GPU)
python -m ballistics_solver.run scenarios/apfsds_vs_rha.yaml --out ../caches/apfsds_vs_rha
```

## If Taichi fights Blackwell (sm_120)

Don't sink time into it — switch to NVIDIA Warp (root §5). The architecture is
built so that's a contained change: the visualizer never sees the solver.
