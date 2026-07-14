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
| `materials.py` | Material library, all constants in mm-ms-g | data (all fields now consumed: elasticity, yield, ductile `damage_threshold`, `brittle`, reactive block) |
| `cache_writer.py` | Writes manifest.json + frames.bin (the contract) | working |
| `mpm.py` | MLS-MPM transfer kernels + substep loop (Warp) | **elastic + von Mises plasticity + ductile & brittle damage + multi-material stack + reactive ERA/NERA layer (milestone 5)** — see below |
| `run.py` | CLI: scenario.yaml → cache dir | working (Warp init + GPU assert + bake) |

## Build order (root §9) — where we are

Grow the reference MLS-MPM incrementally, validating visually with
`tools/inspect_cache.py` at each step:

1. **elasticity** — ✅ done. Fixed-corotated MLS-MPM in Warp; rod + plate seeded
   from the deck, elastic *impact* (rod decelerates and rebounds, plate bulges;
   **no** perforation — correct for elastic-only). `apfsds_vs_rha` bakes clean
   on the RTX 5090 and passes `validate_cache`.
2. **von Mises plasticity** — ✅ done. Perfectly-plastic radial return in
   log-strain space (2×2-SVD `_return_mapping`, run per particle after G2P);
   isochoric, no hardening. `apfsds_vs_rha` bakes clean on the RTX 5090 (no
   NaN/Inf), passes `validate_cache`, and the rod **mushrooms** (length 60→42 mm,
   width 8→40 mm) while the plate craters — **no perforation hole** (that needs
   damage, milestone 3). Bulk stress reads ~yield; a thin over-read tail at the
   compression shock front is a fixed-corotated/no-EOS property, tamed
   viewer-side by a percentile colormap clamp (see `_von_mises`, PHYSICS §3).
   Equivalent plastic strain accumulates into an internal `alpha` array (guarded
   against inversion/over-compression spikes via `MAX_DALPHA`) — wired for
   milestone 3; the `damage` cache column stays 0 until then.
3. **damage/spall** — ✅ done. `_update_damage` latches `damage=1` once `alpha`
   (equivalent plastic strain) crosses the material's `damage_threshold`; `_p2g`
   then drops that particle's stress term so it becomes a cohesion-free free
   fragment (mass + momentum only, still grid-coupled so it collides but can't
   hold tension). `_return_mapping` keeps running on spalled particles to pin
   their deviatoric F to yield (no F blow-up / NaN readout). Fixed particle count
   — spall = flagged + detached, never created/destroyed (contract §4); no schema
   bump (filled the existing `damage` zeros column). `apfsds_vs_rha` bakes clean
   on the RTX 5090 (no NaN/Inf), passes `validate_cache`, and shows a
   **penetration channel lined with spall + a crater-lip spall spray** — ~16% of
   RHA spalls, localized to the impact axis (not whole-plate). The `MAX_DALPHA`
   guard held: shock-front particles don't spall spuriously. Rod erodes/perforates
   into the plate (leading edge 99→137 mm over the window). Threshold reachability
   is the tuning knob: too high → no spall, too low → whole plate flags.
4. **multi-material armor stack** — ✅ layered + spaced + brittle done (NERA
   deferred to its own milestone). The seeding loop and per-particle constitutive
   arrays (`mu/lam/yield/dthr/brittle/mass/mat_id`) already handled multiple
   layers and standoff gaps from milestone 1; this milestone added the two decks
   (`apfsds_vs_composite` bonded RHA/ceramic/RHA, `apfsds_vs_spaced` spaced RHA)
   and the **brittle fracture model** that was the one genuine physics gap.
   Brittle materials (`brittle: true`, ceramics) latch damage on a *stress*
   trigger — von Mises Cauchy ≥ `yield_strength`, or max tensile principal ≥
   `0.1·yield_strength` — independent of plastic strain, so they shatter with
   ~zero plastic flow (PHYSICS §3). Without it, ceramic was a near-indestructible
   ductile wall at KE velocities (empirically confirmed before implementing).
   Ductile metals are untouched: with the `brittle` flag off the `alpha` path is
   unchanged, and `apfsds_vs_rha` RHA spall stays ~16% as before (spall %
   verified, not bytes — MPM grid `atomic_add` ordering isn't deterministic).
   All three KE decks bake clean on the RTX 5090 and pass `validate_cache`;
   verified visually (M:\claud_projects\temp\m4_probe\*.png). No schema bump —
   `brittle` is solver-internal, cache columns unchanged.
   - **Probe lesson:** `heat_vs_composite` (7000 m/s HEAT stand-in) is the *wrong*
     multi-material test — its plate fly-away / wall pile-up are momentum
     artifacts of the deferred jet model, not multi-material issues. Use the KE
     decks. Plate anchoring is NOT needed at sane KE velocity (stacks stay put).
5. **NERA/ERA reactive layer** — ✅ mechanism done (0° honest; obliquity deferred
   to milestone 6). A reactive filler (`era_filler`, `reactive: true`) ignites on
   shock (`det(F) < ignition_compression`) and releases an isotropic detonation
   overpressure for `burn_time` — a **pressure source term in `_p2g`** that flings
   the sandwich plates apart through the ordinary grid (emergent, not a scripted
   rod kick). Reactive particles run a self-contained elastic → detonation →
   debris state machine (`_update_reactive`) and are excluded from the
   ductile-spall path (else the soft filler would spall in the same substep it
   should ignite and silently no-op the detonation — see the reactive note in
   `mpm.py`). A persistent NERA bulge is the unignited soft-elastic branch held
   open — `ignition_compression=0` so the filler *never* ignites — **not** merely
   `detonation_pressure=0` (that still ignites on the impact shock and collapses
   to debris: bulge-then-collapse). Untested — no NERA deck baked yet. Two
   stability guards, reactive-particles-only: burning **and** spent
   filler get `F` pinned to identity (no elastic memory; return-mapping skips
   them so `F` would otherwise drift to inf and overflow the host readout), and
   speed is clamped at `REACTIVE_VMAX` (the `F`-independent source would otherwise
   drive unconfined debris to a CFL-breaking ~14 km/s once the plates separate).
   All gated on `reactive > 0.5`, so the three non-reactive KE decks are byte-for-
   effect unchanged. Both ERA decks bake clean on the RTX 5090 (no NaN/Inf) and
   pass `validate_cache`; verified visually (M:\claud_projects\temp\era_react_vs_inert.png).
   - **Honest limitation (verified, not a bug):** at **0° the reactive layer does
     not meaningfully degrade the rod.** Measured against an equal-areal-mass
     inert twin (`era_filler_inert`), residual penetration into the main plate is
     within noise (15.7 vs 15.5 mm; residual v, rod damage, main-plate spall all
     within scatter). Physical reason: at normal incidence the detonation flings
     the plates *laterally, symmetric about the rod axis* — the debris sweeps
     sideways and never crosses the rod to cut it. Real ERA needs **obliquity**
     (plates sweep across a long rod). The A/B decks (`apfsds_vs_era` /
     `apfsds_vs_era_inert`) are byte-identical in geometry, areal mass, and timing,
     so the near-zero delta cleanly isolates the reactive contribution. See
     PHYSICS §3.1.
6. **Oblique reactive armor** — next. Rotate the rod *rectangle* (`angle_deg`
   currently tilts only the projectile velocity) + seed oblique slabs, so the
   lateral flyer sweep can actually erode/deflect the rod — where ERA earns its
   keep.

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
