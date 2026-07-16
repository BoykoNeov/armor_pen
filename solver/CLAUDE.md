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
| `mpm.py` | MLS-MPM transfer kernels + substep loop (Warp) | **elastic + von Mises plasticity + ductile & brittle damage + multi-material stack + reactive ERA/NERA layer + oblique rod seeding (milestone 6)** — see below |
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
   to debris: bulge-then-collapse). ✅ **Baked and verified** (`nera_filler`,
   deck `apfsds_vs_nera`, geometry-identical to the two ERA decks — all three seed
   at 179189 particles): filler damage is **0.000 across all 550 frames** (never
   ignites, never spalls). That is the decisive claim and it has now reproduced
   across two rebakes (geometry change, then the nose). Supporting signal is
   cohesion: filler thickness reaches only 39.5 mm, where the inert twin's shreds
   to damage 0.462 and spreads to 83.5 mm and the reactive twin's latches 1.000 and
   is flung to 125.6 mm. The sandwich opens 18.0→21.1 mm and holds — but **measure
   the bulge BESIDE the channel, not in it** (inside, the plates are perforated and
   dragged downrange, which reads as a huge fake "gap"), and the bulge **decays with
   distance** (24.3 mm at a 10–20 mm band, 21.1 at 12–25, 18.4 at 15–30) so no single
   separation number is meaningful. **Correction:** the old claim that NERA separates
   *less* than the inert twin (21.1 vs 24.2) is **reversed** — the inert twin reads
   18.0→18.5 flat at every band and does not bulge at all; the 24.2 reproduces only
   as *NERA at a different band*, so those two arms were not sampled alike. NERA
   bulging *more* is what the construction predicts (it is perfectly elastic — it
   springs; the inert filler yields/shreds and dissipates). Visually confirmed
   (`--shots`): the interlayer is still large coherent bent slabs, spall spray
   coming from the steel plates, not the filler. Stable, no NaN, passes
   `validate_cache`. **Caveat (PHYSICS §3.3):** the rod also ends up
   shallower/slower/more-damaged vs NERA than vs either ERA twin, but that is
   confounded *by construction* and is NOT an armor-performance claim —
   `reactive=True` skips **both** `_return_mapping` (plasticity) and
   `_update_damage` (spall), so `nera_filler` can neither yield nor break and its
   `yield_strength` / `damage_threshold` are **dead fields**. It illustrates the
   damage model (spalled particles keep momentum but drop their stress term in
   `_p2g`, so they stop resisting), not armor. Two
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
6. **Oblique reactive armor** — ✅ done. `_seed` rotates the projectile by
   `angle_deg` about its tip so the rod strikes nose-first along its velocity
   (armor slabs stay vertical — only the relative rod/plate-normal angle is
   physical, so this is frame-equivalent to tilting the slabs and leaves the M1–M5
   armor seeding untouched; the *rotation* at `angle_deg=0` is exact identity,
   though the rod it rotates is now nose-carved at every angle (see the nose note
   below), so 0° decks no longer seed bit-for-bit as at M1–M5). New decks
   `apfsds_vs_era_oblique` (+ `_inert` twin) at 55°. **Result (verified,
   PHYSICS §3.2): at 55° the reactive layer measurably protects the backing plate
   — main-plate spall ≈16 % lower (0.137 vs 0.163)** — and the *same* mechanism
   buys ≈8 % at 0°, so the plate-side benefit is real at both angles and roughly
   doubles with obliquity. But the tungsten **rod is NOT cut or deflected** at
   either angle (penetration delta +0.8 % at 55°, +0.3 % at 0°; rod damage −0.5 %
   / +2.3 %): thin few-hundred-m/s flyers can't erode a tough long rod. **That —
   not the plate — is the honest null.** Residual velocity is the one real rod-side
   effect (−8.5 % at 55°). Mechanism: the detonation shoves the main plate forward
   (plate body +7.7 mm vs the inert twin at 55°, +1.6 mm at 0° — the shove and the
   protection scale together). NOT chased steeper and det_pressure NOT cranked
   (confirmation-bias tuning, §10). **Error bars are two different sizes and must
   not be conflated:** run-to-run scatter is ≤0.11 % (measured by repeat bakes), so
   the protection is ~150× the numerical floor; but model sensitivity is large —
   the same A/B has read 40 % / 21 % / 16 % across geometry and nose changes.
   Sign robust, magnitude not portable. The earlier "flyer sweep erodes/deflects
   the rod" expectation did not hold — reported honestly as backing-plate
   protection instead.
Don't rewrite from scratch. **Milestone 7 = HEAT/shaped-charge jet** — the open
capability gap; `heat_vs_composite` is still a tungsten-rod stand-in.

### Rod nose — geometry fix, not a milestone (done)

The projectile was seeded as a flat-faced rectangle; a real APFSDS is **pointed**.
`_seed` now carves a nose (`nose_shape`: conical (default) / ogive / blunt,
`nose_length` default 1.5 calibers) out of the rod rectangle before the rotation.
Defaulted in `config.py`, so all 9 decks went sharp with **zero YAML edits**.

Verified against a `blunt` control twin (same deck, same probe, nose the only
difference): **final penetration is unchanged** (232.3 vs 232.7 mm, −0.2 %) — the
textbook result, since the nose is consumed in ~1 µs and depth is then nearly
nose-shape-independent (eroding/hydrodynamic regime). It is **geometric realism,
not a penetration-accuracy fix.** Where it shows is the **early crater**: 63 % less
RHA spall at 6.4 µs (0.004 vs 0.010) — the pointed rod cleaves in where the flat
face slaps — converging by ~24 µs once both have mushroomed. Rod ends ~10 % less
damaged. **Confound:** the nose is carved OUT, so a pointed rod is ~10 % lighter at
equal length (not compensated — that would change the scenarios, which were to be
kept). All 9 decks rebaked, validated, finite; A/B twins still match
particle-for-particle. See PHYSICS §1.2.

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
