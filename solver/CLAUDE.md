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
| `mpm.py` | MLS-MPM transfer kernels + substep loop (Warp) | **elastic + Murnaghan EOS + von Mises plasticity + ductile & brittle damage + multi-material stack + reactive ERA/NERA layer + oblique rod seeding + velocity-graded shaped-charge jet (last touched by milestone 8)** — see below. Milestones 9 and 10 are decks + `tools/` only: they added **zero** kernel code, which is the point — the scenario schema already carried `velocity` and `standoff`. |
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
   damage, milestone 3). Bulk stress reads ~yield; the thin over-read tail that
   used to appear at the compression shock front was a no-EOS property and
   **milestone 8 removed its cause** (PHYSICS §3.5), so the viewer's percentile
   colormap clamp is now a styling default, not a cover for the physics.
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
   is flung to 125.6 mm. **Separation: state the metric or the SIGN flips.**
   Plate-wide median (the original probe's metric): NERA 18.0→**16.1** vs inert
   18.0→**18.5** — NERA ends *tighter*, reproducing the milestone-5 claim that a
   cohesive interlayer holds the plates together. Beside the channel
   (`12<|y-axis|<25`): NERA 18.0→**21.1** vs inert flat 18.5 — locally *wider*, and
   the bulge decays with distance (24.3 @10–20, 21.1 @12–25, 18.4 @15–30). Both are
   one behaviour: **holds the bulge open where the rod passes, holds the plates
   together elsewhere.** **Never** measure inside the channel (perforated plates
   dragged downrange = a huge fake "gap") and **never** compare a plate-wide figure
   to a banded one — they disagree in sign. Visually confirmed
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
   doubles with obliquity. The tungsten **rod is NOT cut or deflected** at either
   angle (rod damage delta −0.5 % at 55°, +2.3 % at 0°): thin few-hundred-m/s
   flyers can't erode a tough long rod, and that failed *a priori* expectation —
   not the plate — is the honest null. **But "not cut" ≠ "not affected":** at 55°
   residual velocity is **−8.5 %** (679 vs 741 m/s), ~75× the noise floor — a real,
   modest degradation. Only 0° is a full rod null (+2.2 %, nothing). Don't lean on
   "penetration +0.8 %" as rod-unaffected evidence: both rods perforate, so that is
   a free-flight *position* at the final frame, not resistance — a slower rod at
   equal position is diverging. Mechanism: the detonation shoves the main plate forward
   (plate body +7.7 mm vs the inert twin at 55°, +1.6 mm at 0° — the shove and the
   protection scale together). NOT chased steeper and det_pressure NOT cranked
   (confirmation-bias tuning, §10). **Error bars are two different sizes and must
   not be conflated:** run-to-run scatter is ≤0.11 % (measured by repeat bakes), so
   the protection is ~150× the numerical floor; but model sensitivity is large —
   the same A/B has read 40 % / 21 % / 16 % across geometry and nose changes.
   Sign robust, magnitude not portable. The earlier "flyer sweep erodes/deflects
   the rod" expectation did not hold — reported honestly as backing-plate
   protection instead.
7. **Shaped-charge (HEAT) jet** — ✅ done. `heat_vs_composite` is a real jet, not
   a tungsten-rod stand-in. What makes it a jet is one initial condition:
   `Projectile.tail_velocity` seeds the projectile **velocity-graded** (7000 m/s
   tip → 2000 m/s tail), computed per particle in `_seed` from the axial distance
   behind the tip — in rod-local coords *before* the §3.2 rotation, same as the
   nose carve, since the direction is uniform and only the magnitude grades. New
   material `copper_jet` (id 6). **No new kernel, no SPH, and no schema bump** (the
   gradient is an initial condition; `material_id` is an existing column).
   `tail_velocity=None` is the default, so all 8 KE decks seed **bit-for-bit** as
   before — verified by A/B re-bake, every metric within 0.022 % (`apfsds_vs_rha`:
   spall 0.1818 vs 0.1818, rod tip 232.3341 vs 232.3345) against a ≤0.11 % scatter
   floor.
   - **Verified (PHYSICS §3.4): the jet stretches at the kinematic rate, +0.1 %.**
     Measured Lagrangianly (persistent particle indices = material labels), because
     tip-to-tail length is confounded by tip erosion. The A/B is
     `heat_vs_composite` vs the committed twin **`heat_vs_composite_uniform`** —
     same copper, geometry, mass, nose, timing, particle count (9210 both), with
     `tail_velocity` omitted, so the gradient is the ONLY variable. Free-flight
     markers 60/110 mm behind the tip separate at **2.093 measured vs 2.083
     predicted mm/µs** (+0.5 %; re-measured after the milestone-8 EOS — the pre-EOS
     value was 2.085, i.e. +0.1 %, so the EOS moved it 0.4 % and the claim stands).
     The jet's body **stretches +32.5 mm** (118.6 → 151.1) where the control's
     **shortens −74.9 mm** (118.6 → 43.7) by erosion — opposite in sign, which is
     the whole claim. The rate also reproduces on **tungsten** (+0.0 %), a 7.5×
     different yield: a kinematic prediction computed from seeded velocities alone
     MUST be material-independent, and it is.
   - **Strength shows up off the rate, and it is signal:** the tungsten control
     holds its markers (−0.003 mm/µs) where the softer copper control *contracts*
     (−0.064) as the shock runs back into it; in tension, tungsten drags on its own
     stretching jet hard enough to accelerate its tail ~5 % (2248 → 2371 m/s) while
     copper transmits ~20× less (straight-line residual 0.001 vs 0.020 mm). Tensile
     coupling scales with yield — which is *why* a real jet is soft copper.
   - **Fluid-like erosion needed no work:** at a 7 km/s stagnation point (~2e5 MPa)
     copper's 200 MPa yield is ~1000× smaller, so the existing von Mises return
     mapping caps deviatoric stress near zero on its own. The advisor's predicted
     "reachability window" (too stiff → elastic recoil; too weak → premature
     breakup) **did not bind** — public textbook copper landed in the good regime
     with **zero tuning**.
   - **Particulation does NOT fire in-window — reported, not claimed.** Damage is
     confined to the leading ~40 mm (that is *erosion* at the armor; the damage
     front marches backward through the jet as it is consumed) and the free-flight
     body reads exactly **0.000**. The arithmetic agrees it shouldn't fire: stretch
     F_xx = 2.0 → equivalent plastic strain ≈0.8 vs copper's 1.5 reserve. Real jets
     particulate at ~100 µs; this deck runs 25 µs, so **staying continuous is the
     correct result**. `damage_threshold` was NOT lowered to force breakup (§10).
   - **Do NOT compare this deck's depth to the old stand-in** — energy-confounded
     twice (graded carries less KE than uniform-7000; copper is half tungsten's
     density). The clean energy-neutral depth test is a **standoff** study, and
     `ArmorLayer.standoff` already exists, so it needs no code. Not done.
   - **~~Known limit, the honest one:~~ fixed in milestone 8** (PHYSICS §3.5). This
     used to read "the volumetric response has no EOS, so jet-tip pressure is the
     least trustworthy quantity in the model." It was true and it is now closed:
     the volumetric response is a **Murnaghan EOS**, monotone and stiffening,
     tangent-matched at `J=1` so KE decks barely move. Measured: the jet tip goes
     `J` 0.0706 → **~0.43** (two figures, not four — it is dt-dependent; see below).
     SPH would not have fixed it either — the §1 SPH hedge stays retired on
     evidence (PHYSICS §1).
   - **Retracted while fixing it:** an interim diagnosis called the old law a
     "softening branch the jet crushes through". That was a **Kirchhoff-vs-Cauchy
     units error** — `½ρv²` is Cauchy, the turnover is Kirchhoff, and the Cauchy
     response is monotone, so nothing ran away. The law was simply far too
     compressible. Don't reintroduce the crush-through framing.

6. **Milestone 8 — equation of state (PHYSICS §3.5).** Murnaghan
   `p(J) = (K₀/K′)(J^−K′ − 1)`, `K₀ = λ+µ`, `K′ = 4` in `materials.EOS_KP`. Zero
   new per-material constants. Two things to know before touching it:
   - **The CFL bound is no longer the rest sound speed.** The EOS stiffens, so
     `c ~ J^(−K′/2)`; `bake` sizes `dt` from the deck's predicted stagnation
     compression plus `EOS_CFL_J_MARGIN`, then *measures* the sound speed actually
     reached and warns on a breach. Read that audit line on every new deck —
     especially a faster one. It is how the sweep geometry's breach was caught.
   - **What is still wrong, and it is velocity-dependent.** Murnaghan is a *cold*
     curve with no shock heating: vs copper's public Hugoniot it reads 0.93× at
     `J=0.9` but 0.68× at a 7 km/s equilibrium. So the EOS **shrank** the
     velocity-dependent pressure error (~1.70× → ~1.37× across the jet's own
     gradient); it did not remove it. A velocity sweep still inherits it. The real
     fix is Mie-Grüneisen + per-material `c₀/s/Γ`.
   - **Do NOT quote tip-`J` to four decimals — it is not dt-converged.** 0.3923 (47
     substeps) → 0.3971 (98) → 0.4315 (240), rising as the substep shrinks, because
     what it measures is the undamped **shock ring**, not the EOS. An earlier
     "1.2 % converged" claim in the docs came from the 47→98 pair alone and was
     wrong; 98→240 moves it +8.7 %. Two points are not a convergence study. The
     ceramic figure (0.9910) *is* robust — quote that one.

7. **Milestone 9 — velocity sweep vs the hydrodynamic asymptote (PHYSICS §3.7).**
   Ten `sweep_*` decks, `{tungsten, copper} × {1500..7000} m/s` into an identical
   120 mm semi-infinite RHA half-space. Measure with
   `tools/measure_penetration.py` (solver-free; finds the penetrator as whatever
   moves at t=0, reads `v` from frame 0, derives its window from the erosion curve).
   - **The claim is the RATIO, not the absolute.** Both arms land at the same
     0.937× of their own asymptote, so the shortfall is the model's, not the
     material's, and cancels: measured 1.1614 vs 1.1609 predicted from density,
     **0.04 %**.
   - **Read the trend, not the third decimal.** `u/v` is not fully dt-converged
     (163→500 substeps moves copper@7000 +2.6 %), and `u` is the *erosion-front*
     velocity, so it inherits erosion's partly-numerical component. The defence is
     that both are systematic and cancel in a trend/ratio — not that the metric is
     clean.
   - **Don't reuse a perforating deck for this.** `u` is only a penetration
     velocity while the penetrator is inside the target; after perforation the
     leading edge is a free residual flying at ~`v`, which fits a *beautiful*
     straight line (`apfsds_vs_rha` reads u/v=0.729 at R²=0.999, above a ceiling
     that is physically unreachable). R² cannot catch that; the tool's back-face
     check does.
   - **Honour the `steady` flag.** tungsten@1500 is R²=0.985 and excluded: its rod
     *decelerates* (Tate) rather than reaching steady state, because tungsten's
     yield is 7.5× copper's. That is physics, not a probe bug — and not a deck to
     re-tune until it "works".

8. **Milestone 10 — standoff (PHYSICS §3.8).** The energy-neutral depth experiment
   milestone 7 deferred. Four shipped decks `standoff_s00/s30/s60/s90` (S = 0/30/60/90
   mm of free flight before an identical 150 mm RHA half-space) plus six
   `standoff_conv_*` decks that exist to measure how wrong the shipped four are.
   Zero new solver code — `ArmorLayer.standoff` did the whole job. Measure with
   `tools/measure_standoff.py --family` / `--convergence`.
   - **The shipped decks under-read the effect ~2.3× on the excess and are NOT
     grid-converged. Do not quote their ratio as the model's answer.** Measured
     S90/S0 = 1.229 vs 1.536 predicted a priori. The physics lives in the
     `standoff_conv_*` decks; the shipped ones are for playback and the trend's shape.
   - **The jet is only 8 cells across, and stretching thins it to ~3.** This is the
     transferable lesson: `cells across the jet` is the controlling parameter for any
     jet DEPTH claim, and it is reached identically by refining `dx` (1.229 → 1.383 →
     1.429 at 8/12/16 cells) or by fattening the jet (6 mm at the shipped `dx` = 16
     cells → 1.501, within scatter of the prediction). The derivation is
     diameter-independent, which is what licenses the second route.
   - **This is why §3.4 was right to refuse a depth comparison** — for a second,
     independent reason it did not know about. Kinematic claims are safe (free-flight
     markers don't lean on grid coupling); depth claims are not.
   - **Match on consumed fraction, never on lab time.** Depth at the end of the window
     FALLS with standoff (105 → 80 mm) because a longer standoff impacts later. The
     obvious metric reports the opposite sign of the real effect.
   - **No Richardson number is quoted, deliberately.** The observed order swings ~0.7
     to ~5 depending on the matching point. A fifth grid would not fix it (the
     conditioning is the small high-res increments, not the count). Report the trend.
   - **Don't read the saturation as a rollover.** The under-resolved increments taper
     (4.7 → 3.4 → 3.1 mm), which mimics the textbook standoff optimum. This jet does
     not particulate (§3.4, and free-flight damage at S=90 is exactly 0.0000), so no
     optimum is reachable and none was manufactured.

Don't rewrite from scratch. The full solver arc (milestones 1–10) is done.

**Stale-number correction (measured 2026-07-16):** the "~16 % RHA spall" quoted in
the milestone 3/4 notes above and the 15.6 % in milestone 6 were measured at
**96 637 particles**, before the plate-not-block geometry change. `apfsds_vs_rha`
now seeds **135 557** particles and spalls **18.2 %**. The ordering and every
conclusion those milestones drew are unaffected; only the absolute moved. Do not
compare a figure across geometries — see the memory note on rebakes invalidating
documented results.

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
