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
| `materials.py` | Material library, all constants in mm-ms-g | data (all fields now consumed: elasticity, yield, ductile `damage_threshold`, `brittle`, reactive block, and the `shock` Hugoniot block `s`/`Γ₀` — milestone 13) |
| `cache_writer.py` | Writes manifest.json + frames.bin (the contract) | working |
| `mpm.py` | MLS-MPM transfer kernels + substep loop (Warp) | **elastic + Mie-Grüneisen EOS with an energy equation + von Mises plasticity + ductile & brittle damage + multi-material stack + reactive ERA/NERA layer + oblique rod seeding + velocity-graded shaped-charge jet + artificial viscosity (default ON) + working free-slip walls on all four sides — last touched by milestone 13** — see below. Milestones 9 and 10 are decks + `tools/` only: they added **zero** kernel code, which is the point — the scenario schema already carried `velocity` and `standoff`. |
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
   `_p2g`, so they stop resisting), not armor.
   **⚠️ M12 SUPERSEDES THIS CAVEAT'S PREMISE — and its numbers.** `nera_filler` is
   non-reactive and ductile now; both fields are **live** and the A/B is
   single-variable (milestone 12 below). The confound is gone — but so is the
   material, so **the rod deltas this caveat refers to are NOT re-measured and must
   not be quoted**. Un-confounding them does not promote them to an armor claim; the
   "it illustrates the damage model, not armor" reading is the part that survives.
   Two
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
     the volumetric response got a **Murnaghan EOS**, monotone and stiffening,
     tangent-matched at `J=1` so KE decks barely move. Measured: the jet tip goes
     `J` 0.0706 → **~0.43** (two figures, not four — it is dt-dependent; see below).
     SPH would not have fixed it either — the §1 SPH hedge stays retired on
     evidence (PHYSICS §1).
     **⚠️ Milestone 13 SUPERSEDED the law and the number: the EOS is now
     Mie-Grüneisen** (Murnaghan survives only as the pole guard's fallback branch),
     and the jet tip reads **0.5226**. M8's limit — "a *cold* curve with no shock
     heating", 0.68× copper's Hugoniot at 7 km/s — is what M13 closed. The `~0.43`
     is the Murnaghan-era value; the "don't quote it to four decimals" posture was
     right and this move is the fourth demonstration of it.
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
   - **Do NOT quote tip-`J` to four decimals.** 0.3923 (47 substeps) → 0.3971 (98)
     → 0.4315 (240). An earlier "1.2 % converged" claim in the docs came from the
     47→98 pair alone and was wrong; 98→240 moves it +8.7 %. Two points are not a
     convergence study. The ceramic figure (0.9910) *is* robust — quote that one.
   - **⚠️ This note used to say the climb "measures the undamped shock ring". It
     does not — milestone 11 falsified that.** The number is the **first impact
     transient** (not the jet-tip stagnation, and not the ceramic interface); the
     climb is ordinary coarse-`dt` error that flattens by ~400 substeps with AV
     *off*; and the ring is ~0.9 %, far too small to be the story. The 47/98
     points are also **not reproducible** now — the solver takes
     `min(deck_dt, cfl_dt)` and this deck's CFL floor is 240 substeps, so those
     came from a different `EOS_CFL_J_MARGIN` era (0.8 → 0.55 → 0.35). See §3.5/§3.9.

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

10. **Milestone 12 — the NERA filler's dissipation path (PHYSICS §3.6.1/§3.6.2).**
   Zero kernel code; `materials.py` only. Read §3.6.1 before trusting any
   `worst live J` in this repo.
   - **`nera_filler` was mis-encoded, and the fix was written in the deck's own
     header.** It was `reactive=True` with `ignition_compression=0` — a filler that
     never ignites. But `reactive` *also* gates out `_return_mapping` (L538) and
     `_update_damage` (L653), and that gate's stated reason is "must not spall
     before it detonates" — which **cannot apply to a filler that never detonates.**
     It inherited a gate written for its igniting twin and had no dissipation path.
     Now `reactive=False`, `damage_threshold` 0.02 → **3.0**, `yield_strength`
     unchanged at 50 MPa. Both fields were **dead**; both are now **live**.
   - **The headline finding: `J=0.2159` was never a bulk state.** PHYSICS §3.6 said
     the filler "genuinely reaches ~79 % volume loss" and sized `EOS_CFL_J_MARGIN`
     (~2.5× substeps on **every** deck) from it. Measured: **25 of 36 966 particles**
     below J=0.5 (**0.068 %**), median **0.9932**, and the mean live J **never** drops
     below 0.9495. `worst live J` is a **min over every particle over every frame** —
     it is one particle, and §3.9 already warned that a min-over-a-set is the wrong
     instrument.
   - **They are not even in the interlayer.** The filler seeds at x=156.1–167.9; the
     crushed particles sit at **x=200.8–202.0**, inside the *main plate's* crater,
     pinned by the rod tip (leading edge 201.87) **34 mm downrange across the standoff
     gap**. 68.95 % of the filler is still behind the back plate.
   - **The "converged" check refined the wrong axis.** 0.2159@110 vs 0.2120@336 is the
     same trapped particles both times; `dt` cannot dissolve a geometric trap. Fourth
     time this repo has been bitten by a convergence claim — see the memory note.
   - **The CFL bonus did NOT arrive, and it never could.** Ratio 0.442 vs the jet's
     0.713, so nera **still binds** and the margin **stays 0.35**. The sub-0.3
     particles carry `alpha`=**2.91 of a 3.0 reserve (97 %)** — they are not failing
     to yield, they are **saturating** the yield surface and are still crushed.
     Plastic flow is isochoric, so it cannot relieve volumetric confinement however
     hard it engages. **Relief needs a VOLUMETRIC (compaction) criterion.** Not done.
   - **Do NOT compare `worst live J` across arms with different `damage_threshold`.**
     `era_filler_inert` reads a lovely 0.6813 *because* it shreds — the crushed
     particles spall out of the **live** set. Lowering `dthr` to buy CFL headroom is
     buying it with cohesion, i.e. tuning toward the answer (§10).
   - **Counterintuitive and measured:** spall makes worst-J **worse** (0.2421 with
     `dthr=3.0` vs 0.2903 with spall off). A spalled particle drops its stress term in
     `_p2g` and stops resisting, concentrating the crush on its live neighbours.
   - **Cohesion survives — the go/no-go.** vs the shredding twin (one field apart):
     spall **18.65 % vs 69.59 %**, coherent **66.0 % vs 30.4 %**. A plasticity-only
     control attributes the bulge change to **plasticity, not spall** (53.8 vs 53.6 mm
     banded separation at 0 % vs 18.65 % spall). The control is **not** the ship
     candidate: `dthr=∞` scores better on CFL (44 %) but re-creates a **dead field**,
     the exact defect M12 removes.
   - **⚠️ The sequencing rationale FAILED — carry this into M13.** M12 was done first
     so Mie-Grüneisen would land where every material is inside its Hugoniot's valid
     range. `nera_filler`'s pole is at `J = 1−1/s ≈ 0.5`; M12 leaves worst live J at
     **0.2421**, still far past it. **The MG pole guard stays load-bearing on this
     deck** and must be designed, not assumed.
   - **What did NOT get re-measured** (stated, not buried): §3.3's plate-separation
     figures (16.1/21.1) and rod deltas (tip 261.8 etc.) were measured on the pre-M12
     filler. An independent probe that reads exactly 18.000 at t=0 does **not**
     reproduce 16.1/21.1 even on the pre-M12 bake (it gets 14.1/13.3 — opposite sign
     beside the channel). M5's probe was ad-hoc and never committed. Don't quote them.

9. **Milestone 11 — artificial (shock) viscosity (PHYSICS §3.9).** Built the fix
   milestone 8 asked for, measured it, and **retired milestone 8's diagnosis**.
   `SolverParams.av_c_q` / `av_c_l`.
   > **⚠️ SUPERSEDED BY MILESTONE 13 — AV IS NOW ON BY DEFAULT** (`av_c_q=1.5`,
   > `av_c_l=0.6`). M11 weighed +57 % substeps against damping a ~0.9 % ring and
   > correctly called it a bad trade — but **that was the wrong question**. AV's real
   > job is carrying shock heating into `e`; without it MG lands on the *isentrope*,
   > not the Hugoniot (`p/p_H` 1.000 → 0.923). M11's own closing note said exactly
   > this would happen ("the moment a thermal term lands, AV heating SHOULD raise
   > thermal pressure"). **The bullets below are M11's reasoning, kept because it was
   > right at the time — do not act on the "off" parts.** Two are stale twice: AV
   > shipped for **all 30** decks, not just the jet; and the "≤0.20 % on KE decks"
   > inertness figure was measured under **Murnaghan at matched dt**, while M13's
   > `apfsds_vs_rha` spall moved 18.2 → 25.1 % (MG + AV-on + the boundary fix — three
   > variables, so it does NOT isolate AV, but the old figure is not evidence about
   > today).
   - **Default off is a measured decision, not laziness.** AV costs **+57 %
     substeps** (240 → 377 on the jet: it raises the signal speed the CFL bound is
     sized from) to damp a ring that is **~0.9 % peak-to-peak**. With the
     coefficients at 0 the term is identically zero *and* the CFL bound is
     untouched, so all 30 baked decks are bit-for-effect the pre-AV solver.
     `tests/test_artificial_viscosity.py` pins that. Turning it on means rebaking
     and re-measuring everything — do it only with a reason.
   - **AV goes in `_p2g`, NEVER in `_fixed_corotated_pft`.** The latter is the
     constitutive law: it feeds the brittle triggers and is mirrored on the host by
     `_von_mises`, which sees only `F` and cannot know `div v`. A numerical term
     there would shatter ceramic for a numerical reason and break the two-path pin.
     A test asserts the law stays rate-free. The `stress` column excludes `q`.
   - **Frame-cadence metrics ALIAS the ring — do not try to measure it from a
     cache.** Period ~159 substeps vs 400–1600 substeps per frame. Use
     `mpm.bake(..., j_trace={...})`, the windowed per-substep debug hook, and trace
     a SINGLE particle: a min-over-a-set traces the envelope of out-of-phase
     oscillations and hides the ring. The CFL audit has the same blind spot.
   - **AV is inert below hypervelocity** (`apfsds_vs_rha` moves ≤0.20 % at matched
     dt, i.e. the scatter floor) because KE decks barely compress. So a future
     Mie-Grüneisen milestone can switch it on for the jet without re-tuning the KE
     decks — which is the reason it is kept at all.

11. **Milestone 13 — Mie-Grüneisen, the EOS gets an energy equation (PHYSICS §3.10).**
   `p(J,e) = p_cold(J) + Γ₀ρ₀e`, two new per-material constants (`s`, `Γ₀` in
   `ShockEOS`, **required, no default** — a new material must state its Hugoniot
   rather than silently inherit copper's). Closes the limit §3.5 named for itself.
   Read §3.10 before touching any of it; five things to know:
   - **There is NO cheap MG — the cold curve alone is a REGRESSION** (0.63× copper's
     Hugoniot at jet stagnation, *below* Murnaghan's 0.73×; the `(1−Γη/2)` factor
     subtracts pressure). **The whole benefit lives in the ENERGY EQUATION**, solved
     in closed form because MG is linear in `e`. That vindicates §3.9's
     AV-before-MG ordering: AV work is what feeds `e`. **AV is therefore ON by
     default now**, reversing M11's measured "off" — its reason (the work dissipated
     to nothing) no longer exists. `c₀` needs **zero** new constants (`√(K₀/ρ₀)`,
     within 1–10 % of public shock data).
   - **`p(J,e_H)=p_H` is a TAUTOLOGY** — built into the MG algebra, true for any Γ.
     It validates the algebra, not the scheme. The **1-D piston** earns the
     milestone: `p/p_H` = 1.000/1.000/0.999 and `u_s = c₀+s·u_p` matched to <2 %
     having been fitted to nothing. **The falsifier is the point:** no AV work fed
     to `e` → 0.923 (the isentrope); no `e` at all → 0.755 (the cold curve).
   - **The pole guard IS load-bearing on nera** (worst live J **0.5434** vs
     `J_sw`=0.55 — inside the fallback, 2 live particles), exactly as §3.6.2
     predicted a priori. An interim AV-off bake saw 1 particle and was briefly read
     as "a backstop"; that did not survive the shipped config. **Never silence it by
     lowering `MG_F_SWITCH`.** It naming `copper_jet` or `rha` = M13 quietly not in
     effect where it matters.
   - **MG relieved the NERA crush M12 could not:** worst live J **0.2421 → 0.5434**.
     §3.6.2 said relief "needs a VOLUMETRIC criterion, not a deviatoric one" — MG's
     thermal pressure `Γρ₀e` **is** that criterion. M12 was right about the *kind* of
     mechanism and wrong that MG would not supply it.
   - **`EOS_CFL_J_MARGIN` STAYS 0.35.** Decks use 5–22 % of budget, so it is
     conservative = **correct, merely slow** (root §1: bake cost is irrelevant).
     Recalibrating a *global* stability constant inside a change that already moves
     the EOS **and** the boundary condition is three variables at once. Those
     percentages are the evidence base for doing it later, as its own A/B.
     > **✅ DONE — milestone 14 took this follow-up, and the deferral was right for
     > the wrong reason.** The 5–22 % was not the constant being cautious; it was
     > **M13 succeeding**. MG relieved the nera crush (worst live J 0.2421 → 0.5434)
     > that the margin had been sized against, so the constant went stale at M13 and
     > the audit line said so. The formula, not the constant, was the defect —
     > `EOS_CFL_J_MARGIN` is retired for `EOS_CFL_P_MARGIN`. See milestone 14 below.
   - Schema **v1 → v2**: `internal_energy`, **specific, per unit MASS (J/kg)**. NOT
     `temperature` — that needs a per-material `c_v` *and* would under-read exactly
     in the shear zones a viewer most wants (the `e` update drops plastic
     dissipation; stated in §3.10 and CACHE_FORMAT §2, not discovered later).
   - **The `e ≥ 0` clamp: judge it RELATIVE to `e`, never in absolute J/kg.** The
     bake report's verdict was `abs(e_worst) < 1.0` and it is *anti-correlated* with
     the risk — cancellation scales with `e` and the threshold does not, so it
     condemned `apfsds_vs_era_oblique` (e_max 1.07e6, a 9-eps violation on a clean
     700-frame bake) while clearing `apfsds_vs_nera` (e_max 1.00e7, 9.4× larger).
     A negative `e` is a **TRACER, not a cause** — it is universal, born at roundoff
     in every deck, and clamping it did **not** fix the ERA divergence.

12. **Milestone 14 — the CFL margin multiplied a volume RATIO (PHYSICS §3.11).**
   The follow-up M13 deferred. `EOS_CFL_J_MARGIN` → **`EOS_CFL_P_MARGIN = 4.0`**;
   new host helper `_impact_pressure`; `SolverParams.cfl_p_margin` for per-deck
   override. Read §3.11 before touching the substep bound.
   - **`Jd = 0.35 * J_eq` scaled a volume RATIO.** `J` is in (0,1] with the EOS
     diverging at J→0, so that is not "35 % of margin" — it demanded ~3× more
     compression than equilibrium, and the design state landed **past every
     material's MG pole, on 30 of 30 decks, in 54 of 68 (deck, material) pairs**. The
     bound was read off the **pole guard's extrapolated backstop** — the branch
     `J_FLOOR`'s own comment calls "a degeneracy backstop, NOT a physical limit".
     `rha` equilibrates at **J_eq=0.902** (a 10 % compression) and the bound designed
     for **0.316**, reading **137 281 mm/ms** for a material whose real shocked speed
     is ~6 000. That ~20× IS the 5–22 %-of-budget figure, from the other end.
   - **The SCALE was wrong too, and velocity-dependently so — no constant fixes
     that.** `½ρv²` is *steady stagnation*; the substep must survive the **contact
     shock**, which is an impedance match of the two Hugoniots. `p_impact/p_stag` runs
     **3.58× at 1500 m/s → 1.25× at 7000 m/s**. **That spread is the fingerprint on
     the old constant's history** (0.8 → 0.55 → 0.35): one number re-cut to patch an
     error that varies 3× across the repo's own velocity range. `_impact_pressure`
     costs **zero** new material constants — it reuses M13's `u_s = c₀ + s·u_p` fit.
   - **THE PRIZE IS CALIBRATION, NOT THE ~4–5× OF SUBSTEPS** (root §1: bake cost is
     irrelevant; and I/O partly masks it anyway — nera's `frames.bin` is 2.76 GB). The
     old bound **never failed**: it erred in the SAFE direction, which is exactly why
     it survived four milestones unexamined. **A bound that over-predicts 20× is not
     conservative, it is UNCALIBRATED** — and an uncalibrated bound is free to
     *under*-predict on the next deck. That is not hypothetical: margin 0.8 did it on
     `heat_vs_composite`. Same shape as [[instruments-that-cannot-see-the-failure]].
   - **`apfsds_vs_nera` is priced in its OWN deck (`cfl_p_margin: 20.0`), because what
     it covers is NOT a shock.** Its binding particles are 2–4 of 36 966 filler
     particles in a **kinematic vise** (§3.6.1) whose J is set by geometry.
     `nera_filler` sits at its pole (J=1−1/s=0.5) where the EOS asymptotes, so
     pressure is a near-flat lever: **12× the shock moves the design J only
     0.597 → 0.511**, and covering it globally needs **P≈50** — a global stability
     constant sized against a 2-particle extremum, the exact anti-pattern the old
     comment argued against and then committed.
   - **The override is licensed by the measurement being STABLE:** `c_eff` moves only
     **−3.2 %** (113 228 → 109 571) across a **6.8× dt change**, because a geometric
     trap does not dissolve under refinement (§3.6.2 said so). Contrast the jet's
     shock-ring ratio, which drifts with dt and must never be sized against. Shipped:
     **248 substeps, 76 % of budget** (was 1047 at 18 %).
   - **A BREACH IS NOT A DIVERGENCE — read the audit ratio correctly.** It is a
     fraction of the **CFL=0.3 SAFETY FACTOR**, not of the stability limit. Global P=3
     breaches nera 1.22× = Courant **0.37** against a limit near 1, and it bakes
     **finite** with cohesion moving **−0.06 %**. The override buys a warning-free bake
     at a real margin; it does **not** prevent an instability.
   - **P=3 SHIPPED FIRST AND WAS WRONG — P=4 ships (4395de2).** P=3 audited 29 of 30
     clean and `heat_vs_composite_uniform` at **101 %**: the jet compresses past its own
     design J, and overshoot is exactly what this constant is the allowance for. A
     `CFL_AUDIT_TOLERANCE=0.98` shipped for one commit to hold it open, then was
     **deleted** — a tolerance buys **no safety** (dt, physics and bake unchanged; only
     the warning goes). Shipped tally: **12–76 %, ZERO breaches, 30/30**.
   - **⚠️ P=4 IS NEAR A CEILING (~4.05) — do not raise it casually.** `era_filler`
     designs to J=**0.5504** vs its guard switch **0.5500**; raising P 3→4 ate **97 %**
     of that clearance and the crossing is between P=4.05 and P=4.10. Past it the four
     ERA decks size from the guard's extrapolated backstop = **the M14 defect itself**
     (verified red at P=5). More headroom = a **per-deck `cfl_p_margin`**, never a
     bigger global P. The P-sweep table offering P=5/P=6 was computed for `copper_jet`
     on ONE deck and could not see this: **a one-deck sweep cannot license a global
     constant.**
   - **Pin `c_eff ≤ c_max`, NEVER the diagnosis.** "Each material's design J bounds its
     own live J" is **circular** — live J is read off a bake whose dt that design J set
     — and `worst live J` is an extremum besides. That test first went red against a
     **stale P=3 live J**, i.e. it compared a fresh prediction to another
     configuration's data. Don't restate it as prose either.
   - **What moved:** all 30 rebaked twice (P=3, then P=4). Nera is **untouched** — its
     dt comes from its own override, so the global constant cannot reach it. `apfsds_vs_rha`
     reads **spall 0.2085, rod tip 231.13 mm** at 135 557 particles (M13: 0.251 /
     228.27), so that figure has now read 16 → 18.2 → 25.1 → **20.85 %** across four
     configs. Numbers move, conclusions hold — sixth demonstration. **The interim P=3
     deltas ("spall +10 %, tip +2.2 %") are RETRACTED, not carried forward:** P=4
     overwrote those caches, so they are not re-derivable — don't quote them.
   - **The stale comment that hid it:** "0.55 let `apfsds_vs_nera` breach by 2.41×" was
     measured at nera's **pre-M13** worst live J of 0.2421. MG relieved that crush, so
     the breach stopped constraining the constant at M13 and nobody noticed. **M13 made
     the margin over-conservative BY SUCCEEDING.**
   - `tests/test_cfl_sizing.py` pins the design state on the physical branch, derived
     from `materials.py` — **not** by re-running the sizing arithmetic, which would be
     satisfied by copying the bug. **Verified to FAIL on the old formula first.** One
     of its asserts was itself wrong and got fixed by the code: a per-pair
     `p_impact > p_stag` claim fails on rha/copper at 7 km/s, because a lower-impedance
     target genuinely cannot sustain the striker's stagnation pressure. Only the
     **deck-wide max** (which includes the symmetric self-impact) carries the claim.

### The free-slip HIGH walls never fired — a BC fix, not a milestone (done)

Found while chasing what looked like an M13 EOS bug and was not. `_grid_op` tested
`i > nx - bound`, but `nx` is the **allocated** width, carrying 3 pad nodes past the
domain for the stencil — so the high band sat in the pad, where `_g2p`'s position
clamp guarantees nothing goes. **8 of 8 high walls unreachable across four deck
shapes. Since milestone 1.** The low walls worked the whole time (indices count from
0), which is why nobody noticed: every deck ran a working mirror on its low edges and
no wall at all on its high ones.

The position clamp became the BC instead, and it is a **vice**: infinitely rigid,
arresting *displacement* while leaving *velocity* untouched. 2342 particles sat
welded to `y=119.61` at 1699 m/s for 130 frames. M13 only made it visible, by giving
the crush an energy equation to blow up (`e` on the pinned set: **24 → 7.1e5 J/kg in
exactly the frames it reached the clamp**). Fixing it took the J floor **269 509 → 0**
and the resolution guard **217 829 → 0** — neither was ever an EOS problem.

- **Never assert a boundary quantity against an ARRAY SHAPE. The pad is not the
  domain.** `tests/test_boundary_walls.py` derives reachability from the position
  clamp, not from `_grid_op`, so it cannot be satisfied by copying the kernel's own
  mistake — and it was verified to FAIL on the old band before being trusted.
- **The cheap check, reusable:** the decks are symmetric about y=60 *by
  construction*, so the material must be. Dead walls: `rha` `pos_y` 0.88…**119.61**.
  Live: 0.88…119.12 — mirrors to **0.88 vs 0.88, exact**.
- **Armor touching the walls is NOT the defect** — `_seed` spans the full height
  *precisely so* the mirror makes a plate that continues beyond the frame.
- **It invalidated every boundary-adjacent figure** (see the stale-number note
  below, and PHYSICS §1.1.1). All re-measured; all conclusions held.

Don't rewrite from scratch. The full solver arc (milestones 1–13) is done.

**Stale-number correction (measured 2026-07-16, updated 2026-07-17):** the "~16 %
RHA spall" quoted in the milestone 3/4 notes above and the 15.6 % in milestone 6
were measured at **96 637 particles**, before the plate-not-block geometry change.
`apfsds_vs_rha` then seeded **135 557** particles and spalled **18.2 %**.

**Milestone 13 moved it again — to 25.1 %** (rod tip 232.33 → **228.27 mm**), at the
*same* 135 557 particles, so this one is physics rather than geometry: Mie-Grüneisen
(PHYSICS §3.10) resists the shock harder and does more plastic work, artificial
viscosity is now ON by default, and the §1.1.1 boundary fix gave the plate a working
mirror at its top edge for the first time. **+38 % is the largest single move any
headline figure in this repo has taken.** The ordering and every conclusion those
milestones drew are still unaffected; only the absolute moved — again.

**The pattern is now the point, and it is worth more than any of the values:** this
number has read 16 % → 18.2 % → 25.1 % across three changes, and every time the
conclusions held. Do not compare a figure across geometries, EOS laws, or boundary
conditions; treat every absolute here as a reading of one configuration. See the
memory note on rebakes invalidating documented results.

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
