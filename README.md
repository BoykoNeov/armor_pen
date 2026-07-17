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

All ten solver milestones are done, and the Godot viewer plays real bakes back in
motion. Every deck bakes on the RTX 5090 (NVIDIA Warp, sm_120) and passes
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
7. **Shaped-charge (HEAT) jet** — a real jet, not a rod stand-in: the projectile is
   seeded **velocity-graded** (7000 m/s tip → 2000 m/s tail), so it stretches in
   flight and erodes fluid-like.
8. **Equation of state** — a Murnaghan volumetric law, tangent-matched at `J=1`, so
   the hypervelocity stagnation point stops under-resisting.
9. **Velocity sweep** — the first claim that is a *trend*: two arms converging on
   two different a-priori hydrodynamic asymptotes.
10. **Standoff** — the jet's energy-neutral depth experiment, and the first study
    whose headline is a **grid-convergence limit** rather than a result.

The rod is also **pointed** (conical nose, `nose_shape` in the deck), not the
flat-faced cylinder it used to be — a real APFSDS is sharp (`docs/PHYSICS.md` §1.2).

Every deck runs long enough to resolve **perforate-or-stop**, in a domain sized so
the armor spans the full field height: the target is a plate that continues past
the frame (armor on a vehicle), not a block floating in vacuum.

**Headline result (verified, `docs/PHYSICS.md` §3.1–3.2).** Measured against an
equal-areal-mass *inert* twin, the reactive layer **protects the backing plate, and
slows the rod without ever cutting it**. Main-plate spall is ≈16% lower at 55°
obliquity and ≈8% lower at 0°, tracking a single mechanism: the detonation shoves
the main plate forward (+7.7 mm vs the twin at 55°, +1.6 mm at 0°), and a plate
moving *with* the rod defeats less penetrator.

The rod is **not cut or deflected** at either angle — that specific expectation
("the flyer sweep erodes the rod") is what failed, and it is reported as it came
out rather than tuned toward. But "not cut" is **not** "not affected": at 55° the
rod's residual velocity is **8.5% lower** (679 vs 741 m/s), a real degradation
(~75× the noise floor below). A tough tungsten rod is slowed by thin
few-hundred-m/s flyers; it is not severed by them. At 0° even that is absent
(+2.2%, i.e. nothing), which is the honest null.

*Read those numbers with two different error bars attached.* Run-to-run scatter is
**≤0.11%** (measured by re-baking identical decks), so the protection is ~150× the
numerical noise floor — it is signal. But **model sensitivity is large**: the same
A/B has read ≈40% (old floating-block geometry), ≈21% (plate geometry, blunt rod)
and ≈16% (plate geometry, pointed rod). The **sign is robust across every condition
tried; the magnitude is not portable** — quote it as "roughly 10–20%, sign-stable."
At 0° the margin's sign has actually flipped across a geometry change, which is why
55° earns confidence and 0° does not, though both clear the numerical floor.
Plausibility, not prediction (see the scope note above).

**Second result (verified, `docs/PHYSICS.md` §3.4).** The shaped-charge jet needed
**no new kernel and no SPH** — what makes a jet a jet is one initial condition, a
tip-to-tail **velocity gradient**, and the rest emerges. Because each element flies
at its own constant speed, the jet **stretches**, and that is kinematic, so it is
predictable in advance and therefore falsifiable: free-flight markers separate at
**2.085 mm/µs measured against 2.083 predicted (+0.1%)**. The control is the
convincing half — `heat_vs_composite_uniform` is the same deck with the gradient
*omitted and nothing else changed*, and its body **shortens 40 mm** by tip erosion
where the jet **stretches 40 mm**: symmetric, opposite in sign. The rate reproduces
on tungsten too (+0.0%), a 7.5× different yield — which is what "kinematic" has to
mean, since the prediction uses only the seeded velocities. Fluid-like erosion came
free: at a 7 km/s stagnation point copper's yield is ~1000× below the pressure, so
the ordinary von Mises return mapping caps deviatoric stress near zero by itself.

Two things it does **not** claim. **Particulation never fires** — a real jet
eventually tears into a fragment train, and this one stays continuous. The
arithmetic says it should (stretch reaches only ~half copper's ductile reserve) and
real jets particulate at ~100 µs against this deck's 25 µs, so staying continuous is
the *correct* answer — but the breakup threshold was not lowered to force a prettier
result. And the jet's **penetration is not compared** to the rod stand-in it
replaced: that comparison is energy-confounded twice over (a graded jet carries less
KE than a uniform one, and copper is half tungsten's density), so the claim is
scoped to kinematics, which is immune to both.

**Milestone 8 — an equation of state.** The honest gap a hypervelocity jet exposed
is closed: the volumetric response is a **Murnaghan EOS** (`p = (K₀/K′)(J^−K′−1)`,
`K₀ = λ+µ`, `K′ = 4`) instead of a law with no EOS at all, which used to let a
220 GPa stagnation point crush copper to `J≈0.15` where reality gives `≈0.61`.
Costs **zero** new material constants and is tangent-matched at `J=1`, so it is a
large-strain-only change — KE decks barely move. Measured: the jet tip goes `J`
0.0706 → **~0.46**, the RHA plate 0.1747 → **0.50**, and ceramic comminution stays
put at **0.9910 vs 0.9912** — the a-priori prediction that no volumetric fix could
move it, confirmed. (Only the ceramic figure deserves four decimals; the tip `J` is
a noisy extremum. Milestone 11 later showed that number is the **first impact
transient**, not the jet-tip stagnation, and *not* the undamped shock ring this
README used to blame it on.) Independent check: `K₀ = λ+µ = 136.4 GPa`
derived from the elastic moduli agrees to **2 %** with `ρ₀c₀² = 139.1 GPa` from
public shock data. Still honest about it: Murnaghan is a *cold* curve, so it under-
reads pressure by ~0.68× at a 7 km/s tip. See PHYSICS §3.5.

**Milestone 9 — velocity sweep vs the hydrodynamic asymptote.** The first
experiment that varies impact velocity, and the first whose claim is a *trend*.
Ten decks: `{tungsten, copper} × {1500…7000} m/s` into an identical semi-infinite
RHA half-space. Ideal-hydro (Tate) says the penetration velocity approaches a
density-only ratio — `u/v → 1/(1+√(ρ_t/ρ_p))`, i.e. **0.600** for tungsten and
**0.517** for copper. Two arms, two different numbers fixed a priori: a single arm
hitting a single number could be luck, two arms hitting two different ones could
not. Measured: both rise monotonically toward their own asymptote and neither
crosses it, and the **ratio of the arms converges** on the density prediction as
strength becomes negligible — `+4.8 % → +3.3 % → +1.9 % → +0.04 %` across
2500→7000 m/s:

> at 7 km/s, measured `u/v(W)/u/v(Cu)` = **1.1614** vs **1.1609** predicted from
> density — **0.04 %**. The *convergence* is the claim; 0.04 % is the 7 km/s value,
> not a flat property.

Controlled, not argued: the substep count rises with velocity under the CFL bound,
so the fast arm got less numerical dissipation too. Rebaking all ten at a fixed
`dt` moves each point ~1–2 % and leaves the shape and the ratio intact. And the
sweep genuinely needed the EOS — on the pre-EOS law at matched `dt`, copper@7000
gives `u/v` = **1.032× its asymptote**, i.e. *past* a ceiling that strength cannot
push through. See PHYSICS §3.7.

**Milestone 10 — standoff, and the first result whose headline is its own limit.**
Milestone 7 refused to compare the jet's *depth* to anything, because every
comparison was energy-confounded. Standoff is the clean version: the same jet, the
same energy, only the flight distance before impact differs — and it needed **no new
code**, since `standoff` was already deck data. A velocity-graded jet extrapolates
back to a **virtual origin** `Z₀ = L·v_tip/(v_tip−v_tail) = 168 mm` behind its own
tip, so the seeded jet already carries 168 mm of built-in standoff and the deck adds
to it: `Z = 168 + S`. Depth at a *matched consumed element* is then proportional to
`Z` — with **slope and intercept both fixed a priori** by the seeded gradient, and
provably so for **any** `u(v)`, strength included.

> **The shipped decks under-read it, and that is the finding.** Measured S90/S0 depth
> ratio **1.229** against **1.536** predicted. The jet is 3 mm across = **8 cells**,
> and it *thins as it stretches* to ~3. Refining the grid walks the ratio up
> **1.229 → 1.383 → 1.429**, and a **6 mm jet at the shipped resolution** — 16 cells
> across by fattening instead of refining, an independent route the derivation allows
> because it is diameter-independent — reads **1.501**, within scatter of the
> prediction. **Cells across the jet is the controlling parameter**; the shortfall is
> numerical.

Reported as a trend, not a value: the convergence order is ill-conditioned, so no
extrapolated number is quoted, and "converges toward" is not "converged". Two traps
worth knowing: depth at a fixed *lab time* moves the **wrong way** (105 → 80 mm — a
longer standoff impacts later and penetrates for less of the window), and the
under-resolved curve *saturates*, which looks exactly like the textbook standoff
optimum this jet cannot produce. Six `standoff_conv_*` decks are committed so the
convergence is reproducible rather than asserted. See PHYSICS §3.8.

**Milestone 11 — artificial viscosity, and the defect that wasn't.** Milestone 8
left a named defect: *"nothing damps the shock ring, now the dominant tip
defect."* This milestone built the standard fix (von Neumann–Richtmyer), measured
it, and **retired the diagnosis on evidence** — so the feature ships **default
off**. Three findings, and the measurement is the deliverable. (1) Every
frame-cadence metric **aliases** the ring: it has a ~159-substep period while
frames are 400–1600 substeps apart, which is why the obvious metric went
*non-monotone*. Seeing it at all needs a **per-substep trace of one particle**
through the shock — a min-over-a-set traces the envelope and hides it. (2) The
ring is real but **~0.9 % peak-to-peak**, which cannot explain the ~30 % gap it
was blamed for. (3) That gap was a **category error**: the famous "tip `J`≈0.43"
is the *first impact transient*, while the *steady* stagnation state — the thing
`J_eq=0.6056` actually predicts — sits at **~0.63**. AV costs +57 % substeps to
damp ~1 %, so it stays off; it is kept because it is the prerequisite for
Mie-Grüneisen (AV work currently dissipates to *nothing*). It is inert on KE decks
(≤0.20 %, at the scatter floor). See PHYSICS §3.9.

**Since shipped:** **Mie-Grüneisen** with an energy equation (§3.10 — and AV is
default-**on** now, because carrying shock heating into `e` is its real job, not
damping the ring above), a **dissipation path for `nera_filler`** (§3.6.1 — which
falsified the "79 % volume loss" this repo had believed for two milestones), a
**boundary fix** (§1.1.1 — the free-slip *high* walls had never fired, since
milestone 1), and a **recalibrated CFL bound** (§3.11 — the margin had been
multiplying a volume *ratio*, which put the design state past every material's EOS
pole on all 30 decks).

**Next:** **domain/BC** work so oblique-deck debris never reaches a wall (the
standoff family's wall reflections are also not common-mode at matched fraction),
a **volumetric/compaction criterion** for confined fillers, and the **jet's grid
resolution** — it is only 8 cells across, which limits every jet *depth* claim here
(§3.8). See the per-directory `CLAUDE.md` files for the build order.

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
