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

All seven solver milestones are done, and the Godot viewer plays real bakes back in
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
0.0706 → **0.3971**, the RHA plate 0.1747 → **0.4918**, and ceramic comminution
stays put at **0.9911 vs 0.9912** — the a-priori prediction that no volumetric fix
could move it, confirmed to four decimals. Independent check: `K₀ = λ+µ = 136.4 GPa`
derived from the elastic moduli agrees to **2 %** with `ρ₀c₀² = 139.1 GPa` from
public shock data. Still honest about it: Murnaghan is a *cold* curve, so it under-
reads pressure by ~0.68× at a 7 km/s tip, and MPM has no artificial viscosity so
the shock front rings. See PHYSICS §3.5.

**Next:** the open items are a **velocity sweep** against the hydrodynamic
asymptote `√(ρ_p/ρ_t)`, a **standoff study** (the clean energy-neutral depth
experiment for the jet — needs no code, `standoff` is already deck data),
**Mie-Grüneisen** (the thermal term Murnaghan lacks), and **domain/BC** work so
oblique-deck debris never reaches a wall. See the per-directory `CLAUDE.md` files
for the build order.

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
