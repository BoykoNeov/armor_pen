# Physics Notes

Public, textbook-level physics backbone for the simulation. Everything here is
**representative and illustrative** — order-of-magnitude, not spec-sheet (see
CLAUDE.md §10). The bar is *plausible*, not *validated*.

---

## 1. Method: MLS-MPM

**Moving Least Squares Material Point Method.** Particles carry mass, momentum,
and deformation state; a background Eulerian grid handles stress divergence and
self-contact each step. Chosen over SPH because:

- The grid gives **automatic self-contact** — the penetrator and the armor
  collide simply by sharing the grid; no explicit contact model.
- It **avoids SPH's tensile instability**, which would otherwise wreck exactly
  the spall/fracture behavior we care about.

We **grow the canonical 88-line MLS-MPM** reference (Hu et al.) rather than
rewriting from scratch: add plasticity, then damage, then the multi-material
armor stack, validating visually at each step via `tools/inspect_cache.py`.

SPH may return later, specifically for HEAT-jet fluid-like erosion.

### Transfer cycle (per substep)

1. **P2G** — scatter particle mass/momentum (and APIC/MLS affine term) to grid.
2. **Grid update** — apply forces (stress divergence), gravity if any, boundary
   conditions; convert momentum to velocity.
3. **G2P** — gather updated velocity (and velocity gradient) back to particles.
4. **Particle update** — advect positions; update deformation gradient `F`;
   apply the constitutive model (§3).

---

## 2. Unit system — mm · ms · g

**Work in one consistent, non-dimensionalized system. Never mix raw SI into the
kernels** (raw SI makes stiffness huge and `dt` tiny, inviting float error).

The chosen system is **millimetre – millisecond – gram**, well established for
impact/ballistics. Derived units fall out cleanly:

| Quantity | Unit in this system | Note |
|---|---|---|
| length | mm | |
| time | ms | |
| mass | g | |
| velocity | mm/ms = **m/s** | 1500 ≈ a 1.5 km/s impact |
| density | g/mm³ | steel ρ ≈ **7.85e-3** |
| stress / pressure | g/(mm·ms²) = **MPa** | steel E ≈ **2e5** MPa |
| force | g·mm/ms² = **N** | |

Reference values (steel): ρ ≈ 7.85e-3, E ≈ 2e5 MPa, a 1.5 km/s impact ≈ 1500.

All physical constants live **once**, in these units, in
`solver/src/ballistics_solver/materials.py`. The manifest's `units` field
records the choice (`"mm-ms-g"`).

---

## 3. Material model

Elasticity + rate-independent plasticity + a damage threshold:

- **Elasticity:** fixed-corotated or Neo-Hookean hyperelasticity on the
  deformation gradient `F`.
- **Plasticity (metals):** **von Mises** plastic return-mapping — project the
  trial stress back onto the yield surface each step, storing plastic
  deformation. This is what lets the metal flow and mushroom rather than shatter.
  *Implemented (milestone 2):* a perfectly-plastic (no hardening) **radial
  return in log-strain (Hencky) space** — SVD `F = U Σ Vᵀ` per particle after
  G2P, radially return the deviatoric log-strain onto `‖dev τ‖ = √(2/3)·σ_Y`,
  reconstruct `F`. Plastic flow is isochoric (volumetric log-strain untouched).
  Two plausibility notes (root §1): the `√(2/3)` and the deviatoric split use a
  2D two-principal-strain convention, not exact 3D J2; and because the reported
  stress is fixed-corotated Cauchy von Mises (the momentum-driving stress), it
  reads *approximately* capped near yield — with a small over-read tail at
  extreme volumetric compression at the shock front, since fixed-corotated has
  no equation of state (`λ(J−1)J → 0` as `J → 0`). This is a pre-existing
  property of the elastic model, best tamed viewer-side by a percentile clamp.
- **Damage:** a scalar in `[0, 1]` (latched, irreversible). When a particle
  fails it **detaches** into a free fragment — `_p2g` drops its stress term so it
  keeps mass + momentum but can no longer hold tension/shear. This is the spall
  spray. The particle is *flagged*, never created or destroyed (fixed particle
  count, see CACHE_FORMAT §5). Two failure modes, selected per material:
  - **Ductile (metals)** *— implemented, milestone 3:* accumulated equivalent
    plastic strain `alpha` crossing the material's `damage_threshold`. Metals
    flow and mushroom, then spall along the plastic channel walls / crater lip.
  - **Brittle (ceramics)** *— implemented, milestone 4:* a **stress** trigger,
    independent of plastic strain — brittle solids have no plastic reserve, so
    they shatter the instant the stress state reaches their strength surface.
    A brittle particle latches damage when the fixed-corotated Cauchy **von Mises
    stress ≥ `yield_strength`** (compressive comminution directly under the
    penetrator) **or** the **max tensile principal stress ≥ 0.1·`yield_strength`**
    (mode-I tensile cracking at free surfaces — back-face spall, interface
    debonding at impedance mismatches, the fracture conoid ahead of the rod).
    Note the von Mises branch is evaluated *after* the radial return has already
    capped deviatoric stress at yield, so in practice it fires for **any brittle
    particle that yielded at all** — the model is "brittle = shatters where a
    metal would instead have flowed," not a separate higher fracture stress above
    yield. The tensile branch is the independent one, catching low-stress cracking
    the deviatoric criterion misses.
    `yield_strength` doubles as the fracture strength (no separate field); the
    0.1 tensile ratio is illustrative (ceramics crack in tension at a small
    fraction of their compressive strength). This is what makes a ceramic core
    *shatter* into rubble and read visually distinct from a denting steel plate,
    instead of behaving as a near-indestructible ductile wall at KE velocities.

### Material archetypes (illustrative)

| Material | Character |
|---|---|
| Tungsten / DU rod | Very dense, stiff, high yield — the KE penetrator. |
| RHA (steel) | Baseline ductile armor; mushrooms and spalls. |
| Ceramic / composite | Higher stiffness, **brittle** (`brittle: true`) — fails on the stress trigger above, shattering with ~zero plastic flow. |
| ERA filler | An impulse layer that degrades the penetrator on contact. *(reactive impulse — implemented, milestone 5; see §3.1.)* |
| NERA filler | A soft interlayer that never detonates but stays cohesive, so the sandwich plates bulge apart on the shock alone and the bulge is *held open*. *(the unignited branch of the same reactive path — verified, §3.3.)* |

### 3.1 Reactive layer — ERA/NERA (milestone 5)

A **reactive filler** (`era_filler`, `reactive: true`) models the interlayer of a
reactive-armor sandwich `[plate | filler | plate]`. It is an **impulse layer**:
when the impact shock reaches it, it ignites and releases an isotropic
overpressure that flings the sandwiching plates apart. Modelled as a **pressure
source term carried through the ordinary MLS-MPM grid** — the plate motion is
*emergent* (the source drives the filler, the filler drives the plates through
grid contact), **not** a scripted kick to the rod. Reactive particles run a
self-contained state machine, deliberately excluded from the plastic /
ductile-spall path (see mpm.py's reactive note — the soft filler would otherwise
ductile-spall in the same shocked substep it should ignite, silently no-op-ing
the detonation):

- **unignited** → soft fixed-corotated elastic (the plates bulge from the raw
  shock even with no detonation). A persistent **NERA** bulge is this branch held
  open — a filler that *never ignites*, i.e. `ignition_compression=0` (stays
  soft-elastic), **not** merely `detonation_pressure=0`: a filler with
  `ignition_compression>0` still ignites on the 2% impact shock and, with zero
  pressure, burns to limp debris — that is bulge-*then-collapse*, not a sustained
  NERA bulge. *(Untested — no NERA deck is baked yet; this is the intended knob,
  not a verified result.)*
- **burning** → isotropic detonation overpressure for `burn_time` ms; ignition
  triggers when shock compression drops `det(F)` below `ignition_compression`.
- **spent** → cohesion-free debris (mass + momentum, no stress). `damage` is
  repurposed as the reactive "ignited/spent" latch (and the viewer flag).

Two plausibility guards keep this stable (root §1/§11): burning **and spent**
filler have `F` pinned to identity each substep (a detonating gas / debris has no
elastic reference configuration, and the return-mapping that would otherwise cap
`F` skips reactive particles); and reactive-particle speed is clamped at a
physical detonation-product scale (`REACTIVE_VMAX`), because the `F`-independent
source would otherwise accelerate unconfined light debris to a CFL-breaking
~14 km/s once the plates separate. Both touch reactive particles only.

**Verified result — and an honest limitation.** The mechanism fires cleanly: the
filler detonates and the sandwich plates fly apart at a few hundred m/s. But at
**0° (normal incidence) the reactive layer produces negligible penetrator
degradation** — measured against an *equal-areal-mass inert twin*
(`era_filler_inert`: identical density/stiffness/thickness, reactivity off), the
rod's residual penetration into the protected main plate is within noise
(15.7 mm reactive vs 15.5 mm inert; rod residual velocity, damage fraction, and
main-plate spall all within scatter — if anything the reactive rod is marginally
*deeper* and *less* damaged, a coherent but noise-floor effect: the detonation
clears filler off-axis, so the reactive rod pushes through slightly less on-path
material than the inert rod that keeps its filler in the channel — no protective
benefit either way). This is **correct physics, not a bug**: at 0° the detonation flings the
plates *laterally, symmetric about the rod axis*, so the debris sweeps sideways
and never crosses the rod path to cut it. Real reactive armor gets its
effectiveness from **obliquity**. The A/B decks (`apfsds_vs_era` /
`apfsds_vs_era_inert`) are byte-identical in geometry, areal mass, and timing, so
the near-zero delta cleanly isolates "the reactive contribution at 0° is
negligible." Obliquity is milestone 6, below.

### 3.2 Oblique reactive armor (milestone 6)

At obliquity the rod is tilted relative to the plate normal, so the
detonation-flung plates gain a velocity component **perpendicular to the rod**
(zero at 0°, `∝ sin θ`) and the interaction is no longer symmetric about the rod
axis. Implementation is minimal and protects the validated M1–M5 physics: the
projectile **rectangle** is rotated by `angle_deg` about its tip so the rod
strikes *nose-first* along its velocity (mpm.py `_seed`), while the armor slabs
stay vertical/axis-aligned. Only the *relative* rod-axis/plate-normal angle is
physical, so rotating the rod against fixed slabs is frame-equivalent to tilting
the slabs against a horizontal rod — and `angle_deg=0` is exact identity, so
every normal-incidence deck seeds bit-for-bit as before. Decks:
`apfsds_vs_era_oblique` (+ its `_inert` twin), 55° from the normal, in a taller
(180 mm) domain so the tilted 60 mm rod fits its body above the tip **and** its
descent through the target (the rod drops `~tan 55° ≈ 1.4` mm in y per mm of x).

**Verified result — protection, but not rod-cutting.** Measured against the
equal-areal-mass inert twin, at 55° the reactive layer **measurably protects the
backing plate**, where 0° was a null:

- **Main-plate spall ≈ 40 % lower** for the reactive deck (0.071 vs 0.117 at the
  final frame), and the gap **grows monotonically** over the event
  (0.017 → 0.035 → 0.046 across frames 70/90/119). At 0° the same A/B was a null
  of the *opposite* sign (reactive marginally *worse*, 0.152 vs 0.133), so
  obliquity flips the layer from no-benefit to clear protection.
- **The rod itself is essentially unaffected** — the coherent rod tip reaches the
  same depth frame-for-frame in both decks, and net rod damage / path angle are
  equal (0.49 vs 0.50; 57° vs 56°). The lateral-sweep mechanism is present but a
  tough tungsten rod is not cut or deflected by thin, few-hundred-m/s flyers.
  (The "erode/deflect the rod" outcome was an *a priori* expectation; the sim
  says protection comes through the backing plate instead — reported honestly.)
- **Mechanism.** The detonation **accelerates the main plate forward** earlier and
  faster (front displaced ~8.4 mm / ~180 m/s reactive vs ~4.2 mm / ~143 m/s
  inert) and disperses the coherent flyer-plate + filler mass that, in the inert
  deck, is driven into the plate as a follow-through/tamping slug. The plate
  moving *with* the rod reduces the effective (rod-relative) penetration: with the
  tip near x ≈ 162 in both and the plate front at ~144 (reactive) vs ~140 (inert),
  rod-relative penetration is ~17 vs ~21 mm — **~18 % shallower**. This is the
  textbook "moving/standoff plate defeats less penetrator." *(Order-of-magnitude:
  a centroid-based estimate, and the front face is also cratering.)*

Honesty caveats (root §1/§10): a steeper angle was **not** chased (it only moves
`sin θ` 0.82→0.91 and worsens domain-fit/bottom-wall descent), and
`detonation_pressure` was **not** cranked to force rod degradation (that would be
confirmation-bias tuning toward defeating a system — off-limits per §10). Rod and
debris reach the bottom domain wall (`y ≈ 1`) in **both** decks — a *shared*
artifact, so the A/B **delta** and its time-evolution are the meaningful signal,
not the absolute spall numbers. One bake per condition; the monotonic divergence
across 120 frames plus the sign-flip vs 0° is what makes it clearly not MPM
non-determinism, rather than a repeat bake.

### 3.3 NERA persistent bulge — the unignited branch (verified)

§3.1 describes a **persistent NERA bulge** as the reactive path's *unignited*
branch held open: a filler with `ignition_compression=0` never ignites
(`_update_reactive` gates ignition on `ic > 0`), so it stays soft-elastic and
cohesive and the sandwich plates bulge apart on the impact shock alone. This is
**not** merely `detonation_pressure=0`, which still ignites on the shock, latches
the particle spent, and collapses it to limp debris. That branch was implemented
at milestone 5 but never baked. It is now verified by `apfsds_vs_nera`
(`nera_filler`), geometry-identical to the two ERA decks — all three seed at
126157 particles, so they are equal-areal-mass arms of one A/B family differing
only in the filler's response path.

**The branch works as specified.** Across all 120 frames the NERA filler's damage
fraction is **0.000** — it never ignites and never spalls — and the sandwich opens
without ever collapsing back: front/back plate separation grows monotonically
18.0 → 23.5 mm and is still opening at the end of the window.

The decisive evidence is the **cohesion**, not the shape of the tail: the filler
expands far less than either twin — thickness 10.6 → ~36 mm, levelling off over
the last frames (36.1 at 100 → 35.7 at 119) — where the inert twin's filler
shreds (damage 0.615, thickness still climbing at 50.7 mm) and the reactive
twin's latches 1.000 by frame 20 and is flung to 69.7 mm with the plates driven
58.5 mm apart. Confirmed visually (viewer `--shots` on the deck): at frame 119
the interlayer is still **large coherent bent slabs**, split around the rod
channel but intact, with the white spall spray coming from the *steel plates*,
not the filler. Cohesive, unignited, stable — no NaN, no collapse.

**Model-mechanics note — NOT an armor-performance claim.** In the same bakes the
rod ends up shallower, slower, and more damaged against the NERA filler than
against either ERA twin (tip 160.0 vs 164.1/164.3 mm; median intact-rod speed
1316 vs 1445/1435 m/s; rod damage 0.363 vs 0.275/0.244). **Do not read this as
"non-explosive beats explosive."** The comparison is confounded by construction,
twice over: `reactive=True` makes `mpm.py` skip *both* `_return_mapping`
(plasticity) *and* `_update_damage` (ductile spall) for that particle, so
`nera_filler` can neither yield nor break — its `yield_strength` and
`damage_threshold` are dead fields. So this is not "cohesive vs shredding at
equal toughness"; it is "an unbreakable filler vs one that spalls at threshold
0.02," and "the unbreakable one resists the rod better" is close to tautological.
What it does illustrate is a real property of the damage model: a spalled
particle keeps its mass and momentum but drops its deviatoric stress term in
`_p2g`, so it stops *resisting* — an equal mass of debris loads the rod far less
than an equal mass of cohesive material. A genuine single-variable cohesion test
would be a **non-reactive** filler with a high `damage_threshold` against the
0.02 one; that isolates cohesion without also disabling plasticity, and is not
done here.

Honesty caveats (root §1/§10): one bake per condition, and MPM grid `atomic_add`
ordering is non-deterministic — but the deltas are large, monotonic, and
sign-stable across many frames, so they are not noise. The main-plate spall
fractions quoted here are measured over a frame-0 x-band and are **not**
comparable to the differently-measured 0° absolutes in §3.1 (the *ordering* there
— reactive ≈ inert, reactive marginally worse — does reproduce). The
never-yields property also means the filler stores elastic energy without
dissipating it, i.e. stiffer-than-real; that is a modelling limitation, not a
bug, and it is another reason the rod deltas above are model-specific.

---

## 4. Timestep & why we bake offline

The cost driver is the **CFL timestep, not particle count**. Steel's sound
speed is ~5 km/s; explicit MPM requires

```
dt  <  C_cfl · Δx / c_sound
```

which in SI lands `dt` on the order of **1e-8 – 1e-7 s**. A penetration event is
a ~microseconds physical window, so it is a **short window of thousands of cheap
substeps**. That is precisely why the solver is offline, on the GPU, and dumps
only every Nth substep as a render frame.

**Stability gotcha:** if the sim blows up, suspect `dt` (vs. sound speed × grid
spacing) *before* touching the material model. A sim that explodes or does
nothing is often a **units** mistake, not a physics bug.

---

## 5. Public references

Textbook / public-domain sources this backbone draws on:

- **MLS-MPM:** Hu, Fang, Ge, et al., *"A Moving Least Squares Material Point
  Method with Displacement Discontinuity and Two-Way Rigid Body Coupling"*
  (SIGGRAPH 2018); and the widely circulated **88-line MLS-MPM** reference
  implementation.
- **MPM foundations:** Sulsky, Chen, Schreyer, *"A particle method for
  history-dependent materials"* (1994); Jiang et al., *"The Material Point
  Method for Simulating Continuum Materials"* (SIGGRAPH 2016 course notes).
- **Hydrodynamic penetration:** the **Tate–Alekseevskii** long-rod penetration
  model — standard, textbook, public. Alekseevskii (1966); Tate (1967).
- **Plasticity:** von Mises yield criterion and radial-return mapping — any
  computational plasticity text (e.g. Simo & Hughes, *Computational
  Inelasticity*).

Nothing classified or export-controlled enters this repo; public physics is the
ceiling here by construction (CLAUDE.md §10).
