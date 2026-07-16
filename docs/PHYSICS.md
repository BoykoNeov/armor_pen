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

### 1.1 Boundaries: the target is a plate, not a block

The domain walls are **free-slip**: `_grid_op` zeroes only the velocity component
heading *into* the wall and leaves the tangential component alone. (They were
long mislabelled "sticky reflecting" in the source — sticky would zero the whole
velocity, reflecting would negate the normal component. They do neither.)

A free-slip wall is a **mirror plane**. That fact is what lets us model armor
honestly: the armor slabs are seeded across the **full domain height**, so a slab
plus its mirror images reads as a plate that *continues beyond the frame* — armor
on a vehicle. Previously slabs stopped 10 % of the domain height short of each
wall, which made every target a finite block floating in vacuum: its free top and
bottom edges flared outward into the void, and crater ejecta escaped around them
into empty space instead of interacting with armor.

The projectile is left axis-aligned in this picture and the **rod** is rotated for
oblique decks rather than the slabs (§3.2). That is not only frame-equivalence:
mirroring a *tilted* slab would fold it into a V, so a tilted slab could never
read as a continuous plate. Vertical slabs mirror onto themselves.

Two consequences worth stating plainly:

- **The mirror implies an image of the projectile one domain-height away.** A deck
  must therefore be tall enough that the event resolves before the rod or its
  spray nears a wall. This is per-deck sizing (domain size is data, CLAUDE.md §9),
  not something the kernel can enforce.
- **A finite domain cannot let far-field ejecta leave.** Late in a bake, spall
  spray does reach the top/bottom walls and slide along them. What matters is that
  the penetration channel is many rod-diameters from the wall, so the event we
  measure is unaffected; the artifact is confined to the far-field debris.

Both transfer kernels index the grid at `floor(Xp − 0.5) + {0,1,2}` with no bounds
check, so a particle within half a cell of a low edge would scatter **out of
bounds**. The old 10 % margin hid this; with slabs now at the wall for the whole
bake, `_seed` insets them two cells and `_g2p` clamps particle positions one cell
inside the domain. The clamp is memory safety, not physics — the slip wall
already removes wall-normal velocity, so it almost never binds.

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
`apfsds_vs_era_oblique` (+ its `_inert` twin), 55° from the normal, in a 220 mm
domain with the impact deliberately **off-centre** (`impact_y: 145`): the rod
drops `~tan 55° ≈ 1.43` mm in y per mm of x, so it needs ~145 mm of descent below
the impact but only its own tilted body-length (~49 mm) of headroom above it.
Centring the impact would demand a domain ~2× taller — and ~2× the particles — to
buy headroom the rod never uses.

**Verified result — protection, but not rod-cutting.** Measured against the
equal-areal-mass inert twin (both decks seed at **287 615** particles), at 55° the
reactive layer **measurably protects the backing plate**:

- **Main-plate spall ≈ 21 % lower** for the reactive deck (0.131 vs 0.166 at the
  final frame), and the gap **grows monotonically** over the event
  (0.023 → 0.032 → 0.035 across the 50 / 75 / 100 % marks).
- **The rod is degraded only modestly.** Residual velocity is ~9 % lower (737 vs
  814 m/s) and the tip ends 1.6 mm shallower (234.8 vs 236.4) — a real but small
  effect, not the cutting the mechanism might suggest. A tough tungsten rod is not
  severed or deflected by thin, few-hundred-m/s flyers. (The "erode/deflect the
  rod" outcome was an *a priori* expectation; the sim says protection arrives
  mostly through the backing plate instead — reported as it came out.)
- **Mechanism.** The detonation **shoves the main plate forward**: its leading edge
  ends at x ≈ 192.9, ~6.9 mm *ahead* of where it started (186), while the inert
  twin's leading edge ends at 177.2 — ~8.8 mm *behind* the start, because that
  plate is simply cratering and throwing lips backward. A plate moving *with* the
  rod reduces effective (rod-relative) penetration — the textbook "moving /
  standoff plate defeats less penetrator."

Honesty caveats (root §1/§10): a steeper angle was **not** chased (it only moves
`sin θ` 0.82→0.91 and worsens domain fit), and `detonation_pressure` was **not**
cranked to force rod degradation (that would be confirmation-bias tuning toward
defeating a system — off-limits per §10). One bake per condition; the monotonic
divergence is what argues this is not MPM non-determinism, rather than a repeat
bake.

**These numbers were re-measured after the geometry change, and the magnitude
moved — take that as the error bar.** The same A/B on the old floating-block
geometry read ≈ 40 % rather than ≈ 21 %, and reported the rod as *unaffected*
where it now shows ~9 %. The **sign and the monotonic growth are robust; the
magnitude is not.** The 0° arm makes the point sharper: it used to read 0.152 vs
0.133 (reactive marginally **worse**) and now reads 0.229 vs 0.251 (reactive
marginally **better**) — a margin whose *sign* flips with geometry is noise, so
**0° remains an honest null**, and its instability is the best available estimate
of how much a ~9 % difference is worth here. The 55° result clears that bar; a 0°
result of the same size would not. Spalled rod fragments still reach the bottom
wall late in the run (the intact rod clears it by ~35 mm), a *shared* artifact of
both decks — another reason to read the A/B **delta**, not the absolutes.

### 3.3 NERA persistent bulge — the unignited branch (verified)

§3.1 describes a **persistent NERA bulge** as the reactive path's *unignited*
branch held open: a filler with `ignition_compression=0` never ignites
(`_update_reactive` gates ignition on `ic > 0`), so it stays soft-elastic and
cohesive and the sandwich plates bulge apart on the impact shock alone. This is
**not** merely `detonation_pressure=0`, which still ignites on the shock, latches
the particle spent, and collapses it to limp debris. That branch was implemented
at milestone 5 but never baked. It is now verified by `apfsds_vs_nera`
(`nera_filler`), geometry-identical to the two ERA decks — all three seed at
**180 449** particles, so they are equal-areal-mass arms of one A/B family
differing only in the filler's response path.

**The branch works as specified.** Across all 550 frames the NERA filler's damage
fraction is **0.000** — it never ignites and never spalls — and the sandwich opens
without ever collapsing back: front/back plate separation grows 18.0 → 21.1 mm and
levels off there rather than closing.

The decisive evidence is the **cohesion**: the filler expands far less than either
twin — thickness 11.4 → 44.2 mm, flattening over the last quarter (42.5 → 44.1 →
44.2) — where the inert twin's filler shreds (damage 0.523, thickness still
climbing at 81.2 mm) and the reactive twin's latches 1.000 and is flung to
119.6 mm with the plates driven 81.4 mm apart. Note the NERA sandwich separates
*less* than the inert one (21.1 vs 24.2 mm): a cohesive interlayer **holds the
plates together** as well as holding the bulge open — shredded filler offers no
such restraint. Confirmed visually (viewer `--shots`): the interlayer stays
**large coherent bent slabs**, split around the rod channel but intact, with the
spall spray coming from the *steel plates*, not the filler. Cohesive, unignited,
stable — no NaN, no collapse.

*(These figures were re-measured after the domain/geometry change; the branch
verification — damage exactly 0.000, filler expanding far less than either twin —
reproduced, only the magnitudes moved. The earlier `bulge.py` probe hardcoded the
old x-bands and silently reported `sep = nan` when the armor moved, which is why
filler metrics are now keyed off `material_id` alone.)*

**Model-mechanics note — NOT an armor-performance claim.** In the same bakes the
rod ends up shallower, slower, and more damaged against the NERA filler than
against either ERA twin (tip 259.1 vs 267.1/267.0 mm; median intact-rod speed
1084 vs 1220/1223 m/s; rod damage 0.549 vs 0.482/0.532). **Do not read this as
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
