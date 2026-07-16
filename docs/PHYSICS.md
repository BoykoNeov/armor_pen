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

### 1.2 The penetrator is pointed, not flat-faced

The rod was long seeded as a plain rectangle — a flat-faced cylinder that struck
the plate face-first. Real APFSDS long rods are **pointed** (conical or ogival).
`_seed` now carves a nose out of the rod rectangle, in rod-local coords where the
tip leads, *before* the §3.2 rotation (which is about the tip, so the carve leaves
it in place). `nose_shape` ∈ {`conical` (default), `ogive` (tangent ogive),
`blunt`} and `nose_length` (default 1.5 calibers) are deck data, so the nose is
scenario data like everything else (CLAUDE.md §9). Profiles are illustrative, not
any real system's geometry (§10).

**Be precise about what this buys.** The nose exists mainly for *flight
aerodynamics* and initial bite. At ordnance velocity it is consumed within the
first microsecond, after which penetration is the **eroding/hydrodynamic regime**
(§5, Tate–Alekseevskii) in which final depth is nearly *nose-shape-independent*.
So this is **geometric realism, not a penetration-accuracy fix** — and the sim
reproduces exactly that textbook expectation. Measured against a `blunt` control
twin (same deck, same probe, nose the only difference):

- **Final penetration is unchanged**: rod tip 232.3 mm pointed vs 232.7 mm blunt
  at 0° (−0.2 %). The nose does not buy depth, and the sim agrees.
- **The early crater is where it shows**: at 6.4 µs the pointed rod has spalled
  **63 % less** RHA (0.004 vs 0.010). It cleaves in where the flat face slaps the
  surface and throws a crater lip. By ~24 µs both have mushroomed into the same
  eroding head — the nose is gone and the two histories converge.
- The rod itself ends **~10 % less damaged** (0.396 vs 0.441): a gentler initial
  shock.

**Confound, stated not hidden:** the nose is carved *out of* the rod rather than
added in front of it, so a pointed rod is ~10 % lighter than the blunt one at
equal length. That is deliberate — compensating by lengthening the rod would
change every scenario — but it means "pointed vs blunt" above is really
"pointed vs blunt-and-10 %-heavier". It does not affect any A/B in §3, where both
arms share one nose.

`heat_vs_composite` inherits the conical nose too. A shaped-charge jet is not a
pointed rod, but that deck is a **rod stand-in** awaiting the jet model
(milestone 7), so the nose is no more wrong there than the rod already was.

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
**0° (normal incidence) the reactive layer does not degrade the penetrator** —
measured against an *equal-areal-mass inert twin* (`era_filler_inert`: identical
density/stiffness/thickness, reactivity off), the rod is untouched: penetration
past the main-plate face differs by **+0.3 %** (69.2 vs 69.0 mm — both decks
perforate and the residual flies clear), rod damage by +2.3 %, residual velocity
by +2.2 %. If anything the reactive rod is marginally *faster* and *deeper* — the
opposite of protection, and coherent: the detonation clears filler off-axis, so
the reactive rod pushes through slightly less on-path material than the inert rod
that keeps its filler in the channel.

This is **correct physics, not a bug**: at 0° the detonation flings the plates
*laterally, symmetric about the rod axis*, so the debris sweeps sideways and
never crosses the rod path to cut it. Real reactive armor gets its effectiveness
from **obliquity** (§3.2).

**The backing plate is a separate question from the rod, and the answer differs.**
At 0° the main plate does spall ~8 % less in the reactive deck (0.222 vs 0.243),
driven by the same forward-shove mechanism §3.2 documents at 55°: the detonation
pushes the plate body 1.6 mm further downrange than the inert twin's. That margin
is **~80× the run-to-run scatter** (§3.2), so it is not numerical noise — but its
*sign flipped* when the geometry changed from a floating block to a plate (it
used to read reactive marginally **worse**), so it sits inside **model**
uncertainty even though it clears **numerical** noise. Those two error bars are
different sizes and must not be conflated. Read the 0° arm as: **the null is about
the penetrator** — robust, mechanistically explained, and unchanged across every
geometry tried — while the plate-side margin is real-but-not-portable, and only
earns confidence at obliquity where it is twice the size and sign-stable.

The A/B decks (`apfsds_vs_era` / `apfsds_vs_era_inert`) are byte-identical in
geometry, areal mass, nose, and timing, so these deltas cleanly isolate the
*reactive* contribution. Obliquity is milestone 6, below.

### 3.2 Oblique reactive armor (milestone 6)

At obliquity the rod is tilted relative to the plate normal, so the
detonation-flung plates gain a velocity component **perpendicular to the rod**
(zero at 0°, `∝ sin θ`) and the interaction is no longer symmetric about the rod
axis. Implementation is minimal and protects the validated M1–M5 physics: the
projectile **rectangle** is rotated by `angle_deg` about its tip so the rod
strikes *nose-first* along its velocity (mpm.py `_seed`), while the armor slabs
stay vertical/axis-aligned. Only the *relative* rod-axis/plate-normal angle is
physical, so rotating the rod against fixed slabs is frame-equivalent to tilting
the slabs against a horizontal rod — and the *rotation* at `angle_deg=0` is still
exact identity (`ca=1, sa=0`). (The rod it rotates is no longer the old
rectangle — §1.2 carves a nose out of it first, at every angle — so normal-incidence
decks no longer seed bit-for-bit as they did before milestone 6; the rotation is
what is identity, not the seeding.) Decks:
`apfsds_vs_era_oblique` (+ its `_inert` twin), 55° from the normal, in a 220 mm
domain with the impact deliberately **off-centre** (`impact_y: 145`): the rod
drops `~tan 55° ≈ 1.43` mm in y per mm of x, so it needs ~145 mm of descent below
the impact but only its own tilted body-length (~49 mm) of headroom above it.
Centring the impact would demand a domain ~2× taller — and ~2× the particles — to
buy headroom the rod never uses.

**Verified result — protection, but not rod-cutting.** Measured against the
equal-areal-mass inert twin (both decks seed at **286 355** particles), at 55° the
reactive layer **measurably protects the backing plate**:

- **Main-plate spall ≈ 16 % lower** for the reactive deck (0.137 vs 0.163 at the
  final frame) — roughly **double the ~8 % the same mechanism buys at 0°** (§3.1).
- **The rod is not cut, deflected, or meaningfully slowed on-path.** Penetration
  past the plate face differs by **+0.8 %** (50.7 vs 50.3 mm — the reactive rod is
  marginally *deeper*) and rod damage by −0.5 %: both at the level of "no effect".
  Residual velocity *is* ~8.5 % lower (679 vs 741 m/s) — the one real rod-side
  effect. A tough tungsten rod is not severed by thin, few-hundred-m/s flyers.
  (The "erode/deflect the rod" outcome was an *a priori* expectation; the sim says
  protection arrives through the backing plate instead — reported as it came out.)
- **Mechanism — and it scales with angle, which is the real evidence.** The
  detonation **shoves the main plate forward**: the plate body ends 7.7 mm further
  downrange than the inert twin's, and its front face 18.2 mm further (the inert
  plate's face travels *backward*, cratering and throwing lips upstream). A plate
  moving *with* the rod reduces effective (rod-relative) penetration — the textbook
  "moving / standoff plate defeats less penetrator." The shove grows 1.6 mm → 7.7 mm
  from 0° to 55°, and the spall protection tracks it 8 % → 16 %. **One mechanism,
  monotone in obliquity, consistent across both decks** — a far stronger argument
  than any single number.

Honesty caveats (root §1/§10): a steeper angle was **not** chased (it only moves
`sin θ` 0.82→0.91 and worsens domain fit), and `detonation_pressure` was **not**
cranked to force rod degradation (that would be confirmation-bias tuning toward
defeating a system — off-limits per §10).

**Two error bars, different sizes — do not conflate them.**

1. **Numerical (run-to-run) scatter: ≤ 0.11 %.** Measured directly, by re-baking
   identical decks and re-measuring: every aggregate metric above reproduces to
   ≤0.11 % (the 55 % protection figure lands on 15.8 % both times). MPM grid
   `atomic_add` ordering is non-deterministic, but at this level it is negligible.
   The ~16 % protection is therefore **~150× the numerical noise floor** — this is
   signal, full stop. (This measurement replaces an earlier, weaker argument from
   "the A/B gap grows monotonically over the event". That trend held for the
   blunt rod, +0.023 → +0.032 → +0.035 at the 50/75/100 % marks, but does **not**
   survive the pointed nose: +0.021 → +0.027 → +0.026. The conclusion is unchanged
   and now rests on the repeat bake, which is what should have carried it.)
2. **Model sensitivity: large, and the honest limit on all of this.** The *same*
   A/B has read ≈ 40 % (old floating-block geometry), ≈ 21 % (plate geometry,
   blunt rod) and ≈ 16 % (plate geometry, pointed rod). The **sign is robust across
   every condition tried; the magnitude is not portable.** Quote it as "roughly
   10–20 %, sign-stable", never as a figure. At 0° the margin's sign has actually
   flipped across a geometry change (§3.1) — which is why 55° earns confidence and
   0° does not, despite both clearing the numerical floor.

Read every number here as **plausible and internally consistent, not predictive**
(root §1). Spalled rod fragments still reach the bottom
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
**179 189** particles, so they are equal-areal-mass arms of one A/B family
differing only in the filler's response path.

**The branch works as specified.** Across all 550 frames the NERA filler's damage
fraction is **0.000** — it never ignites and never spalls. This is the decisive
claim and it reproduces exactly, unchanged by the pointed nose.

The supporting evidence is the **cohesion**: the filler expands far less than
either twin — thickness (1st–99th percentile x-span) 11.8 → **39.5 mm**, where the
inert twin's filler shreds (damage 0.462) and spreads to 83.5 mm and the reactive
twin's latches 1.000 and is flung to 125.6 mm. Confirmed visually (viewer
`--shots`): the interlayer stays
**large coherent bent slabs**, split around the rod channel but intact, with the
spall spray coming from the *steel plates*, not the filler. Cohesive, unignited,
stable — no NaN, no collapse.

**The bulge is a profile, not a number — and the old comparison was wrong.**
Plate separation must be measured *beside* the rod channel: inside it the plates
are perforated and their material is dragged downrange, which reads as a huge
"gap" that is debris transport, not bulge. Measured as the median-x separation of
the two steel plates, tracked by frame-0 identity, in a band `12 < |y − axis| <
25 mm`, the NERA sandwich opens **18.0 → 21.1 mm** and holds — it does not collapse
back. But the bulge **decays with distance from the channel** (24.3 mm at a
10–20 mm band, 21.1 at 12–25, 18.4 at 15–30), so any single separation figure is
an artifact of where it was sampled. Quote the *shape*, not the number.

Against that, the inert twin reads **18.0 → 18.5 mm — flat at every band**: it
does not bulge at all. **This reverses the earlier claim that "the NERA sandwich
separates *less* than the inert one (21.1 vs 24.2 mm)".** It separates *more*, and
the old figure of 24.2 is reproducible here only as *NERA measured at a different
band* — i.e. the two arms of that comparison were almost certainly not sampled
alike. The corrected result is also the one the construction predicts: `nera_filler`
is `reactive=True`, so it skips both `_return_mapping` and `_update_damage` and is
perfectly elastic — it *springs* the plates apart and holds them there — whereas
the inert filler yields and shreds (damage 0.462) and dissipates the shock instead
of storing it. A spring bulges; mush does not. (See the model-mechanics note below:
this is a statement about the model's construction, not about armor.)

*(These figures were re-measured after the domain/geometry change and again after
the pointed nose; the branch verification — damage exactly 0.000, filler expanding
far less than either twin — reproduced both times, only the magnitudes moved. The
earlier `bulge.py` probe hardcoded the old x-bands and silently reported
`sep = nan` when the armor moved, which is why filler metrics are now keyed off
`material_id` alone and bands are defined relative to the rod axis.)*

**Model-mechanics note — NOT an armor-performance claim.** In the same bakes the
rod ends up shallower, slower, and more damaged against the NERA filler than
against either ERA twin (tip 261.8 vs 268.3/268.1 mm; median intact-rod speed
1146 vs 1241/1230 m/s; rod damage 0.487 vs 0.452/0.442). **Do not read this as
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
