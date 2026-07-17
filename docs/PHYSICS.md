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

SPH was long hedged as a possible return "for HEAT-jet fluid-like erosion".
**That is now settled, and the answer is no** (milestone 7, §3.4): MPM reproduces
jet stretching to within 0.1 % of the kinematic prediction with no new kernel, and
fluid-like erosion needs no special path at all — at a 7 km/s stagnation point the
jet's yield is ~1000× below the pressure, so the existing von Mises return mapping
caps deviatoric stress near zero by itself. The hedge is retired on evidence, not
abandoned on preference. (The real gap a jet exposed was the missing **equation of
state** for the volumetric response — closed in §3.5 — and SPH would not have
fixed that either.)

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
inside the domain.

#### 1.1.1 The high walls never fired (found in milestone 13)

> This section used to end: *"The clamp is memory safety, not physics — the slip
> wall already removes wall-normal velocity, so it almost never binds."* Every
> clause of that was false on two of the four walls, and the sentence is kept here
> because **the document asserted exactly the invariant the code was violating.**
> A stated invariant is not a tested one.

`_grid_op` tested the far walls as `i > nx - bound`. `nx` is the **allocated**
width, and the grid carries **3 pad nodes past the domain** so that a particle
sitting on the position clamp has somewhere for its 3×3 stencil to land. So the
high band lay entirely in the pad — outside the material, in a region the clamp
guarantees is empty. Measured across four deck shapes: **8 of 8 high walls
unreachable**. Since milestone 1.

The low walls worked the whole time, because grid indices count from `0` and
`i < bound` is genuinely inside the domain. That asymmetry is why it survived so
long: every bake ran a **working mirror on its low edges and no wall at all on its
high ones**, and a half-correct boundary looks like a boundary.

**What replaced the missing wall is worse than no wall.** With nothing to zero the
inbound normal velocity, `_g2p`'s position clamp becomes the boundary condition by
default — and it is a *vice*, not a mirror: infinitely rigid, and it arrests
**displacement** while leaving **velocity** untouched. Material piles onto the clamp
plane still carrying its full inbound speed and is crushed there by everything
behind it. In `apfsds_vs_era`, 2342 particles sat welded to `y = 119.61` (the clamp
plane exactly) reading 1699 m/s, for 130 frames.

The asymmetry is visible in any bake that reaches a wall, and this is the cheapest
way to check it — the deck is symmetric about `y = 60` by construction, so the
material must be too:

| | dead high wall | walls live |
|---|---|---|
| `rha` `pos_y` | 0.88 … **119.61** (on the clamp) | 0.88 … 119.12 |
| mirrored about 60 | 0.39 vs 0.88 — **asymmetric** | **0.88 vs 0.88 — exact** |
| particles on a clamp plane | 2342 | **0** |

**Milestone 13 did not cause this; it made it visible**, by giving the solver an
energy equation. `era_filler` reads `e` in its EOS stress branch, and `e` on the
pinned set jumped **24 → 7.1e5 J/kg in exactly the frames the particle reached the
clamp** — 30× the rest of the filler, while the *median* was unchanged (2945 vs
2815). The pinned **surface**, never the bulk: §3.6.1's rule again, an extremum is
not a state. The bake then diverged at frame 190. `apfsds_vs_nera` stayed clean
throughout, and the reason names the mechanism: **an inert filler is never
detonated into the ceiling.**

The fix tests against the **domain's far corner in grid coordinates**, as a float —
`(xmax − xmin)/dx` is not an integer in `y` for a typical deck (307.2 for the ERA
deck), and rounding it up is what created the pad in the first place.
`tests/test_boundary_walls.py` pins reachability *derived from the position clamp*
rather than from `_grid_op`, so it cannot be satisfied by copying the kernel's own
mistake.

Fixing it cleared every symptom at once, which is what makes it causal rather than
correlated — one kernel change, and two counters go to **exactly zero**:

| | dead walls | walls live |
|---|---|---|
| `v`/`F`/`e` non-finite | substep 197637 | **never** |
| J floor fired | 269 509 | **0** |
| resolution guard fired | 217 829 | **0** |
| worst clamped `e` | **−inf** | −0.087 J/kg (roundoff) |
| CFL audit | *** DIVERGED *** | OK, worst live J = 0.7166 |

Neither the J floor nor the resolution guard was ever an EOS problem. Both were
firing on material being crushed against the clamp.

**The armor still touches the walls, and it should** — `_seed` lays slabs across the
full domain height *precisely so* the mirror makes them a plate that continues
beyond the frame. Material at the wall is not the defect; a wall that isn't there
is. The per-deck sizing duty above is unchanged and still about the **rod** and its
spray (which stays at `y = 45.9…74.1` here, nowhere near a boundary).

**⚠️ Every figure in this document measured near a boundary is affected**, and every
deck has armor at the walls. Re-measure; do not translate. The ERA/NERA numbers are
the most exposed, since the detonation drives filler straight into the ceiling.

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

`heat_vs_composite` used to inherit the conical nose while it was still a rod
stand-in. It no longer does: milestone 7 (§3.4) made it a real jet, and a jet is a
stretching column with no machined nose to speak of, so that deck sets
`nose_shape: blunt`. The choice is close to free either way — the measurement
above found final penetration nose-shape-independent already at 1.6 km/s, and the
jet is 4× faster and further into the eroding regime.

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
  stress is Cauchy von Mises (the momentum-driving stress), it reads
  *approximately* capped near yield rather than exactly. It used to grow a wild
  over-read tail at the shock front (~327 GPa at a jet tip) because the volumetric
  response had no equation of state; **§3.5 removed that cause**, so the tail
  should be gone rather than clamped away. A viewer-side percentile clamp is still
  a fine colormap default — it is just no longer covering for the physics.
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
| Copper jet | The shaped-charge jet: soft, dense, and **velocity-graded**, so it stretches in flight and erodes fluid-like. Its yield is ~1000× below its own stagnation pressure, so it flows without any special "fluid" path. *(verified — §3.4.)* |

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
density/stiffness/thickness, reactivity off), the rod is untouched: rod damage
differs by **−0.9 %** and residual velocity by **+0.9 %** (1022.7 vs 1013.8 m/s),
with the rod tip **+0.3 %** downrange — both decks perforate and the residual flies
clear, so that last one is a free-flight position, not resistance. If anything the
reactive rod is marginally *faster* — the opposite of protection, and coherent: the
detonation clears filler off-axis, so the reactive rod pushes through slightly less
on-path material than the inert rod that keeps its filler in the channel.
*(Re-measured 2026-07-17 under M13 by `tools/measure_reactive_ab.py`. The rod null
is the most robust claim in §3.1/§3.2: it has now read +2.2 % and +0.9 % across
three physics changes and stayed inside the noise every time. The old
"penetration 69.2 vs 69.0 mm" figures came from M5's uncommitted probe and are
**not reproducible** — do not quote them.)*

This is **correct physics, not a bug**: at 0° the detonation flings the plates
*laterally, symmetric about the rod axis*, so the debris sweeps sideways and
never crosses the rod path to cut it. Real reactive armor gets its effectiveness
from **obliquity** (§3.2).

**The backing plate is a separate question from the rod, and the answer differs.**
At 0° the main plate does spall **~17 % less** in the reactive deck (**0.2732 vs
0.3297**, re-measured 2026-07-17 by `tools/measure_reactive_ab.py` under M13 —
Mie-Grüneisen (§3.10) plus the §1.1.1 boundary fix; it read 0.246 vs 0.282 (~13 %)
post-EOS and 0.220 vs 0.242 (~9 %) under the pre-EOS law — **the conclusion has now
survived three physics changes and the magnitude has moved on every one of them**,
9 → 13 → 17 %, with both arms' absolute damage rising each time),
driven by the same forward-shove mechanism §3.2 documents at 55°: the detonation
pushes the plate body **1.58 mm** further downrange than the inert twin's
(re-measured post-EOS; +1.44 mm under the pre-EOS law, same probe). That margin
is **~80× the run-to-run scatter** (§3.2), so it is not numerical noise — but its
*sign flipped* when the geometry changed from a floating block to a plate (it
used to read reactive marginally **worse**), so it sits inside **model**
uncertainty even though it clears **numerical** noise. Those two error bars are
different sizes and must not be conflated. Read the 0° arm as: **the null is about
the penetrator** — robust, mechanistically explained, and unchanged across every
geometry tried — while the plate-side margin is real-but-not-portable, and only
earns confidence at obliquity where it is twice the size and sign-stable.

Note the penetrator null is *fully* a null only here at 0°, where even residual
velocity moves by +2.2 % (i.e. nothing, and in the wrong direction for protection).
At 55° the rod is still never cut, but it *is* measurably slowed — see §3.2. "Not
cut" and "not affected" are different claims; only 0° supports the stronger one.

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
reactive layer **measurably protects the backing plate**.

> **⚠️ RE-MEASURED 2026-07-17 (milestone 13). Every conclusion below survived;
> every NUMBER moved.** The figures are now from `tools/measure_reactive_ab.py` — a
> **committed** tool, which this A/B did not have before: M5/M6 used an ad-hoc probe
> that was never checked in, so its numbers could only be quoted, never re-derived
> (the same defect §3.3 records for the plate-separation figures). Two changes
> invalidated the old values: Mie-Grüneisen (§3.10) and the boundary-condition fix
> (§1.1.1), which hits *these* decks hardest — the detonation drives filler straight
> into a ceiling that, until now, was not there.
>
> | | reactive | inert | delta | M6 quoted |
> |---|---|---|---|---|
> | 0° main-plate spall | 0.2732 | 0.3297 | **−17.1 %** | ~−8 % |
> | 0° rod residual v | 1022.7 | 1013.8 | +0.9 % (null) | +2.2 % (null) |
> | 55° main-plate spall | 0.1033 | 0.1741 | **−40.7 %** | −16 % |
> | 55° rod residual v | 540.5 | 579.0 | **−6.7 %** | −8.5 % |
> | 55° rod damage | 0.7267 | 0.7172 | +1.3 % (not cut) | −0.5 % (not cut) |
>
> **Absolute values are NOT comparable to M6's** (540/579 m/s vs 679/741): its probe
> is gone, so its metric definitions are unknown. Quote the tool, not M6.
>
> This is the "model sensitivity" error bar below doing exactly what it warns of —
> 40.7 % is back inside the 40/21/16 % range this same A/B has already read. The
> *structure* is what is robust: protection at both angles, roughly doubling with
> obliquity (17 → 41 %, where M6 had 8 → 16 % — both ~2.4×), rod not cut at either,
> 0° a rod null and 55° a real modest slowing.

- **Main-plate spall ≈ 41 % lower** for the reactive deck (0.1033 vs 0.1741 at the
  final frame) — roughly **double the ~17 % the same mechanism buys at 0°** (§3.1).
- **The rod is not cut or deflected — but it *is* slowed. Do not conflate those.**
  Rod damage differs by +1.3 % (no effect) and the rod is not severed or turned:
  thin few-hundred-m/s flyers cannot cut a tough long rod, and the *a priori*
  "flyer sweep erodes the rod" expectation is what failed here — reported as it
  came out. But residual velocity **is 6.7 % lower** (540.5 vs 579.0 m/s), ~60× the
  numerical floor below: that is a real, modest degradation. "Not cut" ≠ "not
  affected"; only the first is a null.
  *(Rod tip position reads −2.2 % — but do not lean on it in either direction: both
  rods fully perforate, so at the final frame that number is a free-flight
  **position**, not penetration resistance. Velocity is the leading rod-degradation
  indicator; the position is a snapshot.)*
- **Mechanism — and it scales with angle, which is the real evidence.** The
  detonation **shoves the main plate forward**: the plate body ends 7.7 mm further
  downrange than the inert twin's, and its front face 18.2 mm further (the inert
  plate's face travels *backward*, cratering and throwing lips upstream). A plate
  moving *with* the rod reduces effective (rod-relative) penetration — the textbook
  "moving / standoff plate defeats less penetrator." The shove grows 1.6 mm → 7.7 mm
  from 0° to 55°, and the spall protection tracks it 17 % → 41 %. **One mechanism,
  monotone in obliquity, consistent across both decks** — a far stronger argument
  than any single number.
  *(**The two plate-shove distances — 1.6 mm and 7.7 mm — are NOT re-measured** and
  predate both M13 and the §1.1.1 boundary fix; `measure_reactive_ab.py` does not
  compute them. Stated, not buried: the protection ratio they are paired with was
  re-measured and moved 8→17 % and 16→41 %, so assume these moved too. What survives
  is that the shove grows with obliquity and the protection tracks it — the
  *monotone relationship*, not the millimetres.)*

Honesty caveats (root §1/§10): a steeper angle was **not** chased (it only moves
`sin θ` 0.82→0.91 and worsens domain fit), and `detonation_pressure` was **not**
cranked to force rod degradation (that would be confirmation-bias tuning toward
defeating a system — off-limits per §10).

**Two error bars, different sizes — do not conflate them.**

1. **Numerical (run-to-run) scatter: ≤ 0.11 %.** Measured directly, by re-baking
   identical decks and re-measuring: every aggregate metric above reproduces to
   ≤0.11 % (the 55° protection figure landed on 15.8 % both times when it *was*
   ~16 %; the repeat bake has not been redone since M13 moved it to 40.7 %, and the
   scatter is a property of the solver's `atomic_add` ordering rather than of the
   value, so it carries — but it is an inherited measurement, not a fresh one). MPM
   grid `atomic_add` ordering is non-deterministic, but at this level it is
   negligible. The ~41 % protection is therefore **hundreds of times the numerical
   noise floor** — this is signal, full stop. (This measurement replaces an earlier, weaker argument from
   "the A/B gap grows monotonically over the event". That trend held for the
   blunt rod, +0.023 → +0.032 → +0.035 at the 50/75/100 % marks, but does **not**
   survive the pointed nose: +0.021 → +0.027 → +0.026. The conclusion is unchanged
   and now rests on the repeat bake, which is what should have carried it.)
2. **Model sensitivity: large, and the honest limit on all of this.** The *same*
   A/B has now read ≈ 40 % (old floating-block geometry), ≈ 21 % (plate geometry,
   blunt rod), ≈ 16 % (plate geometry, pointed rod) and **≈ 41 %** (M13: MG + the
   §1.1.1 boundary fix). The **sign is robust across every condition tried; the
   magnitude is not portable** — and the M13 value landing back on the *first*
   figure in that list, after four intervening changes, is the sharpest available
   demonstration that the magnitude carries no information. Quote it as **"tens of
   percent, sign-stable"**, never as a figure.
   *(This entry used to advise quoting "roughly 10–20 %". That was itself a
   magnitude claim dressed as a range, and M13 walked straight out of it. The
   lesson is not to widen the band each time — it is that the band is not the
   result.)*
   At 0° the margin's sign has actually
   flipped across a geometry change (§3.1) — which is why 55° earns confidence and
   0° does not, despite both clearing the numerical floor.

Read every number here as **plausible and internally consistent, not predictive**
(root §1). Spalled rod fragments still reach the bottom
wall late in the run (the intact rod clears it by ~35 mm), a *shared* artifact of
both decks — another reason to read the A/B **delta**, not the absolutes.

### 3.3 NERA persistent bulge — the unignited branch (verified)

> **⚠️ MILESTONE 12 REPLACED THE MATERIAL THIS SECTION MEASURES. READ §3.6.2 FIRST.**
>
> Every NERA figure below was measured on a `nera_filler` that was `reactive=True`
> with `ignition_compression=0` — a filler that could **neither yield nor break**.
> That was a mis-encoding, not a design (§3.6.2), and M12 replaced it with a
> non-reactive ductile filler (`yield_strength` 50 MPa and `damage_threshold` 3.0,
> both now **live**). **The numbers below are therefore stale, and one of them is
> now false outright:**
>
> - *"Across all 550 frames the NERA filler's damage fraction is **0.000** … This
>   is the decisive claim"* — **no longer true, and deliberately so.** The M12
>   filler spalls **18.65 %**. It never *ignites* (that was always about
>   `ignition_compression`, and it is now simply non-reactive), but it can now
>   **tear**, because a filler that can never break is the defect M12 removed.
> - The **cohesion claim survives on a better footing.** Against the shredding twin
>   — now one field apart (`dthr` 3.0 vs 0.02) instead of confounded — NERA spalls
>   **18.65 % vs 69.59 %** and keeps **66.0 % vs 30.4 %** of its filler coherent.
>   That is the real claim, and it is the single-variable test this section's own
>   closing paragraph used to ask for and call "not done here."
> - The **plate-separation figures (16.1 / 21.1 mm) have NOT been re-measured**,
>   and are not merely stale — an independent probe reading exactly 18.000 mm at
>   `t=0` does not reproduce them even on the **pre-M12** bake (it gets 14.1
>   plate-wide and 13.3 banded, i.e. the *opposite sign* beside the channel). M5's
>   probe was ad-hoc and never committed as a tool, so the disagreement is
>   unresolved. **Do not quote 16.1 / 21.1 until the metric is rebuilt and
>   committed.** Flagged rather than quietly re-derived.
>
> The *structural* claim of this section — a cohesive interlayer behaves unlike a
> shredding one, and the two disagree in sign between plate-wide and banded
> metrics — is unaffected. The magnitudes are not.

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
claim and it reproduces exactly, unchanged by the pointed nose. *(Pre-M12. The
filler now spalls 18.65 % by design — see the box above.)*

The supporting evidence is the **cohesion**: the filler expands far less than
either twin — thickness (1st–99th percentile x-span) 11.8 → **39.5 mm**, where the
inert twin's filler shreds (damage 0.462) and spreads to 83.5 mm and the reactive
twin's latches 1.000 and is flung to 125.6 mm. Confirmed visually (viewer
`--shots`): the interlayer stays
**large coherent bent slabs**, split around the rod channel but intact, with the
spall spray coming from the *steel plates*, not the filler. Cohesive, unignited,
stable — no NaN, no collapse.

**The bulge is a profile, not a number — and *where* you measure decides the sign.**
Two facts, both true, and they only look contradictory if the metric is left
implicit:

- **Plate-wide** (median-x separation of the two steel plates over the full plate
  height — the metric the original probe used): NERA **18.0 → 16.1 mm**, the inert
  twin **18.0 → 18.5 mm**. The NERA sandwich ends up *tighter* than the inert one.
- **Beside the channel** (same median, restricted to `12 < |y − axis| < 25 mm`):
  NERA **18.0 → 21.1 mm**, the inert twin a flat **18.5 mm**. Locally the NERA
  sandwich is *open wider*, and the bulge **decays with distance** (24.3 mm at a
  10–20 mm band, 21.1 at 12–25, 18.4 at 15–30).

Both readings describe one behaviour, and it is the one milestone 5 claimed: **a
cohesive interlayer holds the bulge open where the rod passes while holding the
plates together everywhere else.** A cohesive, never-spalling filler that is
stretched open near the channel must pull the plates *in* further out; the inert
filler shreds (damage 0.462), restrains nothing, and its plates simply stay at
18.5 mm at every band. This is what the construction predicts: `nera_filler` is
`reactive=True`, so it skips both `_return_mapping` and `_update_damage` and is
perfectly elastic — it stores the shock and springs — where the inert filler yields
and dissipates it. (See the model-mechanics note below: that is a statement about
the model's construction, not about armor.)

> **⚠️ M12: that explanation described the mis-encoding, not the physics.** The
> filler is no longer perfectly elastic and no longer "stores the shock and
> springs" — it yields at 50 MPa and tears at `dthr=3.0`. It remains far more
> cohesive than the shredding twin (18.65 % vs 69.59 % spall), so the *behaviour*
> this paragraph describes still has a mechanism; it is now **dissipative cohesion**
> rather than **elastic inability to fail**. Measured attribution (§3.6.2): the
> separation change is driven by **plasticity, not spall** — a plasticity-only
> control with 0 % spall gives near-identical separation (53.8 vs 53.6 mm banded).

Never measure this *inside* the channel: there the plates are perforated and their
material is dragged downrange, which reads as a large "gap" that is debris
transport, not bulge. And never compare a plate-wide figure against a banded one —
they disagree in *sign*, so quoting either without its definition is meaningless.

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

> **✅ M12 DID IT — this paragraph's own prescription is now what ships.**
> `nera_filler` is non-reactive with `damage_threshold=3.0`, so it yields *and*
> tears; the fields above are **live**, not dead. The A/B against
> `era_filler_inert` differs in **exactly one field** (`dthr` 3.0 vs 0.02), pinned
> by `tests/test_nera_dissipation.py`, so it now isolates cohesion at equal
> toughness — "cohesive vs shredding" rather than "unbreakable vs shredding". It
> cost **zero kernel code**: both gates key off `reactive > 0.5`.
>
> **The rod deltas quoted just above (tip 261.8, speed 1146, damage 0.487) were
> measured on the pre-M12 filler and are NOT re-measured.** The old confound is
> gone, but so is the material — do not quote them as the current model's answer,
> and do not read the *un*-confounding as promoting them to an armor claim. The
> paragraph's core warning stands on its own: a spalled particle keeps its momentum
> but drops its stress term, so this illustrates the **damage model**, not armor.

Honesty caveats (root §1/§10): one bake per condition, and MPM grid `atomic_add`
ordering is non-deterministic — but the deltas are large, monotonic, and
sign-stable across many frames, so they are not noise. The main-plate spall
fractions quoted here are measured over a frame-0 x-band and are **not**
comparable to the differently-measured 0° absolutes in §3.1 (the *ordering* there
— reactive ≈ inert, reactive marginally worse — does reproduce). The
never-yields property also means the filler stores elastic energy without
dissipating it, i.e. stiffer-than-real; that is a modelling limitation, not a
bug, and it is another reason the rod deltas above are model-specific.

### 3.4 Shaped-charge jet (milestone 7)

A shaped-charge jet is not a fast rod. What makes it a jet is that it is
**velocity-graded**: the tip flies at ~7 km/s and the tail at ~2 km/s. Nearly
everything jet-characteristic is a *consequence* of that one initial condition,
which is why this milestone needed **no new kernel and no SPH** — only a per-
particle seeded velocity (`Projectile.tail_velocity`, mpm.py `_seed`).

**Scope, and it is load-bearing (root §10).** We seed an **already-formed** jet.
Liner collapse and the explosive that drives it are *not modelled* — deliberately.
That keeps the project on the public-physics side of §10 by construction, and
costs nothing: the gradient is the only part that matters downstream. This is
textbook Birkhoff/PER jet theory (§5).

**Verified: the jet stretches, at the right rate.** Stretching is *kinematic* —
each element flies at its own constant speed, so the jet elongates — which makes
it predictable a priori and therefore falsifiable. Tip-to-tail length is a
**confounded** metric (the tip erodes against armor while the tail falls back), so
the measurement instead tracks **Lagrangian markers**: the cache contract fixes the
particle count and keeps particles persistent (CACHE_FORMAT §5), so a particle
index is a material label. Two 4 mm bands in the free-flight body, at 60 mm and
110 mm behind the tip:

The shipped A/B is `heat_vs_composite` against `heat_vs_composite_uniform` — the
same copper, geometry, mass, nose, timing and particle count (9210 both), with
`tail_velocity` omitted in the control, so **the gradient is the only variable**:

Re-measured after the milestone-8 EOS, on the 30 µs window both decks now carry
(§3.4's timing note). Same probe both rows: markers 60 and 110 mm behind the tip,
tracked while both are in free flight (i.e. until either latches `damage`).

| deck | material | predicted | measured | separation | body length |
|---|---|---|---|---|---|
| `heat_vs_composite_uniform` | copper | 0 mm/µs | **−0.025 mm/µs** | 50.0 → 47.6 mm | 118.6 → **43.7** mm |
| `heat_vs_composite` (the jet) | copper | 2.083 mm/µs | **2.093 mm/µs** (+0.5 %) | 50.0 → 112.4 mm | 118.6 → **151.1** mm |

**The claim survives the EOS, which is the point of re-measuring rather than
assuming.** Stretching is kinematic — each element flies at its own seeded speed,
so the rate is material-independent *by construction* and an equation of state
should not touch it. Measurement agrees it barely did: the identical deck on the
**pre-EOS solver** gives **2.0851 mm/µs (+0.10 %)** against **2.0933 (+0.50 %)**
now. A rate is a slope, so those two are directly comparable even though the
windows differ; the separation and body columns are not, and are quoted for the
current 30 µs window only. The prediction is still met to well under a percent —
the agreement is simply no longer suspiciously perfect. "Immune by construction"
was a reason to check it, not a reason to skip checking.

> **Milestone 13 — checked, not skipped, and the distinction is the point.** MG
> (§3.10) and the §1.1.1 boundary fix moved most figures in this document, and this
> one is expected to be immune for **two independent reasons**: the markers are in
> **free flight** (no grid coupling to lean on), and the jet's **nearest approach to
> any wall is 32.2 mm** — measured on the M13 cache, not assumed — so a boundary
> change cannot reach it. Both were verified rather than argued. The figures above
> stand as the pre-M13 measurement; the *reasons* they should not move are what was
> re-checked. This is the same posture as the sentence above it: a claim that ought
> to be immune is a claim worth confirming, and "checked and unaffected" is a
> different statement from "not re-measured".

*(The pre-EOS table quoted `−0.064 mm/µs`, `50.0 → 45.3`, `117.5 → 77.5` for the
control plus two development tungsten rows. The graded row reproduces exactly under
this probe — 2.085 and 50.0 → 101.7 at the old 25 µs window — so the method matches
where the claim lives. The control row does not: the earlier figure ran past the
markers' free flight into their erosion, where this probe stops. The tungsten rows
were development-only, their decks no longer exist, and they are dropped rather
than left silently un-re-measured.)*

The control is the decisive half: the jet's body **stretches** +40.4 mm while the
control's **shortens** −40.0 mm by tip erosion — near-perfectly symmetric, and
opposite in sign. Residual from a straight line is 0.001 mm (copper) and 0.020 mm
(tungsten), i.e. ballistic free flight.

**The rate reproduces across two materials whose yields differ 7.5×** (+0.1 % and
+0.0 %). That is not redundancy — it is the point: the prediction is computed from
the *seeded velocities alone*, so a material-independent result is what "kinematic"
has to mean. If strength mattered to the rate, these two rows would disagree.

**Strength does show up, just not in the rate — and it is a real finding, not
noise.** The two controls differ: stiff tungsten holds its markers at −0.003 mm/µs,
where soft copper *contracts* at −0.064 mm/µs as the impact shock runs back into
it. The same coupling appears in tension: tungsten's 1500 MPa yield drags along the
stretching jet hard enough to **accelerate its own tail** ~5 % (2248 → 2371 m/s) as
the faster material ahead pulls it, while copper at 200 MPa transmits ~20× less and
flies almost perfectly ballistically (residual 0.001 vs 0.020 mm). Tensile coupling
scales with yield. A real jet is soft copper for exactly this reason.

**Fluid-like erosion is free.** At a 7 km/s tip the stagnation pressure is
~0.5·ρv² ≈ 2×10⁵ MPa, about **1000× copper's yield**, so von Mises return-mapping
caps deviatoric stress near zero on its own. No "fluid" branch exists or is needed
— the hydrodynamic regime is what this material model *already* does when the
pressure dwarfs the strength.

**Particulation does NOT fire in this window — reported, not claimed.** A real jet
eventually tears into a fragment train, and the emergent path for it exists (the
ductile-damage gate). It does not trigger here, and the arithmetic says it
*shouldn't*: the markers stretch F_xx to 2.0, so log strain is ln 2 ≈ 0.69 and
equivalent plastic strain ≈ 0.8 against copper's 1.5 reserve — roughly half way.
Real jets particulate at ~100 µs; this deck runs 25 µs. **A jet that stays
continuous for 25 µs is the correct answer, not a shortfall**, and `damage_threshold`
was **not** lowered to force breakup on cue (that would be confirmation-bias tuning
toward a prettier result — §10). Damage is confined to the leading ~40 mm, which is
**erosion** at the armor, not particulation: the damage front marches *backward*
through the jet as it is consumed (the 20–40 mm band goes 0.062 → 0.995 over the
window) while everything beyond 40 mm reads exactly 0.000.

**Honest limits.**

- **~~No equation of state, and this is where it bites hardest.~~** *Superseded by
  §3.5 (milestone 8): there is an EOS now.* The claim this bullet used to make —
  that jet-tip pressure was the least trustworthy quantity in the model — was
  true, and it was the defect that motivated §3.5. `yield_strength` still caps
  only the **deviatoric** response, but the volumetric response is no longer
  fixed-corotated: it is a Murnaghan EOS, monotone and stiffening. What remains
  untrustworthy at the tip is smaller and different in kind — see §3.5's own
  honest limits.
- **Do not compare this deck's penetration to the uniform stand-in it replaced.**
  That comparison is **energy-confounded twice over**: a graded jet carries far
  less kinetic energy than a uniform-7000 rod of the same mass, *and* copper is
  half tungsten's density (hydrodynamically, depth ≈ L·√(ρ_jet/ρ_target), so
  copper-vs-RHA is √1.14 ≈ 1.07 where tungsten-vs-RHA is √2.24 ≈ 1.50). The
  stretching claim above is deliberately scoped to *kinematics*, which is immune to
  both. The clean, energy-neutral depth experiment is a **standoff** study — same
  jet, same energy, different flight distance before impact — and `standoff` is
  already deck data (`ArmorLayer.standoff`), so it needs no code. Not done.
- **A bounded domain cannot hold the whole event, and grading makes that
  structural rather than incidental.** The tail flies at 2 km/s and needs ~100 µs
  to reach the armor, by which time a 7 km/s tip is ~850 mm downrange. A finite
  field can contain a graded jet's *tip passage* or its *tail transit*, never both.
  This deck claims the tip's passage — where the penetration happens; the trailing
  body is honestly out of frame.

---

### 3.5 An equation of state (milestone 8)

§3.4 shipped with an admission: the volumetric response had **no equation of
state**, and a hypervelocity stagnation point is exactly where one matters most.
This section is that hole being filled, and the measurements that say so.

**What was actually wrong — and a retraction.** A first diagnosis of this defect
called it a *softening branch* that the jet "crushed through". **That was wrong,
and it was a units error:** the Kirchhoff stress `τ = P(F)Fᵀ` (which drives the
P2G scatter) really does peak and collapse toward zero as `J → 0`, but the
stagnation demand `½ρv²` is a **Cauchy** pressure, and `σ = τ/J` is *monotone*
(`dσ_xx/ds = 2µ/s² + 2λs > 0`). Compared in one currency there is no ceiling and
nothing runs away. The true defect was quieter: the law was simply **far too
compressible**. Under a 220 GPa demand the model equilibrated at `J ≈ 0.15` where
real copper gives `≈ 0.61`. The same root cause wore two faces — Kirchhoff
collapsing (dynamics under-resist) while Cauchy `τ/J` diverged (the `stress`
column over-read ~327 GPa at the tip, 1600× copper's yield, which the viewer was
quietly clamping away).

**The law.** Deviatoric and volumetric responses are now split, and the pressure
comes from a **Murnaghan EOS** (Murnaghan 1944 — textbook finite-strain, public):

```
τ  =  dev₂[ 2µ(F−R)Fᵀ ]  −  p(J)·J·I
p(J) =  (K₀/K′) · (J^−K′ − 1)          K₀ = λ+µ,  K′ = 4
K(J) =  −J·dp/dJ = K₀·J^−K′            (tangent bulk modulus)
```

Three properties earn it its place:

- **Monotone and stiffening.** `p → ∞` as `J → 0`, so compression always finds an
  equilibrium. This is the property the old law lacked.
- **Zero new material constants.** `K₀ = λ+µ` already follows from `E`/`ν`; `K′≈4`
  is the standard default for metals and dense solids and lives once, in
  `materials.EOS_KP`. Contrast Mie-Grüneisen, which needs per-material `c₀/s/Γ`.
- **Tangent-matched at rest.** `K(1) = K₀ = λ+µ` is *exactly* the rest stiffness
  of the term it replaces, so the EOS-aware p-wave speed at `J=1` is bit-identical
  to the old `√((λ+2µ)/ρ)`. Milestone 8 is a **large-strain-only** change by
  construction: a 1600 m/s KE deck barely moves, a 7 km/s jet tip moves a lot.

Note the deviator: `2µ(F−R)Fᵀ` is **not** purely deviatoric at finite strain — under
isotropic compression `F = sI` it is `2µ(s−1)s·I`, a pure pressure, which is why
the old rest bulk modulus was `λ+µ` and not `λ`. Now that the EOS owns pressure,
that trace is removed or it would be double-counted. The 2D deviator splits at
`tr/2`, matching `_return_mapping`'s `e_mean = (e1+e2)/2`, so the stress the yield
surface caps is the stress `_p2g` actually scatters. Plastic flow stays isochoric,
so plasticity and the EOS are genuinely orthogonal.

**An independent corroboration of K₀.** `K₀ = λ+µ = 136.4 GPa` is derived from
copper's `E`/`ν`. Public shock data gives `ρ₀c₀² = 139.1 GPa` from the bulk sound
speed — a completely unrelated route. **They agree to 2 %.** Nothing was tuned to
make that happen.

**Measured (`heat_vs_composite`, same probe before and after).**

| quantity | pre-EOS | post-EOS | note |
|---|---|---|---|
| worst live **jet** `J` | 0.0706 | **~0.43** | 93 % → 57 % volume loss; **dt-dependent, see below** |
| worst live **rha** `J` | 0.1747 | **0.50** | target no longer crushed |
| worst live **ceramic** `J` | 0.9912 | **0.9910** | **unmoved — predicted a priori** |

The ceramic row is the falsifiable one, and it is also the only one worth quoting
to four decimals. §3.4 argued that ceramic fails at `J≈0.98` (its brittle threshold
is ~3 GPa, i.e. mild compression), where *any* monotonic volumetric law shares the
same tangent bulk modulus and agrees to <1 % — so no EOS could move ceramic
comminution. It held. The jet tip rides `J≈0.93` in free flight and dives only at
an interface: impact-driven compression that **recovers**, not a permanently
crushed tip.

**⚠️ REWRITTEN BY MILESTONE 11 (§3.9). What this section used to call "the tip
`J`" is the FRONT-PLATE IMPACT TRANSIENT, and comparing it to `J_eq = 0.6056` was
a category error.** The old text quoted a dt table (0.3923 / 0.3971 / 0.4315 at
47 / 98 / 240 substeps) "against an equilibrium `J_eq = 0.6056`", concluded the
gap was an undamped **shock ring**, and named the ring the dominant tip defect.
Milestone 11 built the artificial viscosity that was supposed to fix it, measured
the tip **every substep** (frame-cadence sampling *aliases* the ring — nothing at
frame rate can see it), and found:

| what, measured per substep | AV off |
|---|---|
| front RHA plate impact, t≈0.4 µs | `J` **0.4589** |
| ceramic interface impact, t≈9.8 µs | `J` 0.5704 |
| **steady penetration**, t≈12 µs | `J` **0.6287** |

* The whole-bake "worst live `J`" (the CFL audit's number, and this table's old
  ~0.43) is the **first impact shock**, not the steady stagnation point — and not
  the ceramic interface either, which this section previously claimed owned it.
* `J_eq = 0.6056` is a **steady** stagnation prediction (`p_stag = ½ρv²`). The
  sustained penetration state actually sits at ~0.63 — just *above* it, which is
  the right side, since by t≈12 µs the material arriving is from further back in a
  7000→2000 graded jet and so is slower than the tip. Consistency, **not** a
  precision check: do not quote an agreement percentage, because the arriving
  velocity is not 7000.
* So the ~30 % "gap" was mostly **two different physical states being compared**,
  plus coarse-`dt` error. An impact shock is genuinely more severe than steady
  stagnation — `mpm.py`'s own CFL comment says so.
* **The dt-drift converges on its own by ~400 substeps, with AV off**
  (0.4615 / 0.4595 / 0.4652 at 400 / 800 / 1600 — flat to ±0.6 %). The 47→240
  climb was simply the coarse-`dt` regime; the shipped deck runs at 240, just
  inside it. Those extrema are themselves aliased, so read them only as "the
  frame-level state is dt-converged by ~400".

Quote the tip as **~0.46 at first impact, ~0.63 in steady penetration** — and
never to four decimals: it is a single-particle extremum over ~179 k particles and
150 frames, which wobbles ~1 % run to run (the repo's ≤0.11 % scatter floor is for
*aggregates* and does not license precision here). That was milestone 8's exact
trap and it is easy to fall into twice.

**Honest limits — the two that remain.**

- **Murnaghan is a *cold* curve: no shock heating.** It carries no thermal
  pressure, so it stays too soft, and *how* too soft still depends on velocity.
  Against copper's public shock Hugoniot (`u_s = c₀ + s·u_p`, which does include
  heating) the model reads **0.93×** at `J=0.9` (KE regime — negligible), **0.68×**
  at the jet's 7 km/s equilibrium, and **0.28×** at the ~0.43 tip excursion.
  So milestone 8 shrank a velocity-dependent error, it did **not** remove one:
  across the jet's own 2→7 km/s gradient the spread went from ~1.70× to ~1.37×.
  **Anything that reads absolute pressure, or sweeps velocity, still inherits
  this.** Fixing it properly means Mie-Grüneisen and real per-material `c₀/s/Γ`.
- **~~No artificial (shock) viscosity, so the front rings — and this is now the
  dominant tip defect.~~ RETIRED ON EVIDENCE by milestone 11 (§3.9).** The ring is
  real but it is **~0.9 % peak-to-peak** on `J`, carrying ~8 % of an already tiny
  residual — it cannot explain a ~30 % discrepancy, and §3.5 above explains what
  did. Artificial viscosity is now implemented (von Neumann–Richtmyer) and, **as of
  milestone 13, ships default ON** — not to damp the ring (that trade never made
  sense) but because **AV work is what carries shock heating into `e`** (§3.9's
  banner, §3.10). It was kept as a prerequisite for Mie-Grüneisen, and it was
  needed as one.
- For reference: copper's Hugoniot poles at `J = 1 − 1/s = 0.328`, i.e. real copper
  essentially cannot be compressed past that. The measured tip at ~0.43 sits above
  it — severe, but inside physics, where the old law's 0.0706 was not.
  **⚠️ Re-measured under milestone 13: the jet's worst live `J` is 0.5226**, not
  ~0.43 — MG resists the tip harder, so it clears copper's pole by 59 % rather than
  31 %, and clears its own `J_sw`=0.396 by 32 % (the pole guard does **not** engage
  on the jet). The `~0.43` figures throughout this section are the Murnaghan-era
  measurement and are kept for the comparison the section is making. §3.5's posture
  is unchanged and was right: **do not quote tip-`J` to four decimals** — it is
  dt-dependent, and the number moving again under a new EOS is the fourth
  demonstration of that.

**Cost, and why the substep had to be re-derived.** The EOS *stiffens* under
compression, so the rest-state sound speed is no longer the CFL bound:
`c(J) = √((K₀·J^−K′ + µ)/ρ)` climbs as `J^(−K′/2)`. `bake` now sizes `dt` from the
compression the deck's own stagnation pressure predicts, with `EOS_CFL_J_MARGIN`
of headroom for the ring above — and then **measures**, every frame, the sound
speed actually reached, warning if the margin was breached.

> **⚠️ MILESTONE 14 REPLACED THAT BOUND, AND EVERY `EOS_CFL_J_MARGIN` NUMBER IN
> §3.5/§3.6/§3.9/§3.10 IS FROM THE SUPERSEDED FORMULA.** The margin multiplied `J`
> — a volume *ratio* — which is violently nonlinear and put the design state past
> every material's MG pole on **all 30 decks**, sizing `dt` from the guard's
> extrapolated backstop. The scale was wrong too: `½ρv²` is *steady stagnation*,
> not the *contact shock* the substep has to survive. Both are fixed in **§3.11**,
> which is the section to read; the tables below are kept as the record of how the
> old constant was cut, not as guidance.

That audit is not decoration; it caught two things a clean-looking bake hid.
**Margin 0.8** survived `heat_vs_composite` only because the deck's ceramic
(stiffer, so a higher design sound speed) donated global headroom the copper tip
borrowed — a jet into plain RHA has no such donor. **Margin 0.55** covered the jet
(ratio 0.648) but let `apfsds_vs_nera` breach by **2.41×**: it validated clean,
produced no NaN, and was wrong. The binding case is not the hypervelocity jet at
all — see §3.6. The honest value is **0.35**, at which every deck passes (79 % of
budget used on the deck that binds, 27–57 % elsewhere). Cost: `heat_vs_composite`
goes 18 → 245 substeps/frame, and the whole 20-deck set bakes in ~20 minutes.
Irrelevant for an offline solver (root §1); a bake that validates clean and is
quietly wrong is not.

**A bug the old law could not have had.** A divergent EOS makes degeneracies
dangerous rather than merely wrong. With a raw negative `J` from a momentarily
inverted element, `−p(J)·J` **flips sign** and reports colossal *tension* — which
`_stress_invariants` feeds straight to the brittle tensile-fracture trigger,
shattering ceramic for a purely numerical reason. The old decaying law couldn't
produce this. Hence `mpm.J_FLOOR`: one shared floor (0.05) so the Warp and NumPy
paths floor identically, positioned as a **degeneracy backstop, not a physical
limit** — it is ~25 000× beyond what a 7 km/s stagnation point demands, the
measured worst live `J` is 8× above it, and `bake` warns if live material ever
reaches it. `tests/test_stress_paths.py` pins both paths together.

### 3.6 What the EOS did to the NERA filler (an unwelcome finding)

> **Written at milestone 8, and describing a `nera_filler` that no longer exists.**
> Kept because its *diagnosis of the mechanism* was right and is what motivated the
> fix. Its magnitude claim was wrong (§3.6.1) and its cause is now removed (§3.6.2).
> Everything below is in the **past tense as of M12**; the tense in the original has
> been left alone so the record reads as it was.

The deck that binds the substep is not the 7 km/s jet. It is **`apfsds_vs_nera`**,
and the reason is a real interaction rather than a numerical nuisance. *(Still true
after M12: it still binds, at ratio 0.442 vs the jet's 0.713 — but for a reason
§3.6.1 restates, and the margin stays 0.35.)*

`nera_filler` **was** `reactive=True` with `ignition_compression=0`, which meant
`mpm.py` skipped **both** the return mapping and the ductile-spall gate for it, and
it never ignited. So it could neither yield, nor break, nor self-vent — §3.3 and
`materials.py` said this outright ("it can neither yield nor break, so it stores
elastic energy without dissipating it — stiffer-than-real"). It is soft
(`K₀ ≈ 8.9 GPa`), and it had no dissipation path and nowhere to go. **M12 gave it
one** (§3.6.2): it is now non-reactive and ductile, and both fields are live. The
paragraph below diagnoses why that mattered — and §3.6.1 corrects how much.

Under the pre-EOS law that was *harmless*: the rod squeezed it, `λ(J−1)J` decayed
toward zero, and it went limp. Under a stiffening EOS the identical situation
gives a **50 000 mm/ms sound speed**:

| `EOS_CFL_J_MARGIN` | substeps/frame | worst live `J` | audit |
|---|---|---|---|
| 0.55 | 45 | 0.1942 | **BREACH, 2.41×** |
| 0.35 | 110 | 0.2159 | OK, 79 % of budget |
| 0.20 | 336 | 0.2120 | OK, 27 % of budget |

*(All three rows are **pre-M12**, on the no-dissipation filler. Post-M12 the shipped
0.35 row reads `J = 0.2421` at **63 %** of budget; the 0.55 and 0.20 rows were not
re-measured. The margin **stays at 0.35** — see `mpm.EOS_CFL_J_MARGIN` for why the
~23 % substep saving M12 arithmetically permits is deliberately not taken.)*

> **⚠️ THE PARAGRAPH THAT STOOD HERE WAS WRONG, AND MILESTONE 12 MEASURED IT WRONG.**
> It read: *"That `J ≈ 0.21` is real, not the instability eating itself. Shrinking
> `dt` by 3× moves it 1.8 %, so it is converged: **the filler genuinely reaches
> ~79 % volume loss**."* The `J` value is real and reproduces exactly. **The
> sentence built on it is not: the filler's bulk is never meaningfully compressed
> at all.** See §3.6.1 — the rest of this section's *mechanism* survives, its
> *magnitude* and *location* do not.

Its predicted-vs-reached ratio is **0.394**, far worse than the copper jet's
**0.713** — which is why it, not the jet, sets `EOS_CFL_J_MARGIN`. That the
filler's ratio is *stable* under refinement while the jet's drifts (0.648 → 0.713
as substeps go 47 → 240, because the jet's is a shock-ring artifact) is what makes
a single margin trustworthy: the binding number is the one that does not move.

The honest reading: this is **not a new defect the EOS introduced**, it is an old
one the EOS made *visible*. A material with no dissipation path was always going
to be squeezed arbitrarily far; the old law just hid it by going soft at exactly
the moment it should have resisted.

### 3.6.1 What `J = 0.2159` actually is (milestone 12)

**It is 27 particles out of 36 966, and they are not in the interlayer.**

The audit's `worst live J` is a **min over every live particle over every frame** —
a single-particle extremum, the least trustworthy metric class in this repo by its
own lessons (§3.9: *"a min-over-a-set traces the envelope … trace a SINGLE
particle"*). §3.6 above read that extremum as a **bulk** statement. Traced properly
(`_trace_j` on material 5, every substep, whole event, unmodified solver):

| quantity | measured |
|---|---|
| worst live filler `J` | **0.2159** (reproduces the pre-M12 docs exactly) |
| **mean** live filler `J` at that same instant | **1.0105** |
| **minimum the mean ever reaches**, whole event | **0.9495** |
| **median** live filler `J` at the worst frame | **0.9932** |
| particles below `J=0.5` at the worst frame | **25 / 36 966 = 0.068 %** |
| particles below `J=0.3` | **3 = 0.008 %** |

The bulk filler loses ~5 % of its volume at worst. There is no 79 % volume loss.

**And the crushed particles are 34 mm downrange of the interlayer.** The filler
seeds at `x = 156.1–167.9`; the sub-0.5 particles sit at `x = 200.8–202.0` — inside
the **main plate's** crater (`199.1–228.9`), pinned there by the rod tip (leading
edge `201.87` at that frame). 68.95 % of the filler is still behind the back plate,
median `x = 165.0`. So this was never the interlayer being squeezed in the sandwich;
it is filler debris dragged across the standoff gap and caught in a
**tungsten-rod-vs-RHA vise**.

**Why the convergence check passed and still misled.** 0.2159 at 110 substeps vs
0.2120 at 336 is the *same handful of trapped particles* in both. Refining `dt`
cannot dissolve a geometric trap — the axis was wrong. This is exactly the failure
§3.8 catalogued for the jet (*"cells across the jet is the controlling parameter"*),
arrived at from the opposite direction, and it is the fourth time this repo has been
bitten by a convergence claim that measured the wrong thing.

**What survives, and it is the part that mattered.** The *mechanism* §3.6 names — a
material with no dissipation path gets squeezed arbitrarily far — is correct. It
just applies to ~0.07 % of the filler in a vise, not to the interlayer in bulk. That
mechanism is real enough to have set `EOS_CFL_J_MARGIN` for the whole repo, and
milestone 12 fixed its cause.

### 3.6.2 The fix, and the bonus that did not arrive (milestone 12)

`nera_filler` was `reactive=True` with `ignition_compression=0` — a filler that
never ignites. That flag exists to run the ERA state machine, but `mpm.py` **also**
uses it to gate out `_return_mapping` and `_update_damage`, and the stated reason
for those gates is that a filler *"must not spall before it detonates."* **That
reason cannot apply to a filler that never detonates.** `nera_filler` inherited a
gate written for its igniting twin, and the price was no dissipation path at all.

The fix is what `apfsds_vs_nera.yaml`'s own header already prescribed — *"a
NON-reactive filler with a high damage_threshold"* — and it cost **zero kernel
code**, because both gates key off `reactive > 0.5`. `yield_strength` (50 MPa,
unchanged) and `damage_threshold` (0.02 → 3.0, representative elastomer
elongation-to-failure) are now **live fields**; they were dead. Nothing physical was
lost: `_update_reactive` was a verified no-op here (`ic=0`, `burn=0`, `damage=0` →
every branch falls through) and `_p2g` takes the identical elastic term for an
unignited particle. One real difference, reported rather than waved away:
`_clamp_reactive_v` no longer caps this filler, and on the pre-M12 bake that clamp
bound on **exactly one particle across frames 158–159 of 550**.

**Measured.**

| arm | `dthr` | worst live `J` | CFL budget | filler spall | coherent |
|---|---|---|---|---|---|
| pre-M12 (no dissipation) | — | 0.2159 | 79 % | 0.00 % | 68.3 % |
| **M12 (shipped)** | **3.0** | **0.2421** | **63 %** | **18.65 %** | **66.0 %** |
| control: plasticity only | ∞ | 0.2903 | 44 % | 0.00 % | 77.7 % |
| `era_filler_inert` | 0.02 | 0.6813 | 17 % | 69.59 % | 30.4 % |

**Cohesion holds — the claim the fix had to not break.** Against the shredding twin
(one field apart), NERA spalls **18.65 % vs 69.59 %** and keeps **66.0 % vs 30.4 %**
of its filler coherent, sitting beside the pre-M12 arm (68.3 %), nowhere near
era_inert. The bulge change is attributable to **plasticity, not spall**: the
plasticity-only control gives near-identical plate separation (53.8 vs 53.6 mm
beside the channel) with 0 % spall instead of 18.65 %.

> **Read that as "more cohesive than the shredding twin, therefore ship-safe" — NOT
> as "the bulge is preserved", which is NOT shown.** The distinction is the evidence
> each rests on:
> - **Spall %** is measured over **all** filler particles and is the claim's spine.
>   It is the one figure here that is *not* live-set-confounded, and it is what
>   establishes that the arm did not collapse into `era_filler_inert`.
> - **Coherent % and x-extent** are computed over **live** particles, so they inherit
>   the same selection effect that hands era_inert its flattering 0.6813 worst-`J`.
>   Directionally right, quantitatively soft.
> - **The separation figures are from a probe that reads exactly 18.000 mm at `t=0`
>   but cannot reproduce §3.3's published 16.1/21.1 even on the PRE-M12 bake** (it
>   gets 14.1/13.3 — opposite sign beside the channel). The plasticity-vs-spall
>   *attribution* is sound because it compares two arms through the *same* probe; the
>   absolute millimetres are not, and M12 also flipped the baseline's own separation
>   behaviour (13.3 → 53.6). **The bulge GEOMETRY is not re-established, and
>   reconciling or rebuilding that metric is genuine documentation debt** — it is the
>   first thing to fix for anyone revisiting the NERA arm.

**A tidy corroboration, and a diagnosis of what was wrong before.** `_clamp_reactive_v`
no longer caps this filler (it is not reactive), so a velocity runaway was the thing to
check. There is none — and the numbers say why the clamp was needed at all. Pre-M12 the
filler reached the **full 3000 mm/ms clamp**, nearly **2× the 1600 mm/ms rod driving
it**: a solid that cannot yield stores the shock and springs. Post-M12 it peaks at
**1586 mm/ms** — 53 % of the removed clamp, just under the rod's own speed. **A filler
that dissipates does not out-run the thing hitting it.** The fix removed the clamp's
*reason*, not merely its effect.

**`era_filler_inert`'s lovely 0.6813 is a trap, not a target.** Its filler is not
uncrushed — the crushed particles spall instantly and leave the *live* set.
`worst live J` is **not comparable across arms with different `damage_threshold`**:
it measures different populations. Lowering `dthr` to buy CFL headroom is buying it
with cohesion, and it would be tuning toward the answer (§10).

**The CFL bonus did not arrive, and here is why it never could.** M12's ratio is
`0.2421 / 0.5480 = 0.442`, still below the jet's 0.713, so **`apfsds_vs_nera` still
binds `EOS_CFL_J_MARGIN` and the margin stays at 0.35.** No substep saving. The
mechanism is measurable rather than arguable: at the worst frame the sub-0.3
particles carry **equivalent plastic strain `alpha = 2.91` against a 3.0 reserve** —
**97 %**, and 4.5× the bulk's 0.65. They are not failing to yield; they are
**saturating** the yield surface and are still crushed. Plastic flow is **isochoric**
(§3.5), so it cannot relieve volumetric confinement no matter how hard it engages.
That orthogonality is not a tuning problem, and it predicts the counterintuitive
ordering above: spall at `dthr=3.0` makes worst-`J` *worse* than never spalling
(0.2421 vs 0.2903), because a spalled particle drops its stress term in `_p2g` and
stops resisting, concentrating the crush on its live neighbours.

**So: relieving this needs a VOLUMETRIC criterion (compaction/pore collapse), not a
deviatoric one.** That is a separate milestone and it is not done. Two consequences
worth stating plainly:

- Any conclusion resting on the NERA filler's stiffness — notably the cohesive-bulge
  A/B — was confounded before, and the confound is **now removed**: the A/B is
  single-variable (`dthr` 3.0 vs 0.02, everything else equal), pinned by
  `tests/test_nera_dissipation.py`.
- **The stated reason for doing this before Mie-Grüneisen was not achieved.** M12
  was sequenced first so MG would land on a solver where every material stays inside
  its Hugoniot's valid range. `nera_filler`'s pole sits at `J = 1 − 1/s ≈ 0.5`, and
  M12 leaves the worst live `J` at **0.2421** — still far past it. **A pole guard
  stays load-bearing on this deck under milestone 13**, and it must be designed as
  such rather than treated as a formality.

**Honest limit on the fix itself.** A real elastomer dissipates *viscoelastically*;
von Mises plastic flow is the dissipation path this solver has. It is the right
*kind* of thing — irreversible, isochoric, cohesion-preserving — rather than the
right constitutive model. Plausible, not predictive (§1).

---

### 3.7 Velocity sweep vs the hydrodynamic asymptote (milestone 9)

The first experiment that **varies impact velocity**, and the first whose claim is
a *trend* rather than a state. Ten decks, one factorial: `{tungsten_rod,
copper_jet}` × `{1500, 2500, 3500, 5000, 7000}` m/s into an identical 120 mm
semi-infinite RHA half-space, everything else held fixed (`sweep_*.yaml`).

**What is predicted, and why it is falsifiable.** Ideal hydrodynamic
(Tate–Alekseevskii) penetration is a pressure balance,
`½ρ_p(v−u)² = ½ρ_t·u²`, so as strength becomes negligible the penetration velocity
`u` approaches a ratio fixed by **density alone**:

```
u/v  →  1 / (1 + √(ρ_t/ρ_p))        tungsten 0.5996     copper 0.5165
```

Two arms, two *different* a-priori numbers, one physics. A single arm approaching
a single number could be coincidence; two arms approaching two different numbers
computed beforehand from density could not. And the bound has a direction: strength
can only hold `u` **below** the ideal limit, never past it — so exceeding the
asymptote is not "inaccurate", it is impossible.

> **⚠️ RE-MEASURED 2026-07-17 (milestone 13). The claim survived and got BETTER —
> the tables below are the Murnaghan-era measurement, kept for the comparison.**
> Under Mie-Grüneisen (§3.10) + the §1.1.1 boundary fix, at v=7000:
>
> | | M9 (Murnaghan) | **M13 (MG)** |
> |---|---|---|
> | tungsten, fraction of its own asymptote | 0.937× | **0.9609×** |
> | copper, fraction of its own asymptote | 0.937× | **0.9622×** |
> | measured ratio vs the 1.1608 density prediction | 1.1614 (+0.04 %) | **1.1593 (−0.13 %)** |
>
> **MG moved BOTH arms closer to the hydrodynamic asymptote** (0.937 → 0.961),
> which is exactly the direction a stiffer, Hugoniot-calibrated EOS should move
> them — strength holds `u` below the ideal limit, and a better-resisting EOS
> approaches it. The two arms still land within **0.14 %** of *each other*, so the
> shortfall is still the model's rather than the material's, and still cancels in
> the ratio. The ratio agreement loosened 0.04 % → 0.13 %; **do not read that as a
> regression** — 0.04 % was always finer than the metric deserves (`u/v` is not
> dt-converged; see the caveats below), and both figures are far inside it.
>
> `sweep_tungsten_v1500` still reads **R²=0.9855, steady=False** — the deck M9
> excluded, reproducing its 0.985 exactly. That is Tate deceleration, i.e. physics,
> and it is still correctly refused rather than re-tuned.

**Measured** (`tools/measure_penetration.py`, which identifies the penetrator as
whatever is moving at t=0, measures `v` from frame 0 rather than being told it, and
derives its fit window from the erosion curve):

| v (m/s) | tungsten `u/v` | vs asym | copper `u/v` | vs asym |
|---|---|---|---|---|
| 1500 | 0.4040 | 0.674× *(not steady — see below)* | 0.3819 | 0.739× |
| 2500 | 0.5457 | 0.910× | 0.4484 | 0.868× |
| 3500 | 0.5576 | 0.930× | 0.4649 | 0.900× |
| 5000 | 0.5613 | 0.936× | 0.4745 | 0.919× |
| 7000 | 0.5620 | **0.937×** | 0.4839 | **0.937×** |

**Both arms rise monotonically toward their own asymptote and neither crosses it.**
The sharp test is the **ratio** of the two arms, because a shortfall common to both
cancels there — and the ratio *converges* on the density prediction as strength
becomes negligible, which is precisely what Tate says should happen:

| v (m/s) | tungsten/asym | copper/asym | measured ratio | vs 1.1609 predicted |
|---|---|---|---|---|
| 2500 | 0.910 | 0.868 | 1.2170 | +4.83 % |
| 3500 | 0.930 | 0.900 | 1.1994 | +3.32 % |
| 5000 | 0.936 | 0.919 | 1.1829 | +1.90 % |
| 7000 | 0.937 | 0.937 | **1.1614** | **+0.04 %** |

The convergence is the claim; **0.04 % is the 7 km/s value, not a flat property of
the model.** Note the two fractional shortfalls only *coincide* at 7 km/s — at
2500 they are 0.910 vs 0.868, plainly material-dependent, because that is where
strength still bites and the two arms' yields differ 7.5×. What survives at the top
of the range is a shortfall the arms share — chiefly the cold-curve pressure error
of §3.5, plus the grid resolution of §3.8; residual **strength** is what makes the
low-v end differ, so it is the thing that cancels *last*, not something that
cancels at all. (This sentence used to list "the undamped shock ring" as a
component. §3.9 measured that ring at **~0.9 % peak-to-peak** and it is not a
plausible contributor at this size — the attribution, not the trend, was wrong.)

**The trend is not a timestep artifact — controlled, not argued.** The production
decks are CFL-sized, and the EOS-aware bound scales with stagnation pressure ~`v²`,
so substeps *rise* with velocity (copper 75→163, tungsten 81→218). Finer `dt` means
less numerical dissipation means higher `u/v`, so the fast arm was getting more
physics *and* less dissipation — the measured rise was physics + artifact in unknown
proportion. Rebaking all ten at a fixed 250 substeps/frame moves each point by only
~1–2 % and leaves the shape and the ratio intact (the ratio above **is** the
uniform-dt number). Uniform `dt` is the right *control* choice and the wrong
*production* one — a 7 km/s deck genuinely needs a smaller step — so the committed
decks stay CFL-sized.

**Did this need the EOS?** Yes, and it is worth being precise, because `u/v` was
partly chosen for being a pressure *balance* in which the EOS error largely cancels
between the two sides. Baking the copper arm on the pre-EOS solver at matched `dt`:

| v | pre-EOS law | with the EOS |
|---|---|---|
| 1500 | 0.3822 (0.740×) | 0.3819 (0.739×) |
| 7000 | **0.5333 (1.032×)** | 0.4839 (0.937×) |

**The pre-EOS law exceeds the hydrodynamic ceiling at 7 km/s.** Not merely
inaccurate — 1.032× is on the wrong side of a bound that strength cannot push past.
The error is velocity-dependent exactly as §3.5 says: ~0 % at 1500, −9 % at 7000.
(The old law's *better-looking* 0.5000 at its own coarse CFL was simply
unconverged; 25× finer `dt` moves it to 0.5333, i.e. coarse-step numerical
dissipation was masking the defect. Nor is the new law perfectly converged —
163→500 substeps moves copper@7000 by +2.6 % — so **read the trend, not the third
decimal**.)

**Honest limits.**

- **`u` is the erosion-front velocity, so this is not assumption-free.** Erosion
  has a partly-numerical component (§3.4: crushed particles latch `damage` and
  leave the live set). The defence is that this is *systematic* and cancels in the
  trend and in the cross-arm ratio — not that the metric is clean. Depth is
  deliberately not reported: it is cumulative and rides on the whole erosion
  history rather than an instantaneous rate.
- **tungsten@1500 is flagged NOT STEADY (R²=0.985) and excluded from the trend.**
  That is physics, not a probe failure: its tip advances 0.86 mm/µs early and
  0.25 mm/µs late — it **decelerates**, which is Tate's rod deceleration under
  target resistance. Tungsten's yield is 7.5× copper's, which is exactly why
  copper@1500 stays straight (R²=0.9999) and tungsten does not. There is no single
  steady `u` there to report, so the fitted slope is a time-average of a varying
  rate and is not comparable to the rest of the column.
- **The asymptote itself assumes incompressible Bernoulli.** Real materials
  compress at the interface, so `√(ρ_p/ρ_t)` from *initial* densities is the
  textbook idealisation the sweep is compared *against*, not ground truth.
  Agreement to 0.04 % in the ratio is better than this model deserves in absolute
  terms and should be read as the density scaling being right, not as validation
  (root §1, §10 — plausible, not predictive).

---

### 3.8 Standoff — the jet's energy-neutral depth experiment (milestone 10)

**Read this first: the shipped decks under-read the effect they measure, by ~1.7×
on the excess, and they are not grid-converged.** `standoff_s00/s30/s60/s90` measure
a depth ratio of **1.31** between S=90 and S=0 where the a-priori prediction is
**1.536**. The cause is resolution, not physics: the jet is 3 mm across = **8 cells**
at the shipped `dx=0.375`, and it *thins as it stretches* to ~1.1 mm ≈ **3 cells** by
the end of the window. The quantitative claim below rests on the six
`standoff_conv_*` decks, **not** on the four shipped ones. Same posture as §3.5's
tip-`J`: quote the trend, never the value.

> **⚠️ RE-MEASURED 2026-07-17 (milestone 13); the tables below are the
> Murnaghan-era measurement.** Under Mie-Grüneisen (§3.10) + the §1.1.1 boundary
> fix the shipped family reads **S90/S0 = 1.312** (was 1.229) against the unchanged
> a-priori **1.536** — so the under-read on the *excess* improved from **~2.3× to
> ~1.7×** (0.536 predicted vs 0.312 measured). MG resists the jet tip harder, which
> is the same direction §3.7's sweep moved.
> **This does not rescue the shipped decks and must not be read as convergence:**
> the deficit is `cells across the jet`, and no EOS can add resolution. The
> `standoff_conv_*` decks still carry the quantitative claim. The matched-fraction
> trend (17.5 → 23.1 mm at f=0.15, 37.6 → 48.9 at f=0.30) and the lab-time trap
> below (94.2 → 72.5 mm, *falling* with standoff) both reproduce unchanged in
> shape.

**Why this experiment exists.** §3.4 built the jet but deliberately **refused to
compare its penetration depth** to anything, because every comparison available was
energy-confounded twice (a graded jet carries less KE than a uniform one; copper is
half tungsten's density). Standoff is the energy-neutral version — the same jet, the
same energy, the same everything; only the flight distance before impact differs. It
needed **no new solver code**: `ArmorLayer.standoff` has existed since milestone 1,
and `mpm._seed` places the armor face at mid-domain with the tip 3 cells in front of
it, then adds each layer's standoff *before* placing it. So standoff on layer 0
pushes the target back while the tip stays put — exactly free flight, nothing else.

**What is predicted, and why it is falsifiable.** Each jet element flies at its own
constant speed, so the jet extrapolates back to a **virtual origin** where all
elements coincide:

```
Z0 = L · v_tip / (v_tip − v_tail) = 120 · 7000/5000 = 168 mm   behind the tip
```

The seeded jet therefore already carries 168 mm of *built-in* virtual standoff, and
the deck's `S` adds to it: `Z = 168 + S`. Let `v` be the velocity of the element
currently at the crater bottom. It has flown `Z + P`, and the crater deepens at
`u(v)`:

```
v·t = Z + P(t),   dP/dt = u(v)
 ⇒  v'·t = u(v) − v  ⇒  dt/t = dv/(u(v) − v)  ⇒  t = t₀ · G(v)
 ⇒  P = v·t − Z = Z · [ v·G(v)/V₀ − 1 ]
```

**`P` is proportional to `Z` at matched `v`.** Three properties make this a sharp
test rather than a rising trend:

- It never assumed ideal hydrodynamics. `G` is a function of `v` alone for **any**
  `u(v)`, Tate-with-strength included, because the strength correction is identical
  across the family when compared at the same arriving element. **The linearity is
  structural, not an artifact of a strengthless limit** — so a measured
  non-proportionality cannot be excused as "strength".
- **Both the slope and the intercept are predicted a priori** from the seeded
  velocity gradient, with nothing fitted. Fit measured depth against `S` and
  intercept/slope must come out at 168 mm.
- It is **diameter-independent** — which is what makes the resolution study below
  possible.

Matched `v` == matched material element == **matched consumed fraction**, which is
what `tools/measure_standoff.py` matches on.

**The metric is the whole experiment — the obvious ones lie.** Depth at a fixed
**lab time** is an artifact of the *opposite sign*, and it fires hard here: depth at
the end of the window **falls** 105.0 → 80.1 mm as standoff rises, purely because a
longer standoff impacts later and penetrates for less of the window. Anyone
measuring the obvious quantity would publish "standoff reduces penetration". Depth
at a fixed **post-impact time** is honest but collapses the predicted spread to ~8 %.
Only matching on consumed fraction measures the standoff effect itself — and it needs
just a leading slice of the jet consumed, not the whole thing, which is what makes it
affordable (the tail flies at 2 km/s and never arrives — §3.4).

**Measured** (`python tools/measure_standoff.py --family`), depth in mm at matched
consumed fraction, into a 150 mm RHA half-space — *not* the composite stack, which
perforates and would put a **ceiling** on the one quantity being measured:

| S (mm) | Z = 168+S | f=0.15 | f=0.20 | f=0.25 | f=0.30 | ratio vs S=0 | predicted Z/Z₀ |
|---|---|---|---|---|---|---|---|
| 0  | 168 | 20.7 | 28.2 | 36.3 | 44.9 | 1.000 | 1.000 |
| 30 | 198 | 22.7 | 31.1 | 39.9 | 49.3 | 1.099 | 1.179 |
| 60 | 228 | 24.1 | 33.2 | 42.7 | 52.6 | 1.172 | 1.357 |
| 90 | 258 | 25.1 | 34.9 | 44.8 | 55.4 | **1.229** | **1.536** |

Monotone, right sign, ~200× the 0.11 % run-to-run floor — and **short by more than
half the predicted excess**. The premise test says why: at matched arriving-element
velocity the sim's `u` is **not** a function of `v` alone — it falls from 2272 m/s
(S=0) to 1960 (S=90) at v=5000, −14 %. Since `P ∝ Z` holds for any `u(v)`, that is
where the proportionality is lost, and it is a real defect to be explained rather
than a discrepancy to be shrugged at.

**Three candidate causes, discriminated rather than guessed:**

- **Particulation — ruled out.** Free-flight damage at S=90, the *most*-stretched
  deck, is exactly **0.0000**. The 0.20 reading at S=0 runs the **wrong way** for
  particulation (it should grow with stretch, not shrink); it is back-splashed crater
  ejecta drifting upstream past the sampling band. Consistent with §3.4's arithmetic:
  particulation should not fire in-window, and it does not.
- **Under-resolution — the dominant cause.** The jet thins to ~2.9 cells at shipped
  `dx`. (The thinning itself is a good sign: it tracks constant-volume `3/(1+t/24)`
  to a few %, so MPM is handling the free lateral surface correctly rather than
  dilating the jet.)
- **A real finite-diameter effect — mostly excluded**, by the decisive test. The
  derivation is **diameter-independent**, so a 6 mm jet at the shipped `dx=0.375` has
  **16 cells across — exactly like a 3 mm jet at `dx=0.1875`**. Two routes to the same
  control variable, sharing neither grid nor jet:

| configuration | cells across jet | mean S90/S0 ratio | vs predicted 1.536 |
|---|---|---|---|
| 3 mm jet, `dx=0.375` — **shipped** | 8 | **1.229** | −20 % |
| 3 mm jet, `dx=0.250` | 12 | 1.383 | −10 % |
| 3 mm jet, `dx=0.1875` | 16 | 1.429 | −7 % |
| **6 mm jet, `dx=0.375`** | **16** | **1.501** | **−2.3 %** |

`python tools/measure_standoff.py --convergence`. **Cells across the jet is the
controlling parameter**, reached by refining the grid or by fattening the jet, and
the shortfall is numerical. At 16 cells the fat jet sits within the per-fraction
scatter (±3.5 %) of the a-priori 1.536 — **consistent with the prediction, which is
not the same as confirming it**, and the two 16-cell routes still differ by 5 %, so
cells-across is dominant but not the only term.

**Honest limits.**

- **Do not quote a Richardson extrapolation from these four points.** The observed
  order is ill-conditioned — it swings from ~0.7 to ~5 depending on the matching
  point, and the extrapolated value with it. Adding a fifth grid would not fix that
  (the conditioning is the small high-resolution increments, not the count). The
  monotone trend plus the independent fat-jet route is the honest statement, and
  **"converges toward" is not "converged"**.
- **A trap worth naming: the under-resolved curve looks like the textbook result.**
  At the shipped `dx` the increments saturate (4.7 → 3.4 → 3.1 mm), which reads as a
  curve bending toward a standoff *optimum*. It is a grid artifact. A real optimum
  requires particulation and dispersal at long standoff, which this jet does not do
  (§3.4) — so this family measures **the rising limb only**, and no optimum has been
  manufactured by lowering `damage_threshold` to force breakup (root §10).
- **Wall reflections are not common-mode here.** The slabs span the full height
  against slip walls (§1.1), and at matched consumed fraction the elapsed-from-impact
  time scales with `Z`, so each deck sees a different number of reflections — a
  per-deck systematic the refinement study does not remove. It is second-order
  against a 25 % effect, but it is not zero.
- **This retroactively vindicates §3.4's refusal to compare the jet's depth**, for a
  second and independent reason. At production `dx` the 3 mm jet is 8 cells across and
  thins to ~3, so *any* depth claim about `heat_vs_composite` is grid-limited. §3.4's
  *kinematic* claims (stretch rate, +0.1 %) are untouched: they are measured
  Lagrangianly on free-flight markers, which do not lean on grid coupling. Scoping the
  milestone-7 claim to kinematics is what made it survive.
- **The shipped family stays at `dx=0.375`** because the same four decks at
  `dx=0.1875` would cost ~44 GB of cache against ~11 GB. The convergence decks carry
  the physics; the shipped decks are for playback and for the *shape* of the trend.

---

### 3.9 Artificial (shock) viscosity — and the defect that wasn't (milestone 11)

Milestone 8 left a named, documented defect: *"no artificial viscosity, so the
front rings — and this is now the dominant tip defect."* Milestone 11 built the
standard fix, measured it, and **retired the diagnosis on evidence**. The
deliverable is the measurement, not the feature — the same shape as §3.4 (the SPH
hedge retired) and §3.8 (the headline is the limit).

> **⚠️ SUPERSEDED BY MILESTONE 13: AV IS NOW ON BY DEFAULT** (`av_c_q = 1.5`,
> `av_c_l = 0.6` in `config.py`). **Everything below that argues "default off" is
> milestone 11's reasoning and is kept because the reasoning was correct at the
> time — but do not act on it.**
>
> M11 weighed AV's cost (+57 % substeps) against damping a **~0.9 %** ring and
> concluded, correctly, that this was a bad trade. **It was the wrong question.**
> §3.10 shows AV's real job was never the ring: **AV work is the mechanism that
> feeds shock heating into `e`**, and without it Mie-Grüneisen's energy equation
> lands on the *isentrope* instead of the Hugoniot (`p/p_H` 1.000 → 0.923). The
> velocity-error spread across the piston goes 0.223 → 0.003 with it on.
>
> So M11's own closing sentence — *"AV is the prerequisite for Mie-Grüneisen: its
> work is currently dissipated to NOTHING, and the moment a thermal term lands, AV
> heating SHOULD raise thermal pressure"* — is exactly what happened. **The reason
> AV was off no longer exists**, and the +57 % substeps is now the price of a
> correct energy balance rather than of a 0.9 % cosmetic gain.
>
> **Stale twice:** M11 anticipated switching AV on *"for the jet without re-tuning
> the KE decks"*. M13 ships it on for **all 30**, so that hedge is spent. And M11's
> *"AV is inert below hypervelocity — `apfsds_vs_rha` moves ≤0.20 % at matched dt"*
> was measured **under Murnaghan at matched dt**; M13's `apfsds_vs_rha` spall moved
> **18.2 % → 25.1 %** under MG + AV-on + the §1.1.1 boundary fix. That is three
> variables at once and **does not isolate AV** — but it does mean the ≤0.20 %
> figure must not be quoted as evidence that AV is inert on KE decks *today*. It
> was true of the thing it measured.

**The law.** von Neumann–Richtmyer: a bulk pressure resisting compression *rate*,

> `q = ρ₀·l·(c_q·l·(div v)² − c_l·c·div v)` for `div v < 0`, else `0`

added in `_p2g` only — never in `_fixed_corotated_pft`. That placement is
load-bearing and pinned by a test: the constitutive law feeds the brittle
fracture triggers and is mirrored on the host by `_von_mises`, which sees only
`F` and *cannot* know `div v`. Letting a numerical term reach those would shatter
ceramic for a numerical reason and break the two-path pin. Consequence, stated
because it is real: the cache's `stress` column **excludes** `q`. `div v` is
`trace(C)` — APIC's affine matrix already *is* the velocity gradient, so this
needs no new array and no extra transfer. Coefficients are deck fields
(`SolverParams.av_c_q` / `av_c_l`), default `0.0`.

**Why frame-cadence metrics could not answer the question — the real blocker.** A
grid-scale (~2dx) oscillation has period `~2dx/c`; post-shock RHA at `J≈0.73` has
`c≈9800 mm/ms`, giving **~159 substeps**, while frames are 400–1600 substeps
apart. Every frame-sampled metric therefore lands at effectively random phase.
That is why "worst live `J`" went *non-monotone* across a matched refinement (the
AV-on arm ended up *below* the AV-off arm at the finest `dt` after being above it
at two coarser ones) — aliasing, not physics. The CFL audit has the same blind
spot and says so. The measurement that works is a **per-substep trace of a single
particle** followed through the shock (`mpm.bake(..., j_trace=...)`): a
min-over-a-*set* reduction traces the envelope of out-of-phase oscillations and
can *hide* a ring; one particle cannot. Particle indices persist (contract §4), so
an index is a durable material label — the §3.4 Lagrangian trick again.

**The ring exists, and it is ~1 %.** RHA particle on the axis 3 mm behind the
front face, every substep, matched `dt`:

| | shock arrives | `J` min | `J` end | tail p2p | power in the ~159-substep band |
|---|---|---|---|---|---|
| AV off | substep 693 | 0.7148 | 0.7319 | 0.0063 | **8.2 %** |
| AV on | substep 636 | 0.7400 | 0.7460 | 0.0054 | **3.2 %** |

The ring is **~0.9 % peak-to-peak** on `J≈0.73` and carries ~8 % of an already
tiny residual; the dominant periods are the window length, i.e. detrending
residue. AV cuts the ring band's share 8.2 % → 3.2 % but total amplitude only
~7 %. **A 1 % oscillation cannot explain the ~30 % discrepancy it was blamed for**
— §3.5 explains what did.

**What AV costs, measured.**
- **+57 % substeps** on the jet deck (240 → 377): AV raises the signal speed the
  CFL bound is sized from. The bound and the per-frame audit both account for it
  now (`_av_signal_speed`); a bound left on the bare EOS `c` would under-report
  exactly at the shock front and print "OK" on a breached bake.
- **Shifts the post-shock state +2–3.5 %.** A Hugoniot-preserving viscosity should
  leave the post-shock equilibrium alone; it does not here because penetration is
  ongoing, so `div v ≠ 0` and `q` persists as a small standing pressure wherever
  material compresses.
- **Shocks arrive ~8 % earlier** (substep 693 → 636) — the classic over-strong-AV
  artifact.

**It is inert below hypervelocity, and that is structural.** `apfsds_vs_rha` at
matched 150 substeps moves ≤0.20 % on every metric (tip −0.00 %, RHA spall
+0.20 %, rod velocity −0.00 %) — at the ≤0.11 % repeat-bake scatter floor. Not
luck: a KE deck barely compresses (worst live `J` **0.7818** vs the jet's 0.46)
and its compression rate is an order of magnitude lower (`div_v` **−313** vs
−3606 /ms), and `q` scales with both.

**Why it is kept, and why it is off.** Off, because damping ~1 % is not worth
+57 % substeps plus re-timing and re-measuring 30 baked decks. Kept, because it is
the **prerequisite for Mie-Grüneisen**: AV work is currently dissipated to
*nothing* — there is no energy equation and Murnaghan is a cold curve, so no
thermal pressure exists to feed. That is self-consistent today, but the moment a
thermal term lands, AV heating *should* raise thermal pressure. This is also the
cleanest reason AV had to come **before** Mie-Grüneisen rather than after: you
cannot measure what a thermal term moves while the quantity you would measure it
with is still misattributed. With `av_c_q = av_c_l = 0` the term is identically
zero *and* the CFL bound is untouched, so every existing deck is bit-for-effect
the pre-AV solver — a property pinned by `tests/test_artificial_viscosity.py`.

**Verified by rebake, not by argument** (§3.4's rule): the shipped
`heat_vs_composite` re-baked on the AV code with the default off gives **240
substeps** — the pre-AV count exactly — and **worst live `J` = 0.4314** against the
**0.4315** recorded pre-AV, a match to 1 part in 4300. That agreement is also
*independent corroboration of the misattribution above*: milestone 9 extended this
deck's window 25 → 30 µs, which normally invalidates every quoted figure, yet this
number did not move — **because it is set at the first impact at t≈0.4 µs, inside
both windows.** A figure invariant to a window extension is a figure set early,
which is precisely what "impact transient, not steady stagnation" predicts.

**A near-miss worth recording.** The first spectral test asked for power at
`freqs > 0.25·Nyquist` — period < 8 substeps — found exactly zero, and would have
published *"there is no ring"*. That band is ~20× too fast to contain a
159-substep ring. **A null in the wrong band rules out nothing:** compute the
predicted period *first*, then choose the band.

---

### 3.10 Mie-Grüneisen — the EOS gets an energy equation (milestone 13)

§3.5 shipped Murnaghan and named its own limit precisely: *"a **cold** curve with
no shock heating"*, reading 0.93× copper's Hugoniot at `J=0.9` but **0.68×** at a
7 km/s equilibrium. Milestone 13 closes that, and the closure is **the energy
equation, not the pressure formula**:

    p(J, e) = p_cold(J) + Γ₀ρ₀e          Γ = Γ₀·J   (i.e. Γρ = Γ₀ρ₀)
    p_H(η)  = K₀η / (1 − sη)²             η = 1 − J

**There is no cheap Mie-Grüneisen. Shipping the reference curve alone is a
REGRESSION** — the `(1 − Γη/2)` factor *subtracts* pressure, so the cold part lands
*below* Murnaghan. Measured against copper's Hugoniot, the yardstick §3.5 already
uses:

| J | Murnaghan (M8) | MG cold only | MG + energy eq |
|---|---|---|---|
| 0.90 (KE deck) | 0.95 | 0.90 | **1.00** |
| 0.63 (jet stagnation) | 0.73 | **0.63** | **1.00** |

The Murnaghan column reproduces §3.5's published 0.93× / ~0.68×, which is how we
know the script measures the right thing. **The whole benefit of M13 lives in the
energy equation** — and that vindicates §3.9's ordering: AV work is the shock-heating
mechanism that *feeds* `e`. AV's real job was never damping the ~0.9 % ring; it was
carrying shock heating (velocity-error spread 0.223 → 0.003). **AV is therefore ON
by default from M13**, reversing §3.9's measured "off" — the reason it was off (its
work dissipated to nothing) no longer exists.

**Two per-material constants, not three.** `c₀` needs no new constant: `c₀ = √(K₀/ρ₀)`
with the existing `K₀ = λ+µ` lands within 1–10 % of public shock data (copper
**0.99×**, RHA 1.06×, tungsten 1.10×) and preserves M8's tangent-match at `J=1`, so
MG stays a **large-strain-only** change. Only `s` and `Γ₀` are new.

**Solved in CLOSED FORM — no iteration.** MG is linear in `e`, so the implicit
coupling (p depends on e, e depends on p) resolves algebraically:

    ρ₀e¹(1 + Γ₀·ΔJ/2) = ρ₀e⁰ − [(p_cold(J¹) + p⁰)/2 + q]·ΔJ

`q` is AV's, and it must be **the same q the momentum scatter uses** — a different one
would silently violate the jump conditions. **The deviatoric elastic work is NOT fed
to `e`**: it already lives in `F` (we are hyperelastic, unlike a hypoelastic
hydrocode where all work feeds `e`). Feed `e` the volumetric + dissipative work only.

#### What earns the milestone: a 1-D Lagrangian piston

`p(J, e_H) = p_H` is a **TAUTOLOGY** — it is built into the MG algebra and holds for
any Γ. It validates the algebra, not the scheme. The test that earns the milestone is
a 1-D piston (no MPM, so no transfer confound):

| u_p (m/s) | p/p_H | p_cold/p_H | u_s measured vs c₀+s·u_p |
|---|---|---|---|
| 300 | **1.000** | 0.931 | +1.6 % |
| 1000 | **1.000** | 0.814 | +0.7 % |
| 2000 | **0.999** | 0.709 | +0.4 % |

The energy integration **lands on the Hugoniot**, and `u_s = c₀ + s·u_p` matches to
<2 % having been **fitted to nothing**. **The falsifier matters as much as the
result:** with AV work not fed to `e`, `p/p_H` drops to **0.923** (the isentrope);
with `e` never fed, **0.755** (the cold curve). Three states from one knob — so the
test is sensitive to the accounting, and a broken energy scheme is **worse than
shipping nothing**.

Confirmed in the kernel too, with AV on: live shocked RHA reads `p/p_H` = **0.9959**
against `p_cold/p_H` = 0.9208.

#### The pole is a hard singularity and the guard is LOAD-BEARING

`p_H` poles at `J = 1 − 1/s`, and **past the pole it SOFTENS** (the squared
denominator keeps growing) — losing exactly the monotone-and-stiffening property
§3.5 chose Murnaghan *for*. Below `J_sw = 1 − MG_F_SWITCH/s` the law hands over to
Murnaghan, matched in value and tangent. The fallback region then behaves like the
pre-M13 shipped law, which is the point: **the `u_s`–`c₀`–`s` fit has no meaning past
its own pole.**

| material | J_pole | J_sw | worst live J (M13) | |
|---|---|---|---|---|
| copper_jet | 0.328 | 0.396 | **0.5226** | 32 % clear — guard does not engage |
| nera_filler | ~0.50 | 0.55 | **0.5434** | **inside the fallback; 2 live particles** |

**§3.6.2 predicted this a priori and it held.** M12 warned "the MG pole guard stays
load-bearing on this deck and must be designed, not assumed." It is, and it was. An
interim AV-off bake showed only 1 particle and was briefly read as "a backstop, not
load-bearing" — that reading did not survive the shipped configuration. **Do not
silence it by lowering `MG_F_SWITCH`.** The guard naming `copper_jet` or `rha` would
mean M13 is quietly not in effect where it is supposed to matter.

#### MG relieved the NERA crush that M12 could not

| | worst live J |
|---|---|
| Murnaghan (M12) | **0.2421** |
| Mie-Grüneisen (M13) | **0.5434** |

§3.6.2 concluded that relief "needs a **VOLUMETRIC** (compaction) criterion, not a
deviatoric one", because plastic flow is isochoric and cannot relieve volumetric
confinement however hard it engages. **MG's thermal pressure `Γρ₀e` IS that
volumetric mechanism:** compression feeds `e`, `e` pushes back, and the crush arrests
at the pole instead of driving 2.3× past it. M12 was right about the *kind* of thing
required and wrong that MG would not supply it. Not a `dt` artifact — M12's own
0.2159@110 vs 0.2120@336 shows `dt` cannot move it, and the 2.26× shift is ~200× the
~1 % extremum wobble. **Do not quote the 4th decimal**: it is a min over every
particle over every frame (§3.6.1).

This is *not* a CFL saving in M13. `EOS_CFL_J_MARGIN` **stays at 0.35** — the decks
now use only 18 % (nera) and 7 % (jet) of their budget, so 0.35 is conservative,
which makes a rebake at it *correct, merely slow* (root §1: bake cost is
irrelevant). Recalibrating a **global** stability constant inside the same change as
a new EOS **and** a boundary-condition fix would be three variables at once. Those
percentages are the evidence base for doing it later, as its own A/B.

#### Honest limits, stated rather than discovered later

- **`e` drops plastic dissipation.** The update is volumetric + AV work only, so
  strongly-shearing regions — the crater walls, not the jet stagnation point — are
  missing a real heat source and `e` **under-reads** there. Fine for the
  near-hydrostatic jet tip; a real omission elsewhere. This is also why the cache
  column is `internal_energy` and **not** `temperature` (CACHE_FORMAT §2): a
  temperature would need a per-material `c_v` *and* would under-read exactly in the
  zones a viewer most wants to look at.
- **`e ≥ 0` is a theorem here that float32 violates.** Both compression and tension
  give `ρ₀de = −p·dJ > 0` from rest, and `q ≥ 0` only adds — yet cancellation drives
  `e` slightly negative at *birth*, in every deck. Left alone that seeds a runaway
  (negative `e` ⇒ negative thermal pressure ⇒ spurious tension ⇒ more negative `e`).
  The clamp is one-sided, so it injects a bounded trickle rather than removing any.
  **Judge it RELATIVE to `e`, never in absolute J/kg**: the shipped decks clamp at
  0–9 float32 eps of their own `e_max`, and an absolute 1.0 J/kg verdict is
  *anti-correlated* with the risk — it condemned `apfsds_vs_era_oblique` (e_max
  1.07e6) while clearing `apfsds_vs_nera` (1.00e7, 9.4× larger).
- **A negative `e` is a TRACER, not a cause.** It was universal and born at roundoff
  in every deck long before anything went wrong, and clamping it did **not** fix the
  ERA divergence (§1.1.1 — that was a dead boundary condition).

---

### 3.11 The CFL margin multiplied a volume ratio (milestone 14)

Milestone 13 closed by writing down an observation and declining to act on it:
decks were using **5–22 % of their own CFL budget**, so the margin was
"conservative = correct, merely slow", and recalibrating a global stability
constant alongside a new EOS and a boundary fix would have been three variables at
once. That deferral was right. This section is the follow-up it asked for, and the
suspicion recorded with it — *"I'd suspect the sizing formula rather than the
constant"* — was correct.

**The defect.** `EOS_CFL_J_MARGIN` made headroom by multiplying `J`:

```
Jd = 0.35 * J_eq(p_stag)        # "35 % of margin"
```

`J` is a volume **ratio** in (0,1], and the EOS diverges as `J → 0`. Scaling it is
not a 35 % adjustment; it is a demand that the material compress to roughly a third
of its equilibrium volume ratio. Measured across the shipped decks, the design state
landed **past every material's Mie-Grüneisen pole and below its guard switch — on
30 of 30 decks, in 54 of 68 (deck, material) pairs**. The bound was therefore read
off the pole guard's extrapolated Murnaghan `J^−4` backstop: the branch `J_FLOOR`'s
own comment calls *"a degeneracy backstop, NOT a physical limit"*.

`apfsds_vs_rha` is the clearest case, because nothing about it is exotic:

| | value |
|---|---|
| `rha`'s honest equilibrium under the deck's impact | `J_eq = 0.902` (a **10 %** compression) |
| what the margin designed for | `Jd = 0.316` |
| `rha`'s MG pole (`1 − 1/s`) / guard switch | 0.329 / 0.396 — **`Jd` is past both** |
| design sound speed read off the backstop | **137 281 mm/ms** |
| steel's real shocked sound speed | ~6 000 mm/ms |

`dt` was sized against a 137 km/s wave in a material that never exceeds ~6 km/s.
That ~20× is the 5–22 %-of-budget figure, from the other end.

**The scale was wrong too, and that part no constant could fix.** `p_stag = ½ρv²`
is the *steady* stagnation pressure of an established penetration channel. The
substep has to survive **first contact**, which is a shock, and a shock's pressure
comes from impedance-matching the two Hugoniots — not from a kinetic-energy
density. The two disagree by a **velocity-dependent** factor:

| deck arm | `p_impact / p_stag` |
|---|---|
| 1500 m/s | **3.58×** |
| 3500 m/s | 1.89× |
| 7000 m/s | **1.25×** |

`½ρv²` is quadratic in `v` while the contact shock is closer to `ρcv`, so the error
*shrinks* as the deck gets faster. **That spread is the fingerprint on the old
constant's history.** 0.8 → 0.55 → 0.35 was a single number being re-cut to patch an
error that varies 3× across the repo's own velocity range — which is exactly why it
kept needing re-cutting, and why margin 0.8 could survive `heat_vs_composite` while
failing elsewhere.

**The fix.** `EOS_CFL_P_MARGIN = 3.0` multiplies a **pressure** — the deck's
impedance-matched contact pressure, bisected host-side from the *same*
`u_s = c₀ + s·u_p` fit milestone 13 already ships, so it costs **zero** new material
constants. Linear, interpretable, and velocity-adaptive by construction, which
removes the reason the old constant drifted. 3.0 is not a fudge: a 1-D shock
reflecting off a stiffer neighbour roughly doubles, so 3× is one doubling plus a
half.

**Why the prize is calibration and not the ~4–5× of substeps.** Root §1 says bake
cost is irrelevant, so "merely slow" would barely be a defect. The real problem is
that **the old bound never failed** — it erred in the *safe* direction, which is
precisely why it survived four milestones unexamined. A bound that over-predicts 20×
is not conservative, it is **uncalibrated**, and an uncalibrated bound is equally
free to *under*-predict on the next deck. That is not hypothetical: it is what
margin 0.8 did on `heat_vs_composite`, surviving only because that deck's ceramic
donated headroom the copper tip borrowed. This is the repo's recurring shape —
*green because nothing looked, not because the answer was right.*

#### The deck that is not shock-loaded, and why it is priced in its own YAML

`apfsds_vs_nera` binds, and no pressure bound expresses it. Its worst particles are
**2–4 of 36 966** filler particles dragged 34 mm downrange and pinned in the *main
plate's* crater between the rod tip and the plate (§3.6.1) — a **kinematic vise**
whose compression is set by geometry, not pressure. `nera_filler` sits at its pole
(`J = 1 − 1/s = 0.5`) where the EOS asymptotes, so pressure is a near-flat lever
there: **12× the impact shock moves the design J only 0.597 → 0.511.** Covering the
vise globally would need `P ≈ 50` — which is not a statement about a shock. It is a
global stability constant sized against a 2-particle extremum: the anti-pattern the
old constant's own comment argued against, and then committed.

So the vise is priced where the vise lives — `cfl_p_margin: 20.0` in the deck (root
§9: scenarios are data) — and the global constant stays a statement about shocks.

**Measured, and the measurement is what licenses it:**

| config | substeps | worst live `J` | `c_eff` | audit |
|---|---|---|---|---|
| shipped M13 (`J`-margin 0.35) | 1047 | 0.5385 | 113 228 | OK, **18 %** of budget |
| global `P=3`, no override | 153 | 0.5466 | 109 571 | **BREACH 1.22×** |
| **`P=20` override — ships** | **248** | 0.5462 | 109 761 | OK, **76 %** of budget |

Two things fall out of that table, and the second is the load-bearing one.

- **`c_eff` moves −3.2 % across a 6.8× change in `dt`.** A geometric trap does not
  dissolve under refinement — §3.6.2 said exactly this — so the override is sized
  against a **stable** number. Contrast the jet's shock-ring ratio, which drifts
  with `dt` (0.648 → 0.713) and must never be sized against. *This* is the
  difference between a measured constant and a fitted one.
- **The `P=3` breach is survivable, and the override is not what prevents a
  divergence.** It bakes finite, and nera's go/no-go conclusion (filler cohesion)
  moves **−0.06 %**. Read the audit's ratio correctly: it is a fraction of the
  **CFL = 0.3 safety factor**, not of the stability limit, so a 1.22× breach means
  the substep ran at Courant ≈0.37 against a limit near 1. **A breach warning is
  "you have eaten into the safety factor", not "this diverged".** The override buys
  a warning-free bake at a real margin — not stability.

**What moved, and what held.** Every deck was rebaked. Nera's filler cohesion holds
to **0.10 %**; its spall fraction moves **+10 %** and the rod tip **+2.2 %**, which
is the pattern every milestone here has produced — *the numbers move, the
conclusions hold.* Treat the absolutes as readings of one configuration (see §3.5 on
not quoting tip-`J` to four decimals; this is the fifth demonstration).

**Correcting the record.** The old comment's *"margin 0.55 let `apfsds_vs_nera`
breach by 2.41×"* was measured when nera's worst live `J` was **0.2421**.
Mie-Grüneisen relieved that crush to 0.5434 (§3.10), so the historical breach
**stopped constraining the constant at milestone 13** — and nobody noticed. M13 made
the margin over-conservative *by succeeding*. The 5–22 % it recorded was that fact,
already visible in the audit line, waiting to be read.

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
