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
  **Measured in §1.1.2, and this sentence was over-general**: it happens on 8 of
  30 decks, not as a rule — on the other 22 nothing that started clear of a wall
  ever gets within 2.1 mm of one. Where it does happen the second clause holds
  exactly, and for a reason §1.1 states two paragraphs up without following it
  through: a slip wall zeroes the inbound normal velocity rather than negating it,
  so the spray arrives, stops, and slides. **It does not bounce.**

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

#### 1.1.2 What actually reaches a wall — all 30 decks, and nothing is indicted

§1.1 states two things it never measured: that "late in a bake, spall spray **does**
reach the top/bottom walls and slide along them", and that "the artifact is confined
to the far-field debris". §1.1.1 is the standing lesson about precisely that habit —
*a stated invariant is not a tested one* — so both were measured.
`tools/measure_wall_contact.py` reads baked caches only; there is no rebake here and
no GPU was involved.

**The bar had to be settled first, because the repo carried two.** The README asked
that "oblique-deck debris **never reaches a wall**", which is stricter than the
seeding design permits: `_seed` lays every slab wall-to-wall *on purpose* so the
mirror continues it (§1.1). Material at a wall is not the defect. The bar used here
is the weaker, correct one:

> **No wall-reflected momentum contaminates a quoted figure, inside that figure's
> measurement window.**

**The instrument counts travel, not proximity.** "Particles within 3 cells of a wall"
fires at frame 0 on all 30 decks, because that is where the armor is. A particle is
counted only if it **started > 3.0 mm clear of that wall and later came within
1.2 mm** (≥ 3 cells on every deck; the cache does not record `dx`, so the thresholds
are fixed lengths that bracket the repo, and both are swept). Sixteen tests pin it,
and **twelve mutations of the tool were each caught by exactly one of them** —
including the one that matters most here, a control deck where nothing travels
reading zero against a twin that reads one, *one particle apart*. (The remaining
four tests are the green halves of those pairs: controls, not red-proven
assertions.)

**Result: 8 decks of 30 show any arrival at all; 647 particles in total.**

| family | arrivals | first contact | what arrived | max bound on a per-material damage fraction |
|---|---|---|---|---|
| `apfsds_vs_era_oblique` | 90 at `y_lo` | 115.6 of 140.0 µs | 65 `rha` + 25 rod, **100 % spalled** | `rha` **0.0314 pp**, rod 0.2207 pp |
| `apfsds_vs_era_oblique_inert` | 129 at `y_lo` | 115.0 of 140.0 µs | 88 `rha` + 41 rod, **100 % spalled** | `rha` **0.0425 pp**, rod 0.3620 pp |
| `standoff_s00` / `s30` / `s60` | 62 / 30 / 12, split evenly `y_lo`+`y_hi` | 35.0 / 39.4 / 43.8 of 45.0 µs | `rha`, **100 % spalled** | **0.0123 / 0.0059 / 0.0024 pp** |
| `standoff_conv_*_s00` (3 decks) | 128 / 98 / 98 | 33.6–34.8 of 45.0 µs | `rha`, **100 % spalled** | 0.0253 / 0.0086 / 0.0048 pp |
| the other **22 decks** | **none** | — | — | — |

**Threshold sensitivity, since the verdict must not be one.** The counts above are
the `1.2 mm` column. Widening to `2.4 mm` — roughly twice the 3-cell slip band, so
material there has not entered it — takes **8 decks to 12**, adding
`sweep_tungsten_v2500` (353), `sweep_tungsten_v1500` (210), `sweep_copper_v1500`
(13) and `sweep_copper_v2500` (1). None of those approach closer than 2.11 mm and
none is an arrival. Narrowing to `0.6 mm` takes it the other way, to 2 decks.
(Ignore the `5.0 mm` column: eligibility there needs a start past 5.0 mm and a
loaded plate bulges laterally ~0.8 mm, so it counts seeding —
`standoff_conv_dx188_s90` reads 4091 with zero arrivals at any wall.)

**And a bound is over the population it divides by.** The `k/n` figures above use
`n` = *the whole material*. That is the honest ceiling for a fraction taken over
all of `rha`, but `measure_reactive_ab.py` scopes to the **main plate**, and the
oblique deck carries three `rha` slabs (6 + 6 + 24 mm) under one material id. The
main plate is 138632 of the 206830 `rha` particles, so a main-plate-scoped bound
is ~1.5x larger: pessimistically assuming *every* touched particle landed there —
which late-time crater debris makes plausible — **0.0469 pp** (reactive) and
**0.0635 pp** (inert). Still three orders of magnitude under the −40.7 % effect,
but quote the number that matches the figure's own population.

Three findings, in order of how much they change.

**1. Nothing that touches a wall comes back.** This is the one that decides the
verdict, and it is measured rather than bounded — each wall-touched particle is
followed from its own arrival frame onward. On every one of the eight decks the
answer is the same: **no wall-touched particle ever left the 1.2 mm arrival band**,
observed for 24.4 µs afterward on the oblique deck (and 10–11 µs on the standoff
family; `standoff_s60`'s window is only 1.2 µs, so that deck's clean read is the
weakest of the eight and should not be leaned on). On `apfsds_vs_era_oblique` the
closest any wall-touched particle came back to the centreline was **108.8 mm of a
110.0 mm half-span**.

The mechanism is the boundary condition itself, and it is worth stating because it
makes the result unsurprising rather than lucky: **a free-slip wall does not reflect.**
It zeroes the inbound normal velocity; it does not negate it. Material arrives,
stops, and slides — §1.1's "slide along them", now measured. The ballistic estimate
that first framed this question (1620 m/s × the remaining 24 µs = 39 mm of possible
return travel) was wrong by a factor of ~30 in the safe direction, which is exactly
why it was replaced with a measurement.

What can still travel back is the **stress wave**, and that is grid-transmitted —
squarely in the tool's blind spot. Every contamination figure here is a lower bound
on direct participation only.

**2. The standoff family's "trend" is a window artifact, not a standoff effect.**
Arrivals fall 62 → 30 → 12 → 0 across `standoff_s00/s30/s60/s90`, which reads as a
standoff-dependent wall effect and is not one. Look at *first contact* instead. The
jet tip is 7000 m/s and the standoffs are 0/30/60/90 mm, so impact — and everything
downstream of it — is displaced by exactly `S / 7.0` µs:

| deck | predicted first contact | measured | bake ends |
|---|---|---|---|
| `standoff_s00` | 35.00 µs | **35.0** | 45.0 |
| `standoff_s30` | 39.29 | **39.4** | 45.0 |
| `standoff_s60` | 43.57 | **43.8** | 45.0 |
| `standoff_s90` | 47.86 | **never** | 45.0 |

Residuals are **0.00 / +0.11 / +0.23 µs** against a 0.2 µs frame — the first two
inside one frame, `s60` just over it. (They grow monotonically with `S`, which is
what a jet tip losing a little speed in free flight would look like; it is a
sub-frame effect and nothing here rests on it.) It is the *same event at the same
elapsed time from impact*, shifted; the counts track how much of it fits before
`total_time`, and `s90` reads zero because its first contact falls ~2.9 µs **past
the end of its own bake**. This is §3.8's already-documented elapsed-from-impact
systematic surfacing in a new instrument — it is **not** evidence for a remedy, and
must not be used as one.

**3. Nothing has ever reached the exit face.** `x_hi` reads zero arrivals on all 30
decks, and the closest approach on any deck is **10.65 mm** (`sweep_tungsten_v3500`).
That retires the one branch of this work that would have been kernel code: an
arrival at `xmax` would mirror a *second target* downrange, which no domain resize
can fix. ⚠️ But 10.65 mm in a 300 mm domain is not a comfortable margin, and the
three closest decks are the tungsten sweep at v2500–v5000, where the rod perforates
and exits. **A future deck that runs longer or faster than these will reach `x_hi`**,
and that is the case to re-check before extending the sweep, not a defect today.

**Also confirmed: the §1.1.1 fix, on every deck.** The worst mid-height asymmetry
across all 28 applicable decks is **0.039 mm** (`apfsds_vs_nera`), against the dead
walls' 0.49 mm signature. Wall-spanning material only — the free-ended rod's ends
are not a boundary measurement, and headlining them reported a mushroomed tip as an
asymmetry of 0.216 mm.

**Verdict: no remedy is warranted, and no deck was rebaked to establish it.** The
largest material-wide bound anywhere is **0.36 pp** (the rod on
`apfsds_vs_era_oblique_inert`); the main-plate spall fraction
`measure_reactive_ab.py` actually quotes is bounded at **0.0635 pp** even
pessimistically — against an A/B effect of −40.7 %. The standoff family's depth
figures are read along the axis, tens of mm from either wall, and nothing came back
off a wall at all.

Two caveats travel with that verdict. The tool sees only **dumped frames**, so a
sub-frame excursion is invisible (§3.9's aliasing lesson); and it measures **direct
participation**, so grid-transmitted impulse is uncounted. Neither is idle, but
neither is load-bearing here: arrivals happen only in the last ~25 % of a bake, at
debris speeds, and nothing travels back.

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

**Read this first: the shipped decks under-read the effect they measure, by ~2.0×
on the excess, and they are not grid-converged.** `standoff_s00/s30/s60/s90` measure
a depth ratio of **1.27** between S=90 and S=0 where the a-priori prediction is
**1.536**. The cause is resolution, not physics: the jet is 3 mm across = **8 cells**
at the shipped `dx=0.375`, and it *thins as it stretches* to ~1.1 mm ≈ **3 cells** by
the end of the window. The quantitative claim below rests on the six
`standoff_conv_*` decks, **not** on the four shipped ones. Same posture as §3.5's
tip-`J`: quote the trend, never the value.

> **⚠️ RE-MEASURED 2026-07-27 (milestone 17). Every number in this section is now
> read off the SHIPPED P=4 caches; the older values are kept below as labelled
> history, because the pattern they make is worth more than any one of them.**
>
> | S90/S0 | shipped | dx=0.250 | dx=0.1875 | 6 mm jet | vs a-priori 1.5357 |
> |---|---|---|---|---|---|
> | Murnaghan era (M10, `5b9cc5b`) | 1.229 | 1.383 | 1.429 | 1.501 | fat jet −2.3 % |
> | MG era (M13, `39ffe35`) | 1.312 | *not re-measured* | | | |
> | **today (P=4, M17)** | **1.2657** | **1.4573** | **1.4968** | **1.5587** | **fat jet +1.5 %** |
>
> The **conclusions are untouched** — monotone in `cells across the jet`, the two
> 16-cell routes close on each other from different grids and different jets, the
> shipped row worst — and the absolutes have now moved **three times**. Seventh
> demonstration of the repo's own rule: treat every figure here as a reading of one
> configuration, and **re-measure rather than translate**.
>
> Two things did change in kind, and neither is cosmetic:
> * **The under-read on the *excess* is ~2.0×** (0.536 predicted vs 0.266 measured),
>   not the ~2.3× of M10 or the ~1.7× M13 briefly recorded. `--convergence` now
>   COMPUTES that figure instead of printing it from a string, which is how it went
>   stale in the first place.
> * **The fat-jet row's sign flipped: it now OVERSHOOTS the prediction by +1.5 %**
>   where it used to sit 2.3 % under. The posture below is unchanged — it is inside
>   the ±3.5 % per-fraction scatter either way, so it remains *consistent with* the
>   prediction rather than confirming it — but the row must not keep its old gloss
>   of "approaching from below".
>
> **This does not rescue the shipped decks and must not be read as convergence:**
> the deficit is `cells across the jet`, and no EOS can add resolution. The
> `standoff_conv_*` decks still carry the quantitative claim. The lab-time trap
> below (107.0 → 82.1 mm, *falling* with standoff) reproduces unchanged in shape.
>
> **And a new caveat that §3.8 never carried: the DEPTHS under this ratio are
> `dt`-sensitive where the ratio is not** (§3.14). Refining the substep alone
> suppresses both arms' matched-fraction depth by **3–6 %** while moving their
> quotient only **1.6–1.9 %**. Anyone quoting a standoff *depth* from this family
> inherits that; anyone quoting the *ratio* mostly does not.

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
the end of the window **falls** 107.0 → 82.1 mm as standoff rises, purely because a
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
| 0  | 168 | 21.0 | 28.7 | 37.0 | 46.0 | 1.000 | 1.000 |
| 30 | 198 | 23.3 | 32.1 | 41.5 | 51.3 | 1.116 | 1.179 |
| 60 | 228 | 25.1 | 34.7 | 44.7 | 55.2 | 1.202 | 1.357 |
| 90 | 258 | 26.3 | 36.6 | 47.2 | 58.0 | **1.2657** | **1.536** |

Monotone, right sign, ~200× the 0.11 % run-to-run floor — and **short by half the
predicted excess**. The premise test says why: at matched arriving-element velocity
the sim's `u` is **not** a function of `v` alone — it fell from 2272 m/s (S=0) to
1960 (S=90) at v=5000, −14 %. Since `P ∝ Z` holds for any `u(v)`, that is where the
proportionality is lost, and it is a real defect to be explained rather than a
discrepancy to be shrugged at. **Those two `u` figures are the one thing in this
section NOT re-measured at M17** — they came from an ad-hoc probe that was never
committed, so they are Murnaghan-era and the −14 % should be read as a sign and an
order, not a value.

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
| 3 mm jet, `dx=0.375` — **shipped** | 8 | **1.2657** | −18 % |
| 3 mm jet, `dx=0.250` | 12 | 1.4573 | −5.1 % |
| 3 mm jet, `dx=0.1875` | 16 | 1.4968 | −2.5 % |
| **6 mm jet, `dx=0.375`** | **16** | **1.5587** | **+1.5 %** |

`python tools/measure_standoff.py --convergence`. **Cells across the jet is the
controlling parameter**, reached by refining the grid or by fattening the jet, and
the shortfall is numerical. At 16 cells the fat jet sits within the per-fraction
scatter (±3.5 %) of the a-priori 1.536 — **consistent with the prediction, which is
not the same as confirming it.** It now sits *above* rather than below, and that
changes nothing about the posture: a row inside the scatter band does not confirm a
prediction from either side. The two 16-cell routes still differ, by **4.1 %**, so
cells-across is dominant but not the only term.

**Milestone 17 took that 4.1 % apart** (§3.14). The `cells` column above is no
longer a hand-computed label either — `measure_standoff.py` reads it off the seeded
lattice, so the two 16-cell rows are confirmed to be 16 cells by the cache rather
than by the arithmetic that motivated them.

> **⚠️ "CELLS ACROSS THE JET IS THE CONTROLLING PARAMETER" IS AN APPROXIMATION, AND
> MILESTONE 18 MEASURED THE ERROR** (§3.15). `cells ≡ diameter/dx`, so that sentence
> is a **scale-invariance hypothesis** — a claim the response depends on the ratio
> alone — and it had never been tested, because every row above reaches a cell count
> by moving one factor. A **third 8-cell arm** (6 mm jet at `dx=0.75`, `dt` pinned to
> the shipped clock) is the same discretization scaled 2× against a standoff, a plate
> and a process zone that do **not** scale with the jet. It disagrees with the shipped
> 8-cell row by **−4.10 %** at f=0.30, rising to −12.28 % at the least-converged
> f=0.15 (quote the former). So the claim is good to
> a few percent and is **not an identity** — and the residual scale dependence is the
> same order as the 4.1 % it was invoked to explain. Read the fourth row as
> *consistent with* the prediction, never as reaching it.

**Honest limits.**

- **Do not quote a Richardson extrapolation from these four points.** The observed
  order is ill-conditioned — it swings from ~0.7 to ~5 depending on the matching
  point, and the extrapolated value with it. Adding a fifth grid would not fix that
  (the conditioning is the small high-resolution increments, not the count). The
  monotone trend plus the independent fat-jet route is the honest statement, and
  **"converges toward" is not "converged"**.
- **A trap worth naming: the under-resolved curve looks like the textbook result.**
  At the shipped `dx` the increments saturate (5.3 → 3.9 → 2.8 mm at f=0.30), which reads as a
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

**The fix.** `EOS_CFL_P_MARGIN = 4.0` multiplies a **pressure** — the deck's
impedance-matched contact pressure, bisected host-side from the *same*
`u_s = c₀ + s·u_p` fit milestone 13 already ships, so it costs **zero** new material
constants. Linear, interpretable, and velocity-adaptive by construction, which
removes the reason the old constant drifted. `p_impact` is the *first contact* shock
and a 1-D shock reflecting off a stiffer neighbour roughly doubles, so 4× is one
doubling plus a doubling of headroom for the transient that overshoots it — and the
value is **calibrated, not argued**: `P=3` shipped first and audited one deck over
budget, which is what moved it to 4 (below).

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

Raising the global constant to 4 did **not** make the override redundant: it lifts
nera's `c_max` 89 619 → 109 031 against a measured `c_eff` of 109 767, i.e. still a
breach, at 101 %. (That is an estimate — it holds `c_eff` fixed, the same assumption
the `P` table above gets wrong in the safe direction. It is a *sound* assumption here
and nowhere else: nera's `c_eff` is the dt-stable one, −3.2 % across 6.8×.)

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

**The shipped `P=4` tally: 12–76 % of budget, zero breaches, 30 of 30 decks clean.**
It was 5–22 % under the `J`-margin. `apfsds_vs_nera` is the 76 % and is not evidence
about the global constant — it runs its own `cfl_p_margin: 20`. Excluding it the
spread is **12–68 %**, and the top is `heat_vs_composite_uniform`, the deck that
breached at `P=3`.

**The low end is not a new defect.** The decks that sit low are the **ERA family**
(12/13/16/17 %), and the reason is the nera situation in miniature: `era_filler` is
another soft material near its own pole (`s=2.0`), so the deck-wide-worst pressure
designs it close to the guard while the deck only reaches ~0.68. Its near-pole
stiffness then sets `c_max` **for the whole deck** — measured on `apfsds_vs_era` at
`P=4`, `era_filler` reads `c=66 644` against `rha`'s 15 799 and `tungsten_rod`'s
9 688, a **4.2×** margin, and with the AV terms it reproduces that deck's audit line
(109 031) exactly. So a soft interlayer, not the steel or the rod, is what prices
every ERA deck's substep. The distinction from the old bound is the one that matters:
**that design state is still on the physical MG branch**, not on the guard's
extrapolated backstop. It is conservatism, not miscalibration.

#### The one breach at `P=3`, and what it took to close it

**`heat_vs_composite_uniform` audited at 101 % — the only deck ever over budget.**
It is worth reading closely, because the mechanism is not what it looks like, two
plausible diagnoses died on the way to it, and it is the entire reason `P` is 4 and
not 3.

**Read the figures below by which question they answer**, because this subsection
spans both constants: **the breach and its diagnosis are `P=3`** (that bake is the
calibration data, and `P=4` has since overwritten its caches), while **the remedy,
the ceiling and the verification are `P=4`** (the shipped bake). Each is labelled
where it could be mistaken.

**What it is: the jet compresses past its own design `J`.** `copper_jet` designs to
`J=0.4624` under this deck's `p_design`, and reaches **0.4405** live — past its own
equilibrium, i.e. the transient overshoots the pressure `P=3` allows for. Its *cold*
sound speed there is **34 087 mm/ms**, which reproduces the audited `c_eff` to within
**0.6 %**. The graded twin corroborates independently at **0.06 %**: live `J=0.4558`,
copper cold `c=28 987` vs an audited ≥29 005. The uniform jet goes furthest because
it is never replaced by slower material (§3.4), so it holds peak stagnation longest —
the deck's own reason for existing is why it binds.

**Why `c_max` did not cover it.** `c_max` is a **max over materials** of each
material's cold `c` at *its own* design `J`, and here that max is **`rha` at
`J=0.4873` — a compression rha never reaches** (rha at 0.4405 would give `c=43 497`
and an audit near 69 600, far above the observed 56 197). So the bound had been
covering the jet by *borrowing rha's larger number*, and on the one deck that
overshoots hardest that borrowed cushion ran out. Note this makes `j_design`
(a **min** over materials) and `c_max` (a **max**) different materials — the printed
"EOS design J=0.345" is ceramic's, and is not the state that set the bound.

**Two diagnoses that were falsified — do not re-run them.**

- **"The bound is cold-blind."** `_eos_sound_speed` is sized at `e=0`, so the theory
  was that shock heating stiffens the live material past a cold design. Measured, heat
  at the design `J` is worth only **+6.2 %** (rha 28 045 → 29 784) — far short of the
  ≥33 885 the live material reached. Worse, **the binding particle is essentially
  cold**: `c_hot` at the live `J` (36 210) *overshoots* the observed `c_eff`, which is
  arithmetically impossible. A stagnating jet is not a single shock and deposits far
  less heat than the Hugoniot.
- **"Design on the Hugoniot instead of the cold curve."** This *inverts the bound*.
  At a fixed pressure the Hugoniot state is **less compressed**, and the lost `K_cold`
  beats the thermal gain: `c_hug` lands **11–28 % BELOW** `c_cold` (rha 28 045 →
  21 185), which would drop `c_max` to ~45 500 and make the breach considerably worse.
  `_eos_equilibrium_j`'s docstring already said this — *"it must not assume the shock
  heating that a real trajectory may or may not deposit"* — and the measurement
  vindicates it. **The cold curve is the conservative choice, and it is correct.**

**The remedy is `P`, and `P = 4` ships.** Overshoot past the design pressure is
*precisely* what `EOS_CFL_P_MARGIN` is the allowance for, so this is a calibration
shortfall in the constant, not a structural flaw in the formula (contrast milestone 14
itself, which was structural). Measured on this deck, before the rebake:

| `P` | `Jd(copper_jet)` | `c_max` | substeps vs `P=3` | budget, **predicted** | ERA family |
|---|---|---|---|---|---|
| 3.0 — *was* | 0.4624 | 55 372 | 1.00× | **101 % — breach** (measured) | ok |
| **4.0 — ships** | 0.4440 | 64 101 | 1.16× | 88 % → **measured 68 %** | ok, by **0.0004** |
| 5.0 | 0.4315 | 72 277 | 1.31× | 78 % | ❌ **backstop** |
| 6.0 | 0.4223 | 80 037 | 1.45× | 70 % | ❌ **backstop** |

Two independent caveats sit on that table, and both were found by checking it rather
than reading it.

**It is an upper bound, not a prediction — the `P=4` row proves it.** Every budget
figure but the first scales `c_max` while holding `c_eff` at its `P=3` value. `c_eff`
is not fixed: the rebake measured **68 %**, not the projected 88 %, because this deck's
`c_eff` came in 22 % lower. That direction is not new physics — §3.6.1 already
contrasts nera's `c_eff` (stable to −3.2 % across a 6.8× `dt` change, because a
geometric vise does not dissolve under refinement) with **the jet's ring ratio, which
does drift with `dt`**. This deck is the jet. Read the table as *"`P=4` buys at least
this much"* and the audit line for what it bought.

**Its bottom two rows are not available, and the table could not see that** because it
was computed for `copper_jet` on one deck. `era_filler` designs to `J=0.5504` against
a guard switch at **0.5500**: raising `P` 3 → 4 ate **97 %** of that clearance, and the
crossing is **between `P=4.05` and `P=4.10`**. Past it the four ERA decks size from the
guard's extrapolated backstop — *the milestone-14 defect itself* — and
`test_design_state_is_on_the_physical_eos_branch` goes red on all four (verified at
`P=5`, not assumed). **`P` therefore has a ceiling near 4.05, and 4.0 ships 2.5 % under
it.** That is tight, and it is deliberate: the invariant is pinned by a test, so a
future raise cannot land silently. **If a deck ever needs `P > 4.05`, the answer is not
a bigger `P`** — it is either a per-deck `cfl_p_margin` (the nera precedent, §3.6.1) or
a look at why `era_filler`'s guard sits where it does.

**The calibration target is `c_eff ≤ c_max`, not the diagnosis.** It is tempting to
demand that each material's design `J` bound its own live `J` — the property whose
failure *explains* the breach. Don't: that comparison is **circular**. Live `J` is
read off a bake whose `dt` was sized by the very design `J` being checked, so it is
not an independent target, and `worst live J` is a min over every particle over every
frame besides — the extremum §3.6.1 and §3.9 have both already been burned by. The
requirement is the operational one: **the bound must cover what the bake actually
reached**, with the CFL = 0.3 safety factor intact. `test_cfl_sizing.py` pins the
measured `c_eff` against a recomputed `c_max`, and says why.

**A tolerance was considered and rejected.** `CFL_AUDIT_TOLERANCE = 0.98` briefly
shipped, widening the threshold at which the audit calls 101 % a breach. It bought
**no safety** — `dt`, the physics and the bake were all unchanged, and the deck still
ate 101 % of the same safety factor; only the warning went away. It was also fitted to
one observation (101.5 → 99.5 %) and cost the audit 2 % of its sensitivity on all 30
decks, including any future deck breaching for an unrelated reason. That is this
file's recurring defect — **an instrument that is green because it is blind, not
because it looked** — and the constant it deferred to has a documented history
(0.8 → 0.55 → 0.35) of being re-cut to silence its own instrument. Raising `P` costs
16 % more substeps, which root §1 calls irrelevant, and shrinks `dt` for real.
`test_cfl_sizing.py` pins the deletion.

**The restored warning was verified to FIRE, not assumed to.** Deleting the tolerance
put back the original `c_eff > c_max` test — and at `P=4` nothing breaches, so nothing
in the shipped tally exercises that branch. An instrument that never fires is the
defect this file keeps finding, so it was checked directly: `heat_vs_composite_uniform`
re-baked with `cfl_p_margin: 1.0` prints **`WARNING: CFL margin BREACHED … c_eff=62 028
mm/ms (1.64x the budget)`**. The audit can still see a breach; it is silent because
there is none.

Tightening it further would mean sizing each layer by the shock actually
*transmitted* to it through the stack rather than by the deck's worst pair — more
physical, but a second design change, and the NERA vise is the standing proof that
a transmission estimate does not catch everything. Root §1 (bake cost is
irrelevant) says this is optional; it is left as a documented residual rather than
bundled in. **One variable at a time** — the discipline that made §3.10 defer this
milestone in the first place.

**What moved, and what held.** Every deck was rebaked twice — once at `P=3`, then
again at the shipped `P=4`. On the shipped bake `apfsds_vs_rha` reads **RHA spall
0.2085** and **rod tip 231.13 mm**, at the same 135 557 particles, against milestone
13's 0.251 / 228.27. So the largest headline figure in the repo has now read
16 → 18.2 → 25.1 → **20.85 %** across four configurations, and every conclusion those
milestones drew still stands — *the numbers move, the conclusions hold.* This is the
sixth demonstration; treat every absolute here as a reading of one configuration
(§3.5).

`apfsds_vs_nera` is the one deck the change did **not** touch, and that is a check
rather than a coincidence: its `dt` comes from its own `cfl_p_margin: 20`, so the
global constant cannot reach it. Its cohesion result stands as measured.

**An interim `P=3` figure is retracted, not carried forward.** This section briefly
recorded "spall **+10 %**, rod tip **+2.2 %**" as what milestone 14 moved. Those were
measured on the `P=3` bake, which `P=4` overwrote, so they are **not re-derivable and
must not be quoted** — the same disclosure §3.6.2 makes about M5's ad-hoc probe. The
`P=4` readings above replace them, and the probe that produced them is stated: mean
latched `damage` over the material's own particles, the definition
`tools/measure_reactive_ab.py` uses (verified identical on this cache to 8 digits).

**Correcting the record.** The old comment's *"margin 0.55 let `apfsds_vs_nera`
breach by 2.41×"* was measured when nera's worst live `J` was **0.2421**.
Mie-Grüneisen relieved that crush to 0.5434 (§3.10), so the historical breach
**stopped constraining the constant at milestone 13** — and nobody noticed. M13 made
the margin over-conservative *by succeeding*. The 5–22 % it recorded was that fact,
already visible in the audit line, waiting to be read.

---

### 3.12 The compaction criterion, closed as a negative (milestone 15)

**No kernel code, no new material constant, no rebake, no schema bump, no GPU.**
Milestone 15 was chartered by §3.6.2 — *"relieving this needs a **VOLUMETRIC**
criterion (compaction/pore collapse), not a deviatoric one"* — and carried in
README's Next list. It closes without building anything, for two independent
reasons, and the first one is embarrassing.

#### It was already delivered, by milestone 13

§3.10 says so in as many words: **"MG's thermal pressure `Γρ₀e` IS that volumetric
mechanism"** — compression feeds `e`, `e` pushes back, and nera's worst live `J`
went **0.2421 → 0.5434**, a 2.26× relief against a ~1 % extremum wobble. M12 was
right about the *kind* of thing required and wrong that MG would not supply it.

So the Next-list entry is a **stale carry-forward**: §3.6.2's ask was answered
one milestone later, inside this same document, and the list was refreshed after
M13 and M14 shipped (`3f1ffb4`) without anyone noticing that its top item was
done. The lesson is [[rebake-invalidates-documented-results]] in a documentation
register — a milestone can be closed by a *later section of the same file* and
the summary above it will happily keep asking for it.

#### What genuinely remains is pore collapse — and it is inert here by 34–688×

The specific mechanism never built is a **P-α model** (Herrmann 1969;
Carroll–Holt 1972): a distension `α = v/v_s ≥ 1` that ratchets irreversibly to 1
as pores close. That is a *different* volumetric mechanism from `Γρ₀e`, so it is
worth asking separately. It is inert everywhere this repo could put it.

A P-α material is **fully compacted above its crush-up pressure `p_c`** — `α = 1`,
the pores are gone, and the law reduces *exactly* to the solid-matrix EOS. So it
can only matter where pressures are comparable to `p_c`. Measured over all 30
decks at the design state milestone 14 already builds (`_impact_pressure` →
`_eos_equilibrium_j`) — **not** at a `worst live J` extremum, which §3.6.1 and
§3.11 both say not to size anything from:

| public crush-up pressure (order-of-magnitude, chosen before the run) | vs the **smallest** deck-wide contact shock — 34.4 GPa, `sweep_copper_v1500` |
|---|---|
| polymer foam / syntactic, ~50 MPa | 688× |
| pressed HE / powder compact, ~500 MPa | 69× |
| porous metal — Herrmann's own calibration materials, ~1 GPa | 34× |
| an absurd upper bound, 10 GPa | 3.4× |

That comparison is deliberately made **without** `EOS_CFL_P_MARGIN`: the claim is
about the physical shock, and multiplying by a stability constant that has shipped
at both 3 and 4 would make a physics statement depend on a tunable. The margin only
widens the gap (4× on 29 decks), so dropping it is the conservative direction —
the margined design pressures run 138–2754× instead.

At those ratios no distension and no crush curve moves the design state, the CFL
bound, or the pole guard's clearance. **This is not a close call that better
constants could flip.**

#### And no material here is eligible anyway

The bulk filler *does* sit in the right pressure band — §3.6.1's median live
`J = 0.9932` is **62 MPa** and its min-of-mean `J = 0.9495` is **544 MPa**, which
straddles a pressed powder's crush-up. So the question is fair. Each candidate
still fails, for its own reason:

- **`nera_filler`** is a near-incompressible elastomer. Rubber has no pores;
  distension would have to be *invented* to buy a number, which is tuning toward
  the answer (root §10). And the physical reason its vise was never constitutive:
  a confined near-incompressible solid's relief mechanism is **lateral extrusion,
  not compaction** — and §3.6.1 established those are 25 debris particles pinned
  in the *main plate's* crater 34 mm downrange, i.e. **kinematic and
  resolution-bound, not a missing material model.** No constitutive law was ever
  going to fix it. Same register as §1.1.2's free-slip walls.
- **`era_filler`** is the one material where porosity *would* be physical — a
  pressed/cast explosive is genuinely heterogeneous. But it **ignites at 191.8 MPa**
  (`ignition_compression = 0.98` through the EOS), *below* where a pressed powder's
  pores finish collapsing, and hands off to the detonation state machine. The
  compaction branch would act over a sliver and then be overwritten.
- **`era_filler_inert`** spalls at `damage_threshold = 0.02` and leaves the live
  set almost immediately (§3.6.2).

There is also **no zero-cost way to parameterize it**, which is worth stating
because "just default `α₀ = 1` and it is inert" understates the cost. In the P-α
framing the tabulated `density` / `youngs_modulus` are the **porous** values, so
the model needs *solid-matrix* constants that are not derivable from what
`materials.py` holds — and the porous reference density is `ρ₀₀ = ρ_s0/α₀`, so
making a filler porous changes its seeded mass. That **breaks the equal-areal-mass
A/B family** the three fillers exist to form (§3.6.2). The capability cannot be
added honestly without paying for it in the one comparison the ERA decks are for.

#### A by-product: M14's ERA ceiling is a posture, not a material property

The same arithmetic explains §3.11's tightest open number. `era_filler` designs to
`J = 0.5504` against its guard switch `0.5500` — 0.07 % clearance — because `bake`
sizes **every** material by the *deck-wide* worst contact pressure: **76.4 GPa**
(tungsten on tungsten), **6×** the **12.2 GPa** its own impedance match with the rod
would give. Sized by its own match it would design to `J = 0.6188`, clearing the
switch by 0.0688 — **170× more room**.

**Read this as confirmation, not as a defect.** §3.11 chose deck-wide sizing
deliberately and defended it: *"a CONFINED soft layer is crushed by its stiff
neighbours, not by its own (tiny) impedance. Per-material matching would hand
`era_filler` a comfortable bound precisely because it is soft — which is
backwards."* That reasoning is correct and nothing here is evidence against it.
The ~4.05 ceiling is **the price of a conservative choice**, not a constraint
`era_filler` imposes — which is exactly why the remedy §3.11 prescribes is a
per-deck `cfl_p_margin`, never per-material pressure sizing and never a bigger
global `P`.

For completeness, the one deck that *does* design inside the guard —
`apfsds_vs_nera`, at `J = 0.4828` vs `J_sw = 0.5500` — is §3.11's **named and
tested exception**, not a new finding: the override exists precisely to price that
vise, and `test_design_state_is_on_the_physical_eos_branch` asserts it.

#### What is pinned

`solver/tests/test_compaction_scoping.py` — three relations, all derived from
`materials.py` and the deck glob, none restating a literal from this section:
every deck's contact shock is a decade past any plausible `p_c`; `era_filler`
ignites below a pressed powder's crush-up; the filler's neighbours strike far
harder than its own impedance. **Each was verified to fail first**, against the
mutation that would make compaction genuinely live rather than a scrambled
constant — a 150 mm/ms deck, a lowered `ignition_compression`, a filler impedance
raised toward tungsten's — each with a control confirming the assert stays green
on the shipped configuration
([[instruments-that-cannot-see-the-failure]]). The scoping arithmetic itself is
`M:\claud_projects\temp\m15_compaction\design_state.py`; it imports the solver, so
it cannot live in `tools/` (root §3), the same way §3.11's sizing arithmetic is
cited rather than shipped.

---

### 3.13 The jet's grid resolution — and the timestep riding along with it (milestone 16)

Milestone 10 established that **cells across the jet** controls any jet *depth*
claim (§3.8), and it established that on a 150 mm RHA **half-space**, with a
standoff **ratio** as the metric. The shipped jet deck was never itself refined,
so §3.4's refusal to quote `heat_vs_composite`'s depth rested on an *inference
from a different geometry*. This milestone refines the shipped deck directly.

Three arms, one variable, everything else identical to `heat_vs_composite`:
`heat_conv_dx250` (`grid_resolution` 1200) and `heat_conv_dx188` (1600) join the
shipped 768. The jet seeds **15 / 24 / 32 particle rows** across, i.e. **7.5 / 12 /
16 cells**. Three grids, not two — two points are not a convergence study and this
repo has been bitten by that four times (§3.5, §3.6.2). No Richardson order is
extracted; §3.8 measured that order as ill-conditioned here.

#### The metric, because the obvious one is worse than blind

This stack **perforates** — the tip clears the back face at ~24 µs of a 30 µs
window. §3.8 chose a half-space precisely to avoid that, which is exactly why the
shipped composite deck went unrefined. Depth at the end of the window does not
merely saturate at the stack thickness: it reads **102–112 mm through 84.8 mm of
armor**, because past the back face the leading edge is no longer a crater bottom
but a **free residual in flight**. It is not a ceiling hiding a difference, it is a
*different quantity inventing* one — the same failure `measure_penetration.py`
warns of, where a perforated deck fits a beautiful straight line to an
unreachable number. So `tools/measure_jet_grid.py` reports the penetration front
as a **curve**, and off it the **arrival time at each interface**
(x = 160 / 190 / 215 / 235), every one of which is uncapped until the last.

**Lab time is a legitimate axis here, and that does not contradict §3.8.** That
milestone's "match on consumed fraction, *never* lab time" is a **standoff**
confound: a longer standoff impacts later, so it penetrates for less of the window
and the metric reports the *opposite sign*. These arms differ only in
`grid_resolution` — same seeding, same standoff, same virtual origin, so first
contact is the same instant. Consumed fraction is reported at every interface
anyway, and it agrees.

**The error bar is 1000× tighter than the repo's usual floor, and it was
measured.** A repeat bake of the shipped deck moves **790 475 of 1 256 472 values**
at frame 100 (`atomic_add` ordering is not deterministic) — and yet arrival times
reproduce to **≤0.0024 %** and residual velocity to **0.0037 %**. The ≤0.11 %
figure quoted elsewhere here is for *aggregates*; a **positional percentile of a
large population** is neither an aggregate nor an extremum, and it is far steadier
than either. Every difference below is 240–2000× that floor. Re-reading every
arrival at the 99.0 / 99.5 / 99.9 percentile moves it ≤0.2 %, so the choice of
front definition is not carrying the result.

#### The measurement

Arrival time (µs from t=0) at each interface, and the residual tip **4 µs after
that arm's own breakout** — matched, because the arms break out at different times
and a shared final frame would hand them different amounts of free flight:

| arm | x=160 | x=190 | x=215 | x=235 | v_resid | jet through |
|---|---|---|---|---|---|---|
| **7.5 cells, 110 substeps** (shipped) | 3.039 | 9.241 | 16.448 | 24.093 | 3208 m/s | 0.1802 |
| 12 cells, 171 substeps | 2.962 | 9.106 | 16.033 | 22.880 | 3987 | 0.1528 |
| 16 cells, 228 substeps | 3.022 | 9.299 | 16.283 | 23.028 | 4279 | 0.1457 |

Read as a ladder it is **non-monotone**: the 12-cell arm is the fastest at every
single interface, and the 16-cell arm falls back between it and the shipped one.
At 1000× the scatter floor that is not noise. It also is not a failure to
converge. It is two opposing errors mixing in a ratio that changes along the
ladder — which the ladder alone cannot show, and which is the actual finding.

#### The substep rides along with the grid, and it pushes the other way

`dt` is CFL-bound, so refining `dx` refines the **clock** with it: 110 → 171 → 228
substeps per frame. Every difference above is therefore attributable to
(`dx` **and** `dt`) jointly, and §3.5 already documented that this solver's jet-tip
state is `dt`-sensitive on its own. A grid study that never separates them is
*asserting* the attribution it should be measuring.

Two controls isolate it. `heat_conv_dt_mid` and `heat_conv_dt_fine` hold
`grid_resolution` at the shipped 768 and set the **deck `dt` below the CFL bound**,
where `min(deck_dt, cfl_dt)` simply takes it — 171 and 230 substeps at 7.5 cells,
partnering the 12- and 16-cell arms substep-for-substep. (The alternative, raising
this deck's `cfl_p_margin`, would work through the EOS design state and §3.11 is
explicit that pushing the design `J` down re-creates milestone 14's own defect.)

All deltas vs the shipped arm:

| | x=160 | x=190 | x=215 | x=235 | v_resid | through |
|---|---|---|---|---|---|---|
| **dt only**, 110→171 substeps | +1.24 % | +1.22 % | +1.70 % | +2.42 % | −7.3 % | +8.1 % |
| **dt only**, 110→230 substeps | +2.50 % | +2.44 % | +3.24 % | +4.55 % | −11.8 % | +16.8 % |
| dx+dt, 7.5c/110 → 12c/171 | −2.55 % | −1.46 % | −2.53 % | −5.03 % | +24.3 % | −15.2 % |
| dx+dt, 7.5c/110 → 16c/228 | −0.58 % | +0.63 % | −1.00 % | −4.42 % | +33.4 % | −19.2 % |

**Refining the timestep and refining the grid move every metric in OPPOSITE
directions.** A finer `dt` penetrates *later*, leaves a *slower* residual and
pushes *more* jet mass through; a finer grid does each the other way. The joint
ladder is therefore a **partial cancellation**, and it understates the grid effect
rather than measuring it. It also explains the non-monotonicity exactly: the
16-cell arm carries 228 substeps of the opposing `dt` error against the 12-cell
arm's 171, which drags its arrivals back toward the shipped value.

The dt partners are what make the `dx` effect **directly measurable**, and this is
the point of baking two of them rather than one. `heat_conv_dx250` and
`heat_conv_dt_mid` both run **150 frames × 171 substeps** at the same `frame_dt`,
so their `dt` is not merely comparable, it is *identical* — the pair differs in
`dx` and nothing else. Their difference is therefore a measurement:

| dx at matched dt | x=160 | x=190 | x=215 | x=235 | v_resid | through |
|---|---|---|---|---|---|---|
| 12 cells (`dx250` − `dt_mid`, both **171** substeps) | −3.80 % | −2.69 % | **−4.23 %** | **−7.45 %** | **+31.6 %** | **−23.3 %** |
| 16 cells (`dx188` 228 − `dt_fine` 230) | −3.08 % | −1.81 % | **−4.24 %** | **−8.97 %** | **+45.1 %** | **−36.0 %** |

The 12-cell row is exact; the 16-cell row carries a **0.9 % substep mismatch**
(228 vs 230), which is the closest the deck grid allows.

**What this still cannot reach, and it is not the additivity of the two errors.**
Two things:

- **The `dx` effect at the SHIPPED arm's 110 substeps.** Getting it would need the
  fourth cell of the 2×2 — fine `dx` at *coarse* `dt` — and that cell is
  **unreachable by construction**: `dt_sim = min(deck_dt, cfl_dt)`, so a deck may
  always refine `dt` below the CFL bound and may *never* coarsen it above one.
  (`frame_count` does not buy it: it changes frames-per-`dt`, not `dt`.) So the
  rows above are the `dx` effect measured *at 171 and at ~229 substeps*, not at
  110, and no number here extrapolates one to the shipped arm.
- **The `dx`×`dt` interaction.** Each row is measured at its own `dt`, so reading
  the two of them as a **convergence sequence** (12 → 16 cells) mixes the grid
  ladder with any interaction term. The individual rows are clean; the *trend
  between them* is not, and that is the honest limit on `−7.45 → −8.97` and
  `+31.6 → +45.1`.

**Not every cell of that table is quotable.** Re-reading the whole decomposition at
the 99.0 / 99.5 / 99.9 front percentile moves the **late** columns by ≤0.15 pp on
arrival and ≤1.2 pp on the residual — far under the effects — but the **x=160**
column swings **−4.09 / −3.80 / −2.85** (12 cells) and **−3.90 / −3.08 / −1.60**
(16 cells). At the first interface the front is a few hundred nanoseconds old and
the percentile definition dominates. **Quote x=215, breakout, residual velocity and
mass-through. Do not quote the first interface**, and treat x=190 (spread ~0.4 pp
against a 0.9 pp difference) as marginal.

#### What this settles, and what it does not

- **§3.4's refusal to quote this deck's jet depth STANDS — and now on a
  measurement of this deck rather than an inference from a half-space.** That is
  the deliverable. It is a stronger statement than the partial retirement this
  milestone was opened to attempt.
- **Nothing here is converged, and the late quantities are the worst.** Breakout
  time is the best-behaved (increment ratio 0.20) and even it is unsettled;
  residual velocity (+31.6 → +45.1 %) and mass-through (−23.3 → −36.0 %) are still
  **growing in absolute terms at 16 cells**, with increments shrinking only slowly
  (0.43, 0.55). **16 cells is not enough for the residual state**, and §3.8's own
  warning applies unchanged: "converges toward" is not "converged".
  > **✅ A THIRD RUNG AT 24 CELLS — milestone 19 (§3.16), and this bullet names two
  > quantities that turn out to disagree.** **Residual velocity STOPS GROWING**
  > (+45.1 → **+43.8 %**, the increment reversing +13.55 → −1.36 pp);
  > **mass-through does not** (−36.0 → **−44.2 %**, increments decaying at ratio
  > **0.65 over equal `dx` steps**), so for that one the verdict here holds and the
  > decay is *slower* than two points could show. Also: **the 0.20 / 0.43 / 0.55
  > above are increment-over-VALUE, not increment-over-increment** — with two rungs
  > that is the only ratio available, and it must not be read as a sequence with the
  > genuine ratios §3.16 reports.
- **How wrong the shipped arm is, quantified:** on the raw ladder it breaks out
  ~4.4 % late with a ~33 % slow residual; measured at matched `dt` the grid's own
  contribution is ~9 % and ~45 %. Either way, **the shipped 8-cell cache must not
  be quoted for breakout time or residual velocity.** Its kinematic claims (§3.4)
  are untouched — they are measured Lagrangianly on free-flight markers, which do
  not lean on grid coupling.
- **A residue, stated rather than smoothed — and one part of it withdrawn.** The
  two *early* interfaces move back *toward* the shipped value (−3.80 → −3.08,
  −2.69 → −1.81) where the late ones grow. At **x=160 that is not resolvable**: the
  percentile sweep above moves the same cells by more than the effect, so it is not
  claimed. At **x=190** the difference is about twice its sensitivity and survives,
  and the candidate mechanism is the one thing the design cannot reach — a
  **`dx`×`dt` interaction**, since the two rows are measured at different `dt`.
  Named, not demonstrated.

#### The transferable lesson, and a prediction it makes about §3.8

> **A CFL-coupled refinement study measures the DIFFERENCE OF TWO OPPOSING ERRORS,
> not the grid error. The fix is a substep-matched dt partner per arm — not more
> grids, which only add points to the same confounded ladder.**

That applies retroactively to §3.8's own ladder, and it **predicts the one
discrepancy that section left open**. §3.8 has two routes to 16 cells across the
jet and they disagree by 5 %: refining `dx` gives **1.429**, fattening the jet to
6 mm at the shipped `dx` gives **1.501**, nearer the a-priori 1.536. Those two
routes differ in exactly this variable. `_impact_pressure` and `_eos_equilibrium_j`
take material names and `v_tip` and **never the diameter**, and
`standoff_conv_d6mm_s00` carries `grid_resolution: 1440` — *identical* to
`standoff_s00`. So **the fat-jet route is the `dt`-free route**, while the `dx`
route halves the substep alongside the cell. The sign matches: a finer `dt`
suppresses penetration here, and the `dx` route is the one reading low.

**Not tested — M10's territory, and re-baking it would re-roll a closed
milestone's numbers.** The experiment is specified: bake `standoff_s00`/`s90` at
`grid_resolution: 1440` with a deck `dt` set to the `standoff_conv_dx250` arms'
substep count, and re-run `tools/measure_standoff.py`. It is **falsified** if that
dt-only pair reproduces the shipped 1.229 (the gap is then not about `dt`), and
supported if it moves the ratio *down* from 1.229, in the direction the `dx` route
is dragged.

> **✅ TESTED — milestone 17 (§3.14), and the answer is *partly*. Three corrections
> to the paragraphs above, which are kept as M16 wrote them.**
> * **The figures are Murnaghan-era.** §3.8's table had not been re-read through
>   M13 or M14 when this was written; today the routes are **1.4968** and
>   **1.5587** and the gap is **4.13 %**, not 5 %. The shipped arm is **1.2657**,
>   not 1.229.
> * **The substep target named here is the wrong arm.** dx250 is a **1.5×** `dt`
>   refinement partnering 12 cells; the gap lives between dx188 (**2×**) and the fat
>   jet. Both partners were baked — and the second point mattered, because the `dt`
>   term turns out to **saturate** by 513 substeps.
> * **The falsifier is written on the RATIO alone, and that was the real trap.** A
>   finer `dt` suppresses both arms' *depth* 3–6 % while moving their quotient only
>   1.6–1.9 %, so a ratio-only reading understates the effect ~3×. Neither branch of
>   the falsifier was reached: `dt` accounts for **~39 %** of the gap and **~61 % is
>   not `dt`** — supported in sign, falsified as a complete explanation.

#### A by-product: how M14's CFL budget behaves under refinement

Four arms, one design `c_max = 64101 mm/ms` on every one of them — confirming
`bake`'s own comment that the artificial-viscosity contribution to the bound is
`dx`-independent. Measured `c_eff` as a fraction of that budget:

| | 7.5 cells | 12 cells | 16 cells |
|---|---|---|---|
| along the dx ladder | 63 % | 66 % | 73 % |
| at fixed dx, refining dt (110 / 171 / 230) | 63 % | 59 % | 57 % |

**Opposite signs again.** This is a previously-unmeasured property of
`EOS_CFL_P_MARGIN`: further **spatial** refinement is what would eventually breach
it, and temporal refinement buys headroom. Nothing here breached — the tally is
57–73 %, all four arms clean at the shipped P=4 — so no per-deck `cfl_p_margin` was
needed, and §3.11's rule stands that more headroom means a per-deck override and
**never** a bigger global P.

#### What is pinned

`solver/tests/test_jet_grid.py` — ten contract tests on `measure_jet_grid.py`,
each pairing a reading with the defect that same assertion must catch: a gap-only
interface rule that silently loses the **bonded** ceramic/steel contact at x=215; a
manifest whose `armor` provenance block **lies**; an arrival that clamps to the
window end instead of reading NOT REACHED; a matched residual that silently falls
back to the final frame; the seeded-lattice reading (15 rows is 7.5 cells, not the
7.68 `domain/grid_resolution` implies, because `_fill_rect` rounds each object's
lattice to fit it); and **the tool's own reason to exist** — a synthetic pair that
penetrates at visibly different rates but ends the window in the same place, which
`depth_end` calls identical and arrival time separates by 40 %. **Six mutations
were verified RED first** ([[instruments-that-cannot-see-the-failure]]); the
harnesses are `M:\claud_projects\temp\m16\red_check.py` and `red_check2.py`, which
mutate the tool in memory and are cited rather than shipped.

---

### 3.14 Separating `dx` from the clock CFL drags along with it (milestone 17)

§3.13 closed with a prediction and an experiment written down but untested: §3.8's
two routes to 16 cells across the jet disagree, and the reason should be that
**refining `dx` refines the substep too** while fattening the jet does not. This
milestone built the missing controls and measured it. **Zero kernel code** — four
decks (`standoff_conv_dt513_s00/s90`, `standoff_conv_dt684_s00/s90`), a
`--dt-decomposition` mode on the existing tool, and one host-side refactor.

#### The premise, verified before anything was baked

`bake` sizes `dt = min(deck_dt, cfl_dt)`, and `c_max` carries **no `dx`
dependence** — the artificial-viscosity contribution is `c_q·v_tip` (§3.9), so
`dt_cfl ∝ dx` exactly. Replaying that sizing over the whole standoff family, on the
host with no GPU:

| arm | grid | `dx` | substeps | `dt` (ms) | bound by |
|---|---|---|---|---|---|
| `standoff_s00/s90` — shipped | 1440 | 0.3750 | 114 | 1.754386e-6 | CFL |
| `standoff_conv_d6mm_*` | 1440 | 0.3750 | 342 | **1.754386e-6** | CFL |
| `standoff_conv_dx250_*` | 2160 | 0.2500 | 513 | 1.169591e-6 | CFL |
| `standoff_conv_dx188_*` | 2880 | 0.1875 | 684 | 8.771930e-7 | CFL |
| **`standoff_conv_dt513_*`** | **1440** | **0.3750** | **513** | **1.169591e-6** | **deck** |
| **`standoff_conv_dt684_*`** | **1440** | **0.3750** | **684** | **8.771930e-7** | **deck** |

**The fat-jet route really is `dt`-free** — bit-identical to the shipped arm's
substep, not merely close — and each new partner is bit-identical to the `dx` arm it
pairs with, so **that pair differs in `dx` and in nothing else**. §3.13 asserted the
first of those by reading the code; this is it through the real sizing path,
including the AV term and the frame-cadence `ceil`.

Two consequences worth stating separately. The partners are isolated by the **deck
`dt`**, never by `cfl_p_margin` — §3.11 forbids pushing the design `J` down, since
that re-creates M14's own defect. And the `ceil` window is narrow: any deck `dt` in
`(1.16959e-6, 1.171875e-6]` ms lands on 513, and one that lands on 514 turns a
measured pair back into an inference. **Verify the substep count before baking**,
which `mpm.plan_substeps` now makes possible without a GPU.

#### The headline: a CFL-coupled ladder understates the grid effect

This is the result that generalizes, and it is not the question the milestone was
opened to answer.

| effect | measured | the joint ladder reads |
|---|---|---|
| `dx` alone, 8 → 12 cells (at 513 substeps) | **+17.46 %** | +15.3 % |
| `dx` alone, 8 → 16 cells (at 684 substeps) | **+20.27 %** | +18.4 % |

The confounded ladder **understates the grid effect by ~1.9 pp**, because the `dt`
refinement it drags along pushes the other way and partially cancels it. That is
§3.13's transferable rule — *a CFL-coupled refinement study measures the difference
of two opposing errors* — **reproducing on a second, independent family**: a
different deck, a different target, a different geometry, and a ratio metric instead
of arrival times. M16 could only assert it from one family; it is now a property of
the method rather than of `heat_vs_composite`.

#### The `dt` term saturates, and may have turned over

| substeps at the shipped `dx` | 114 | 513 | 684 |
|---|---|---|---|
| mean S90/S0 | 1.2643 | 1.2407 | 1.2445 |
| vs the shipped arm | — | **−1.87 %** | **−1.56 %** |

Refining the clock alone **suppresses the ratio**, the direction §3.13 predicted.
But the effect is **saturated by 513 substeps**: going 1.5× finer again moves it
back **+0.31 %**, which is only 1.5× the 0.2 % resolution floor. So *saturated* is
claimed and *turned over* is **named, not claimed** — the same posture §3.13 took at
x=160. This is why the second partner was baked despite §3.13's spec asking only for
one: **with a single point there is no way to tell a slope from a plateau.**

#### The ratio hides most of what `dt` does — a caveat §3.8 never carried

| | S=0 depth at f=0.30 | S=90 depth at f=0.30 |
|---|---|---|
| 114 substeps (shipped) | 46.00 mm | 58.03 mm |
| 513 substeps | 44.59 mm (−3.1 %) | 55.59 mm (−4.2 %) |
| 684 substeps | 43.47 mm (−5.5 %) | 54.70 mm (−5.7 %) |

**A finer substep suppresses both arms' depth by 3–6 %, while their quotient moves
only 1.6–1.9 %.** The ratio is a partial common-mode cancellation, so §3.8's
headline metric is roughly **3× less `dt`-sensitive than the quantity underneath
it**. §3.13's falsifier was written on the ratio alone, and a ratio-only instrument
would have reported "no `dt` effect" for a solver that moved every depth it
measures. `test_standoff_dt.py` pins the general form with a synthetic pair whose
depths both move 8 % while the ratio moves **0.0000 %**.

Anyone quoting a standoff **depth** from this family inherits that sensitivity;
anyone quoting the **ratio** mostly does not. Said in §3.8 as well as here.

#### The verdict on §3.13's prediction: right in sign, insufficient in magnitude

- the two 16-cell routes disagree by **+4.13 %** (dx188 1.4968 vs fat jet 1.5587)
- the `dt`-only term at the gap's own 684 substeps is **−1.56 %**
- undoing it puts dx188 at **1.5206**, leaving a residual gap of **+2.50 %**

So **the timestep accounts for ~39 % of the disagreement and ~61 % is not `dt`.**
§3.13's prediction is **supported in sign and falsified as a complete explanation** —
which is a more useful outcome than either branch of the falsifier it wrote down
("reproduces 1.229 ⇒ not about dt" / "moves down ⇒ supported"), because the answer
was *partly*.

**The 39/61 split rests on an assumption this design cannot test**, and that is
stated rather than buried: it transfers a `dt` term measured at **8 cells** onto the
**16-cell** arm, i.e. it assumes no `dx`×`dt` interaction — the one term §3.13 named
as out of reach, for the same structural reason it still is (`dt = min(deck_dt,
cfl_dt)` only ever *lowers* `dt`, so *fine `dx` at coarse `dt`* is unreachable). Read
the split as an estimate with a known open term, not a closed account.

**What the residual 61 % might be** is now the interesting question, and §3.8's own
answer becomes the leading candidate by elimination: it called a real finite-diameter
effect "**mostly** excluded", and this milestone says the hedge in that word was
doing real work. A 6 mm jet is not a scaled 3 mm jet in a solver with an absolute
grid, a fixed damage threshold, and a fixed window.

> **✅ TESTED at M18 (§3.15), and the candidate is NOT SUPPORTED at that resolution.**
> The same route comparison one scale coarser reads **−4.10 % to −12.28 %** against
> this **+2.50 %**: the route difference changes sign *and* size with resolution,
> which is the signature of discretization error rather than of a physical offset.
> **Not a refutation** — both M18 arms are coarser than both arms here, and a
> numerically-dominated coarse pair cannot overturn a finer measurement. The last
> sentence above is the part that survives, and M18 turned it into a measurement:
> a 6 mm jet is not a scaled 3 mm jet, by **4–12 %**.
>
> **⚠️ THE +2.50 % IS A MEAN, AND M19 (§3.16) MEASURED WHAT IT IS A MEAN OF: the
> per-fraction residual runs −3.66 % → +8.67 % across the matching window and
> changes sign inside it.** Nothing above is retracted — this section's whole
> decomposition is on means, consistently — but **M18 compared its per-fraction
> f=0.30 figure against this mean**, which is two statistics. Same-statistic at
> f=0.30 the 16-cell reading is **+8.67 %**, so M18's conclusion is *reinforced*,
> not weakened. Quote the mean against means and the per-fraction figure against
> per-fraction figures; §3.16 carries the matched table.

#### One code change, and why it is not "zero code"

`bake`'s inline CFL sizing became **`mpm.plan_substeps(scenario)`**, which `bake`
now calls — a pure host-side extraction, no kernel touched. It exists because the
claim "these two decks run the same `dt`" can only be pinned against the *real*
sizing path: re-deriving `ceil(frame_dt / min(deck, cfl))` in a test would be
satisfied by copying a bug, the mistake `test_cfl_sizing` was written to avoid. It
also lets a deck be checked before an hour of GPU:

```bash
python -m ballistics_solver.run scenarios/standoff_conv_dt684_s00.yaml --out /dev/null --dry-run
# [dry-run] standoff_conv_dt684_s00: grid 1440 (dx=0.3750 mm), 75 frames x 684 substeps
# [dry-run] dt=8.771930e-07 ms, bound by the DECK (deck 8.780e-07, CFL 1.755e-06 ms)
```

The pairing is legible from that alone — `standoff_conv_dx188_s00` prints the *same*
`dt` and substep count from a different grid, bound by CFL where the partner is bound
by its deck. Verified behaviour-preserving by re-sizing **all 38 decks** and matching
every runtime-printed substep count, including M16's documented 110 / 171 / 228 / 230.
**The caches did not move**: no rebake, and no figure outside §3.8 changed.

#### A by-product: the CFL budget under temporal refinement

All four new arms audit clean at P=4 — **28–34 % of the `c_max=64101 mm/ms`
budget**, no J-floor or resolution-guard fires. Consistent with §3.13's by-product
table: temporal refinement *buys* headroom where spatial refinement spends it, so no
per-deck `cfl_p_margin` was needed and §3.11's rule stands.

#### What is pinned

`solver/tests/test_standoff_dt.py` — 20 tests split by what can check what. The deck
pairing goes through `plan_substeps` because **no cache can check it**
(CACHE_FORMAT §2 records `frame_dt`, never `dt` or a substep count). The tool's
readings are pinned on synthetic caches, each against the defect the same assertion
must catch: `cells across the jet` read off the seeded lattice must survive a
manifest that **lies** about the diameter (§2.1 — provenance, not data); a stride
that silently no-ops; and the mode's reason to exist, the ratio that cannot see an
8 % move in every depth beneath it. **Seven mutations were verified RED first**
([[instruments-that-cannot-see-the-failure]]); the harness is
`M:\claud_projects\temp\m17\red_check.py`, cited rather than shipped, as in §3.13.

One fixture lesson worth carrying: the first draft's synthetic jet had 16 particles,
which made `consumed` a nine-step staircase, and interpolating a staircase is
violently sensitive to which frames a stride keeps. It read exactly like a defect in
the tool. **A fixture too thin to be smooth manufactures the failure it is testing
for.**

### 3.15 The diameter, and the scale invariance §3.8 assumed (milestone 18)

§3.14 left the residual +2.50 % with one named candidate — §3.8's own hedge, "a real
finite-diameter effect, **mostly** excluded" — and no deck able to test it, because
the fat-jet route to 16 cells changes the **cell count** and the **diameter**
together. This milestone built the arm that unpicks them. **Zero kernel code**: two
decks (`standoff_conv_d6mm_dx750_s00/s90`), a `--diameter-decomposition` mode, and
one helper extracted in the same tool.

The new arm is a **6 mm jet at `dx=0.75`** — the fat arm's diameter at the shipped
arm's 8 cells — with the deck `dt` pinned so all three arms run
`dt = 1.754386e-6 ms` bit-identically. Verified through `plan_substeps` before any
GPU (§3.14's gate): the pin is load-bearing, since at `dx=0.75` the CFL bound is
`3.5088e-6 ms` and an unpinned arm would have run **twice** the shipped substep.
`particles_per_cell=4` also puts **16 particles across** this 6 mm jet, exactly as
the shipped 3 mm jet is 16 across at `dx=0.375`, so particle resolution is matched
and not a third variable. Both arms bake at 129 920 particles, audit clean at P=4
(**43 % / 38 %** of the `c_max=64101 mm/ms` budget), J-floor and resolution guard
zero.

#### The headline: the S=0 arm barely notices either knob

| depth at f=0.30, vs the shipped arm | S=0 | S=90 |
|---|---|---|
| shipped — 3 mm, `dx=0.3750`, 8 cells | 46.00 mm | 58.03 mm |
| fat jet — 6 mm, `dx=0.3750`, 16 cells | 44.53 mm (**−3.19 %**) | 71.17 mm (**+22.65 %**) |
| coarse fat — 6 mm, `dx=0.7500`, 8 cells | 43.00 mm (**−6.52 %**) | 52.03 mm (**−10.35 %**) |

**Both knobs are spent almost entirely on the S=90 arm.** The S=0 depths span 7 %
across three configurations that differ by a factor of two in grid and in diameter;
the S=90 depths span 37 %. That is a physical reading rather than a numerical one:
the S=90 jet flies 90 mm and **thins** before it arrives, so it is the arm whose
resolution is actually marginal, and the S90/S0 quotient is close to an S=90
measurement wearing a ratio's clothes. §3.14 said report both arms and never only
the quotient; here that is not a caveat, it is the mechanism.

#### The three reads, none of them `dt`-confounded

| | mean S90/S0 | |
|---|---|---|
| `dx` alone at 6 mm, 0.7500 → 0.3750 (8 → 16 cells) | 1.1740 → 1.5587 | **+32.77 %** |
| diameter alone at `dx=0.3750`, 3 → 6 mm (8 → 16 cells) | 1.2643 → 1.5587 | **+23.29 %** |
| **SCALE**, both doubled at **fixed 8 cells** | 1.2643 → 1.1740 | **−7.14 %** |

All three arms share a bit-identical substep, so no row needs a correction —
**§3.14's two-route comparison could not say that of itself**, and its residual rests
on transferring a `dt` term measured at 8 cells onto the 16-cell arm. That asymmetry
in trustworthiness is a real by-product of this design and is why the third row is
worth more than its size suggests.

> **Why the shipped arm reads 1.2643 here and 1.2657 in §3.8's table.** Same two
> caches; this mode **decimates the shipped arm 225 → 75 frames** to put it on the
> diagnostic arms' cadence, and that costs **−0.113 %** (§3.14 measured it). It is not
> a stale table and not a rebake — both figures are current. The effects below are
> 20–100× that, so nothing turns on which is quoted, but **do not read the two as a
> disagreement.**

The **16-particles-across** claim above is verified from the seeded lattice on the
real caches, not from `diameter / (dx/2)`: `standoff_s00` and
`standoff_conv_d6mm_dx750_s00` both seed **16 distinct jet rows** at pitch 0.1875 and
0.3750 mm — exactly 2× — where `standoff_conv_d6mm_s00` seeds 32. `_fill_rect` rounds
each object's lattice to fit and §3.13 already caught that arithmetic disagreeing with
the seed once, so a claim asserted in three documents was checked against the bytes.

#### The scale row is the test, and §3.8's claim is what it tests

`cells across the jet` is **not an independent variable — it is the ratio
`diameter/dx`**. So §3.8's "cells across the jet is the controlling parameter" is a
claim that the response depends on that ratio **alone**, i.e. a **scale-invariance
hypothesis**. It is testable and it had never been tested: the two 8-cell arms are
the *same discretization scaled by 2×* against physics that does not scale with it —
the 90 mm standoff, the 150 mm plate, `damage_threshold`, the process zone. If the
hypothesis held, the scale row would read 0.00 %.

| scale row, per matched fraction | f=0.15 | f=0.20 | f=0.25 | f=0.30 |
|---|---|---|---|---|
| coarse fat vs shipped | −12.28 % | −6.75 % | −5.49 % | **−4.10 %** |

**It varies 3× across the window and is not one number.** **Quote −4.10 %, at
f=0.30** — the most-consumed, most-converged end — with the range in support. The
f=0.15 end is both the largest and the least trustworthy, for the reason §3.13 gave
at x=160: the earliest matching point is where the decomposed cells swing most. Even
the conservative figure is **20× the 0.2 % floor**. This is §3.13's x=160
lesson applied rather than restated: a mean over a quantity that is not constant
reports one figure with no hint that the figure depends on where you stood, so
`route_difference` is per-fraction by construction. §3.8's claim therefore survives
as an **approximation, not an identity** — good to a few percent, and the residual
scale dependence is of the same order as the two-route disagreement it was invoked to
explain.

#### The reading on §3.14's candidate — weaker than it looks, and better

The same route comparison one scale coarser reads **−4.10 %** (f=0.30; −12.28 % at
f=0.15) where §3.14's dt-corrected residual reads **+2.50 %**. It **changes sign
and changes size with resolution**, and a physical finite-diameter effect would be a
roughly resolution-**independent** offset. So this behaves like discretization error —
which *characterizes* the residual rather than merely ruling a candidate out.

**It does not refute the fine-pair reading, and must not be written as if it did.**
Both arms here (`dx` 0.375 and 0.75) are coarser than both arms there (0.1875 and
0.375), and a numerically-dominated coarse pair cannot overturn a measurement made at
finer resolution. The honest statement is that the finite-diameter candidate is
**not supported at this resolution**, not that it is falsified.

#### What this does not settle

- **Two points are not a trend.** There are exactly two route-difference readings, at
  8 and at 16 cells. Do not read a crossing between them, do not say where it would
  vanish, do not extrapolate. This repo has been bitten three times by exactly that
  ([[convergence-claims-need-real-evidence]]), and a sign flip between two points is
  the most inviting version of it.
  > **✅ A THIRD READING AT 12 CELLS — milestone 19 (§3.16): −4.48 % at f=0.30,
  > dt-free.** Three points on a **two-factor** grid are still not a trend, and the
  > refusals above stand verbatim. What the third point *does* settle is this
  > section's own confound: the 12-cell row is a **1.5× scale row** where these two
  > are 2×, and a factor-driven violation would then be *smaller* — measured
  > 4.48 % against 4.10 %, it is not. **The scale factor is not what sets the
  > magnitude.** The 16-cell row remains the outlier, and whether that is resolution
  > or a `dt` correction that does not transfer is still unreachable.
- **Diameter-at-fixed-cells is inseparable in principle.** `cells ≡ diameter/dx`, so
  holding the ratio and moving one factor moves the other: the scale row is
  *(diameter + dx-at-fixed-cells)* and no deck can make it anything else. The two
  actually-free variables are `diameter` and `dx`; `cells` is their quotient and was
  never a knob.
- §3.14's `dx`-only rows (+17.46 %, +20.27 %) are at 3 mm and 684 substeps where the
  row above is at 6 mm and the shipped clock, so comparing **slopes** still carries
  the `dx`×`dt` interaction — the same open term §3.14 declared, not a new one.

#### What is pinned

`solver/tests/test_diameter_scale.py` — 14 tests. The deck design goes through
`plan_substeps` because no cache can check it (CACHE_FORMAT §2 records `frame_dt`
only), and the assertions state **relations between decks** rather than re-deriving
the sizing arithmetic, which a copied bug would satisfy. The load-bearing one asserts
the asymmetry directly: the scale move scales `dx` and `diameter` by 2×, and the
standoff, thickness, material, jet length, velocities and domain are **identical**.

Instrument-side, `_dt_residual` **re-derives** §3.14's published split from the caches
rather than quoting it (§3.8 went two rebakes stale for exactly that reason) and keys
`DT_ARMS` **by name** — `_dt_decomposition` indexes the same list positionally, so a
reordering is a live hazard that would re-key the published decomposition and stay
green. **Seven mutations verified RED first**; harness at
`M:\claud_projects\temp\m18\red_check.py`, cited rather than shipped, as in §3.13.
> **✅ HAZARD REMOVED at M19 (§3.16), not deferred a second time.** Every arm table
> carries a stable `.key` and every lookup goes through it, so the lists may now be
> sorted or extended freely; a test shuffles all three and asserts this section's
> split is unchanged.

One harness lesson worth carrying, the same shape as the defects being hunted: the
`particles_per_cell` mutation first came back **green** because a `replace(…, 1)` hit
the deck header's prose quoting `particles_per_cell: 4` rather than the YAML line.
**A mutation that does not land reads exactly like a test that does not care.**

---

### 3.16 The third route point, and the 24-cell rung (milestone 19)

Two open items, both left as *named experiments* rather than conclusions. §3.15
closed with "two points are not a trend" and specified the third — a 4.5 mm jet at
`dx=0.375` pinned to `standoff_conv_dx250`'s substep, 12 cells by both routes. §3.13
closed with "16 cells is NOT enough for the residual state" and had only two
matched-`dt` rungs to say it with. This milestone built both. **Zero kernel code**:
four decks (`standoff_conv_d4p5mm_dt513_s00/s90`, `heat_conv_dx125`,
`heat_conv_dt342`), a `--route-difference` mode, a `--dt-ladder` mode, and the
de-positionalisation §3.15 flagged as a live hazard.

All four sized through `plan_substeps` before any GPU (§3.14's gate) and all four
audit clean at P=4.

#### The 12-cell route pair, and why it needs no correction

`standoff_conv_d4p5mm_dt513_*` is a 4.5 mm jet at the **shipped** `dx=0.3750`, so
12 cells by the diameter route, with the deck `dt` pinned to
`dt = 1.169591e-6 ms` — **bit-identical to `standoff_conv_dx250_*`**, which reaches
12 cells by refining `dx` to 0.2500. The pin is load-bearing: unpinned this arm is
CFL-bound at 342 substeps, and the row would have needed exactly the transferred
correction §3.14's 16-cell row still needs. `particles_per_cell=4` puts **24 rows
across the jet on both arms** (pitch 0.1875 vs 0.1250, exactly 1.5×), so particle
resolution is matched and is not a third variable — the same match §3.15 made at 16
rows. Both arms bake at 520 960 particles, **38 % / 30 %** of the `c_max=64101 mm/ms`
budget, J-floor and resolution guard zero.

#### The headline is a correction to how the older figures compare

| f=0.30, fat-jet route against fine-`dx` route | route difference |
|---|---|
| 8 cells — 3 mm/0.3750 vs 6 mm/0.7500 (dt-free) | **−4.10 %** |
| **12 cells — 3 mm/0.2500 vs 4.5 mm/0.3750 (dt-free)** | **−4.48 %** |
| 16 cells — 3 mm/0.1875 vs 6 mm/0.3750 (dt-corrected) | **+8.67 %** |

That last cell is the correction. **§3.14's residual is +2.50 % as a MEAN, and the
quantity it means over runs −3.66 % → +8.67 % across the matching window, changing
sign inside it.** §3.15 compared its own *per-fraction* f=0.30 figure (−4.10 %) to
that *mean* (+2.50 %) — two different statistics. On the same statistic the
16-cell reading is **+8.67 %**, so §3.15's sign-change finding is not weakened by
the correction; it is roughly **3.5× larger** than the comparison it was built on.
Neither figure is wrong. They answer different questions, and §3.15's own rule —
quote per fraction at a stated fraction — was applied to its row and not to the row
it compared against. **Apply a rule to both sides of a comparison or it is not a
comparison.**

#### The scale-factor confound makes a prediction, and it fails

`cells ≡ diameter/dx`, so reaching *N* cells from the shipped 8 moves each factor by
`N/8`: the 12-cell row is a **1.5× scale row** where the 8- and 16-cell rows are
**2×**. That is not a design choice — no deck can make it otherwise — and it is a
real limit on reading the three as a trend, since a smaller perturbation is
generically a smaller violation.

It is also testable. A violation whose size is set by the scale separation predicts
**|12-cell| < |8-cell|**. Measured: **4.48 % vs 4.10 %** — *not* smaller. Across the
whole window the two dt-free rows differ by 0.36–3.86 pp, the largest of that at
f=0.15, the end §3.15 says is least trustworthy. So the scale factor is **not what
sets the magnitude**, and the third point is not an artifact of being a 1.5× row.

What it still cannot say is **which** of two things makes 16 cells the outlier: a
genuine resolution dependence appearing between 12 and 16 cells, or a `dt`
correction that does not transfer. The 16-cell row is an outlier **before** any
correction (+8.94 % raw), so the second would need a `dt` term at 16 cells far larger
than the one measured at 8 — which is precisely the `dx`×`dt` interaction §3.13 named
as out of reach and this design does not reach either. **Still no crossing, no
zero-point, no extrapolation, no order.** Three points on a two-factor grid are not a
trend.

#### The 24-cell rung splits §3.13's "residual state" in two

`heat_conv_dx125` (grid 2400, `dx=0.1250`, 48 seeded rows = **24 cells**, 1 774 720
particles) with `heat_conv_dt342` at the shipped grid — the family's **first exact
`dt` pair**, 342 vs 342, where §3.13's 16-cell rung carries a 0.9 % mismatch (228 vs
230) that was the closest that deck grid allowed.

Each row is the `dx` arm **minus** its `dt` partner, as a percentage **of the shipped
arm** — §3.13's convention, which is what keeps the decomposition additive. The 12-
and 16-cell rows reproduce §3.13's published table exactly; that is the mode's own
regression check.

| cells | substeps | x=215 | breakout x=235 | v_resid | mass-through |
|---|---|---|---|---|---|
| 12 | 171 vs 171 | −4.23 % | −7.45 % | +31.6 % | −23.3 % |
| 16 | 228 vs 230 | −4.24 % | −8.97 % | +45.1 % | −36.0 % |
| **24** | **342 vs 342** | **−5.29 %** | **−9.51 %** | **+43.8 %** | **−44.2 %** |

The rungs are **equally spaced in `dx`** (0.2500 / 0.1875 / 0.1250 — two steps of
0.0625 mm), which is what makes an increment-to-increment ratio readable: over equal
steps a first-order error gives 1.00.

| | increment 12→16 | increment 16→24 | ratio |
|---|---|---|---|
| breakout | −1.52 pp | −0.54 pp | 0.35 |
| residual velocity | +13.55 pp | **−1.36 pp** | 0.10, **sign flip** |
| mass-through | −12.69 pp | −8.24 pp | **0.65** |

**§3.13 named two quantities in one phrase and they do not agree.**

- **Residual velocity has stopped growing.** §3.13 measured it "still growing in
  absolute terms at 16 cells"; the third rung reverses the increment. *Stopped
  growing* is claimed and *turned over* is **named, not claimed** — one reversal is
  one point, the posture §3.14 took on its own saturating `dt` term.
- **Mass-through has not.** Its increments decay at ratio **0.65 over equal `dx`
  steps**, so §3.13's verdict holds at 24 cells and the decay is *slower* than its
  two points could show. **Do not quote this quantity at any resolution in this
  repo.**
- **x=215 is not readable as a ladder at all**: the 12- and 16-cell rungs land on top
  of each other (−4.23, −4.24) and the ratio is withheld, because a near-zero
  denominator is not a small ratio, it is noise amplified. §3.13 said quote x=215;
  that stands for a *single* comparison and not for an increment sequence.

**So "the residual state" was one phrase over two quantities that behave
differently, and two rungs could not tell them apart.** That is the rung's
deliverable — not a convergence claim for either of them, and §3.4's refusal to
quote this deck's jet depth is untouched.

#### The sharpest demonstration of §3.13's rule this repo has

Put the confounded ladder (`--family`, four arms now) beside the decomposed one, both
at breakout, both against the shipped arm:

| cells across | 7.5 | 12 | 16 | **24** |
|---|---|---|---|---|
| joint `dx`+`dt` ladder | — | −5.03 % | −4.42 % | **−1.74 %** |
| `dx` alone, at matched `dt` | — | −7.45 % | −8.97 % | **−9.51 %** |

**They run in opposite directions.** A reader of the confounded ladder alone watches
the effect shrink toward zero and would call it converged — while the grid's own
contribution is *growing* the whole way, and what shrinks is the difference between
it and a `dt` error growing faster. §3.13 stated this rule from two rungs and §3.14
reproduced it on a second family; four rungs make it visible as a picture rather than
an argument. **A CFL-coupled ladder does not merely understate the grid effect — at
enough refinement it can report the opposite sign of its trend.**

#### Two by-product ladders extend, and one is approaching a limit

§3.13's CFL-budget observation continues on both axes, at the shipped P=4:

| | 7.5 cells | 12 | 16 | **24** |
|---|---|---|---|---|
| along the `dx` ladder | 63 % | 66 % | 73 % | **84 %** |
| at fixed `dx`, refining `dt` (110 / 171 / 230 / **342**) | 63 % | 59 % | 57 % | **53 %** |

**Opposite signs, confirmed on a third point each.** Spatial refinement spends
`EOS_CFL_P_MARGIN`'s headroom and temporal refinement buys it. Nothing breached — but
**84 % is the highest budget use any deck in this repo has recorded**, and on this
trend a 32-cell arm is where it would go. §3.11's rule is unchanged and becomes
practical rather than hypothetical: more headroom means a **per-deck
`cfl_p_margin`**, never a bigger global P, which has a hard ceiling near 4.05
(§3.11).

#### One window is longer, deliberately, and the tool now refuses to average over it

`heat_conv_dt342` records for **34 µs** where every other arm records 30. A finer
`dt` breaks out *later* (§3.13: +2.42 % at 171 substeps, +4.55 % at 230), which
extrapolates to ~26 µs at 342 against a window that must still contain
breakout + 4 µs for the matched residual. That margin was ~0 or negative, and a
missed residual would have cost the deck its entire reason to exist. Sized before
baking rather than discovered after.

**Its `dx` partner was NOT extended to match, and that asymmetry is the right one.**
The two arms fail in opposite directions: `dx125` breaks out *earlier* and its
residual is the *fast* one, so 34 µs would fly it to x ≈ 295 mm against the domain's
x=300 wall — retiring §1.1.2's clean negative that nothing in this repo has ever
reached `x_hi`, in exchange for symmetry in a number nobody should read. Instead
`measure_jet_grid.py` now carries `window_us` and **refuses to compare `depth_end`
across arms that do not share a window**. Every quantity it does report is read off
the penetration-front curve or at a matched time after each arm's own breakout, and
neither depends on the recording length. **A window is a recording length as long as
`frame_dt` is untouched — and `frame_dt` is what the test asserts.**

#### What is pinned

`solver/tests/test_route_difference.py` — 27 tests, split the usual way by what can
check what. The deck pairings go through `plan_substeps` because no cache can check
them (CACHE_FORMAT §2 records `frame_dt` only), and every assertion states a
**relation between decks** rather than re-deriving the sizing arithmetic, which a
copied bug would satisfy.

Three are worth naming:

- **The `ceil` band is measured, not assumed.** The band that lands on 342 substeps
  is found by **bisecting the real sizing path** — re-deriving `ceil(frame_dt/dt)` in
  a test is the mistake `test_cfl_sizing` exists to avoid. It is **0.29 % wide**, and
  the deck's `5.856e-10` sits at 47 % of it. The first draft of that test assumed a
  0.5 % nudge would stay inside and went red; the band is narrower than §3.14's
  warning implied.
- **The rung delta's denominator is pinned.** Normalising by the partner rather than
  the shipped arm gives +34.1 % where §3.13 published +31.6 % — the same measurement,
  two figures. That defect was in the first draft of `--dt-ladder` and was caught by
  reconciling against §3.13 rather than by the test; the test exists so it cannot
  come back.
- **Equal `dx` spacing is asserted**, because the printed increment ratio is only
  readable over equal steps and a future rung chosen for a round cell count would
  silently break it.

**Instrument-side, milestone 18's named hazard is removed rather than deferred
again.** Every arm table (`DT_ARMS`, `DIAM_ARMS`, the new `ROUTE_ARMS`, and
`--convergence`'s `cfg`) now carries a stable `.key` and every lookup goes through
it; the test shuffles all three and asserts §3.14's published split is unchanged.
`--route-difference` also **computes** dt-matching from the arms' own `dt_ms` rather
than reading a hand-written flag, so a row cannot be mislabelled "matched" and
present a confounded comparison as a measurement.

**Seventeen mutations verified RED first** ([[instruments-that-cannot-see-the-failure]]);
the harness is `M:\claud_projects\temp\m19\red_check.py`, cited rather than shipped,
as in §3.13, §3.14 and §3.15. It checks the tree is restored afterwards, and it
refuses a mutation whose pattern does not occur exactly once — §3.15's lesson that a
mutation which does not land reads exactly like a test that does not care.

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
- **Porous compaction (P-α):** Herrmann, *"Constitutive equation for the dynamic
  compaction of ductile porous materials"* (1969); Carroll & Holt, *"Static and
  dynamic pore-collapse relations for ductile porous materials"* (1972). Cited
  even though **nothing was built from them** — §3.12 closes milestone 15 as a
  negative, and the argument that pore collapse is inert here rests on what these
  two papers say a crush-up curve *is*. A reason not to build something is a
  public-physics claim like any other and gets a citation like any other.

Nothing classified or export-controlled enters this repo; public physics is the
ceiling here by construction (CLAUDE.md §10).
