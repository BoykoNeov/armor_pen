"""MLS-MPM transfer kernels (NVIDIA Warp) — the physics core.

STATUS: milestone 5 — **elastic + von Mises plasticity + ductile & brittle damage
+ multi-material armor stack + reactive (ERA/NERA) impulse layer** MLS-MPM. A KE
rod and an arbitrary front-to-back armor stack (any number of layers, each its
own material, with optional standoff gaps) are seeded from the scenario and run
through a P2G / grid-update / G2P substep cycle with fixed-corotated
hyperelasticity, followed by a perfectly-plastic von Mises radial return
(``_return_mapping``) that caps deviatoric stress at each material's yield and
lets metal flow. Every material constant is a per-particle array
(``mu/lam/yield/dthr/brittle/reactive/det_pressure/burn_time/ign_comp``), so
heterogeneous stacks just work.

Failure has two modes (``_update_damage``): **ductile** metals accumulate
equivalent plastic strain into ``alpha`` and spall once it crosses
``damage_threshold``; **brittle** ceramics latch on a *stress* trigger instead
(von Mises Cauchy ≥ yield, or max tensile principal ≥ 0.1·yield) so they shatter
with ~zero plastic flow rather than acting as an indestructible ductile wall. A
failed particle is latched ``damage=1`` and ``_p2g`` drops its stress term — it
becomes a cohesion-free **free fragment** (mass + momentum only). Particle count
is fixed — spall = flagged + detached, never created/destroyed (cache contract
§4).

**Reactive layer (``_update_reactive``, milestone 5):** a reactive filler
(``era_filler``) sandwiched between plates ignites when the impact shock
compresses it past ``ignition_compression`` (``det(F)`` threshold) and releases
an isotropic detonation overpressure for ``burn_time`` ms — a *source term* in
``_p2g`` that flings the plates apart through the ordinary grid (emergent, not a
scripted rod kick). Reactive particles run their own elastic → detonation →
debris state machine and are excluded from the ductile-spall path so they can't
spall before detonating (see the reactive note below). A persistent NERA
(inert-bulge) layer is the *unignited* soft-elastic branch held open —
``ignition_compression=0`` so it never ignites — not merely
``detonation_pressure=0`` (which still ignites on the shock and collapses).

Verified on the RTX 5090: ``apfsds_vs_rha`` / ``apfsds_vs_composite`` /
``apfsds_vs_spaced`` all bake clean (no NaN/Inf) and pass ``validate_cache``;
RHA spall stays ~16% (ductile path unchanged), the ceramic core shatters
(interface cracks + comminution ahead of the rod), the spaced front plate is
defeated across a preserved standoff gap, and the reactive deck detonates and
flings the sandwich plates apart (milestone 5). At 0° that detonation does NOT
meaningfully degrade the rod (it sweeps laterally, symmetric about the rod axis).
**Milestone 6 (oblique):** ``_seed`` rotates the projectile *rectangle* by
``angle_deg`` about its tip so it strikes nose-first; the armor slabs stay
axis-aligned (only the relative rod/plate-normal angle is physical, so this is
frame-equivalent to tilting the slabs and leaves M1–M5 seeding untouched;
``angle_deg=0`` is exact identity). At 55° the reactive layer measurably protects
the backing plate (main-plate spall ~40% lower vs an equal-areal-mass inert twin,
absent at 0°) by shoving it forward + dispersing the flyer/filler follow-through —
though the tough tungsten rod itself is not cut. Verified — see PHYSICS §3.1/§3.2.

Two hard rules for whoever grows this (root §2, §11):

  1. This module must NEVER import or reference the visualizer.
  2. Before trusting the GPU, assert it is actually a CUDA device and not a
     silent CPU fallback. ``assert_gpu()`` is the guard; ``run.py`` calls it.

Engine: Warp, not Taichi — Taichi has no wheel for this machine's Python 3.14,
and Warp is verified on the RTX 5090 (sm_120). This is the swap root §5
anticipated; the visualizer never saw the solver, so it cost zero viewer code.

Everything here is in the **mm-ms-g** unit system (root §7). The only SI values
are ``solver.dt`` and ``solver.total_time`` in the deck (seconds by the config's
convention); they are converted to ms once, at the top of ``bake``.
"""

from __future__ import annotations

import math

import numpy as np

from . import materials

# ---------------------------------------------------------------------------
# Numerics knobs
# ---------------------------------------------------------------------------

# CFL number for the explicit substep. Conservative for high-speed impact; the
# substep dt is min(deck dt, CFL * dx / c_p). See root §6/§11 and PHYSICS §4.
CFL: float = 0.3


def assert_gpu(device: str = "cuda:0") -> None:
    """Fail loudly if the solver is not actually on a CUDA device.

    A 'slow but working' bake often means a silent CPU fallback (root §11).
    Call this right after ``wp.init()`` before any real bake.
    """
    import warp as wp  # local import so config/schema stay Warp-free

    dev = wp.get_device(device)
    if not dev.is_cuda:
        raise RuntimeError(
            f"solver resolved device {device!r} to {dev!r}, which is not CUDA — "
            "refusing to bake. A silent CPU fallback usually means a driver/CUDA/"
            "Blackwell issue; see CLAUDE.md §5 and §11. Use --cpu to opt in "
            "explicitly for debugging."
        )


# ===========================================================================
# Warp kernels (defined at import time so Warp can JIT them once).
# ===========================================================================
#
# Standard MLS-MPM with quadratic B-spline weights and APIC affine transfer,
# grown from the canonical 88-line reference (PHYSICS §1/§5). 2D. Fixed-
# corotated elasticity, with the 2x2 rotation from a closed-form polar
# decomposition (no SVD needed in 2D).

import warp as wp  # noqa: E402 — kernels must be defined at module import


@wp.func
def _bspline_w(f: float):
    """Quadratic B-spline weights for the 3 grid nodes covering fractional f."""
    return wp.vec3(
        0.5 * (1.5 - f) * (1.5 - f),
        0.75 - (f - 1.0) * (f - 1.0),
        0.5 * (f - 0.5) * (f - 0.5),
    )


@wp.func
def _polar_r(F: wp.mat22):
    """Rotation part R of the polar decomposition F = R S, closed form (2x2)."""
    xx = F[0, 0] + F[1, 1]
    yy = F[1, 0] - F[0, 1]
    d = wp.sqrt(xx * xx + yy * yy)
    if d < 1.0e-9:
        return wp.mat22(1.0, 0.0, 0.0, 1.0)
    c = xx / d
    s = yy / d
    return wp.mat22(c, -s, s, c)


@wp.func
def _fixed_corotated_pft(F: wp.mat22, mu: float, lam: float):
    """Fixed-corotated Kirchhoff stress term P(F) Fᵀ (root §6, PHYSICS §3).

    The elastic stress that drives the P2G momentum scatter. Factored out so the
    reactive filler's ``_p2g`` state machine (elastic → detonation → debris) can
    reuse the exact same elastic branch as ordinary material without copy-paste.
    """
    J = wp.determinant(F)
    R = _polar_r(F)
    lp = lam * (J - 1.0) * J
    return 2.0 * mu * (F - R) * wp.transpose(F) + wp.mat22(lp, 0.0, 0.0, lp)


@wp.func
def _svd22(M: wp.mat22):
    """Closed-form 2x2 SVD: M = U diag(sx, sy) V^T with U = R(phi), V = R(theta).

    Returns (phi, sx, sy, theta) packed in a vec4. Warp has no ``svd2``; this is
    the standard analytic form. ``sy`` carries sign(det M), so it can be negative
    when an element momentarily inverts (det<=0) under a violent impact substep —
    the caller must floor the singular values before ``log`` or NaNs propagate.
    """
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]
    e = (a + d) * 0.5
    f = (a - d) * 0.5
    g = (c + b) * 0.5
    h = (c - b) * 0.5
    q = wp.sqrt(e * e + h * h)
    r = wp.sqrt(f * f + g * g)
    a1 = wp.atan2(g, f)
    a2 = wp.atan2(h, e)
    theta = (a2 - a1) * 0.5
    phi = (a2 + a1) * 0.5
    return wp.vec4(phi, q + r, q - r, theta)


# Frobenius-norm form of the J2 yield surface: ||dev tau|| = SQRT23 * sigma_Y.
# SQRT23 = sqrt(2/3) is the 3D von Mises convention, reused here in 2D as a
# plausibility knob (root §1) — the plane-strain deviatoric split uses only two
# principal log-strains, so this is not exact 3D J2.
SQRT23 = 0.8164965809277260

# Per-substep cap on the accumulated equivalent-plastic-strain increment. A
# legitimate increment over one tiny CFL substep is minute; a particle that
# momentarily inverts or over-compresses at the shock front would otherwise book
# a garbage-huge increment in a single step and then spuriously spall the instant
# the damage milestone compares ``alpha`` to ``damage_threshold`` (0.8-2.0).
MAX_DALPHA: float = 0.02

# Brittle fracture (milestone 4). Ductile metals fail by accumulating plastic
# strain to a threshold (the ``alpha`` path); brittle materials (ceramics,
# glass) have essentially no plastic reserve — they shatter the instant the
# stress state reaches their strength surface, with ~zero plastic flow. So a
# brittle particle latches ``damage`` on a *stress* trigger instead:
#   * von Mises Cauchy stress >= yield_strength — compressive comminution /
#     shatter directly under the penetrator (the intense contact zone), and
#   * max tensile principal stress >= BRITTLE_TENSILE_FRAC * yield_strength —
#     tensile mode-I cracking at free surfaces (back-face spall, lateral edges,
#     the fracture conoid). Ceramics crack in tension at a small fraction of
#     their compressive strength; ~0.1 is a representative order-of-magnitude
#     ratio (root §10 — illustrative, not a spec-sheet number).
# yield_strength doubles as the brittle fracture strength — no new material
# field. Ductile materials (brittle flag off) are untouched: they stay on the
# alpha path, so the all-metal apfsds_vs_rha bake is bit-for-bit unchanged.
BRITTLE_TENSILE_FRAC: float = 0.1

# Reactive-filler velocity clamp (mm/ms == m/s). The detonation source in `_p2g`
# is an additive per-particle overpressure that is independent of the particle's
# deformation state; once the sandwich plates separate, the now-unconfined light
# filler (density 1.6e-3, ~11x lighter than steel) keeps absorbing the pulse
# every substep with nothing to react against, so a thin tail runs away to
# ~14 km/s — unphysical (detonation-product gas is a few km/s at most) and a CFL
# hazard (0.76 dx/substep). We cap reactive-particle speed at a physical
# detonation-product scale. This bleeds a little energy from the runaway tail
# only (reactive particles only — the rod and plates are untouched), the
# stable/pretty tradeoff root §1/§11 explicitly endorses. Plate fling velocities
# (~400-550 m/s) sit far below this, so the ERA mechanism is unaffected.
REACTIVE_VMAX: float = 3000.0

# Reactive impulse layer — ERA/NERA (milestone 5). A reactive filler (era_filler,
# `reactive: true`) models the interlayer of a reactive-armor sandwich
# [plate | filler | plate]. When the impact shock reaches it, it *ignites* and
# releases an isotropic overpressure that flings the two sandwiching plates apart
# — the front plate is driven back into the oncoming rod (tip erosion + tamping)
# and disrupts the penetrator. This is modelled as a **pressure source term**
# carried through the ordinary MLS-MPM grid (emergent plate motion), NOT a
# scripted kick to the rod.
#
# Reactive particles run a self-contained state machine in `_p2g`, keyed off
# `reactive`/`burn`/`damage` and NEVER the ductile-spall gate:
#   * unignited  -> soft fixed-corotated elastic (the plates also bulge from the
#                   raw shock even with no detonation). A persistent NERA bulge is
#                   this branch held open: ignition_compression=0 so the filler
#                   never ignites — NOT merely detonation_pressure=0 (that still
#                   ignites on the shock, burns zero pressure, then collapses).
#   * burning    -> isotropic detonation overpressure for `burn_time` ms.
#   * spent      -> cohesion-free debris (mass + momentum, no stress).
# They are deliberately excluded from `_return_mapping` and the ductile branch of
# `_update_damage`: era_filler's yield (50 MPa) and `damage_threshold` (0.02, ==
# MAX_DALPHA) would otherwise let it ductile-spall in the *same* shocked substep
# it should ignite — `_p2g` drops the stress term for spalled particles, which
# would silently no-op the detonation. `damage` is instead repurposed for
# reactive particles as the "has ignited / is spent" latch (also the viewer flag)
# and is written ONLY by `_update_reactive`. Everything is gated on
# `reactive > 0.5`, so the three non-reactive KE decks bake identically.


@wp.kernel
def _return_mapping(
    F: wp.array(dtype=wp.mat22),
    mu: wp.array(dtype=float),
    yield_k: wp.array(dtype=float),
    reactive: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
):
    """Perfectly-plastic von Mises radial return in log-strain (Hencky) space.

    Applied per particle after G2P. SVD the (elastic) deformation gradient,
    take principal log-strains, split off the volumetric part, and radially
    return the deviatoric part onto the yield surface. Plastic flow is
    isochoric (the volumetric log-strain is untouched) — correct for metals.
    Accumulates equivalent plastic strain into ``alpha`` (feeds the damage
    milestone; unused this milestone). No hardening: perfectly plastic.

    Reactive filler is excluded (it has its own state machine and must not
    accumulate ``alpha`` / ductile-spall before it detonates — see the reactive
    note above); its F is left to the elastic + detonation path in ``_p2g``.
    """
    p = wp.tid()
    if reactive[p] > 0.5:
        return
    ys = yield_k[p]
    if ys <= 0.0:
        return

    s = _svd22(F[p])
    phi = s[0]
    theta = s[3]
    inverted = s[2] <= 0.0  # det(F)<=0: element flipped this substep
    # Floor both singular values positive before log: an inverted element
    # (det<=0) makes sy<=0 and log(sy) NaN, which then poisons everything.
    # Reconstructing from positive singular values also recovers from inversion.
    s1 = wp.max(s[1], 1.0e-4)
    s2 = wp.max(s[2], 1.0e-4)

    e1 = wp.log(s1)
    e2 = wp.log(s2)
    e_mean = (e1 + e2) * 0.5  # volumetric (pressure) log-strain — preserved
    ed1 = e1 - e_mean
    ed2 = e2 - e_mean
    norm_ed = wp.sqrt(ed1 * ed1 + ed2 * ed2)

    # Deviatoric elastic log-strain the yield surface allows: ||dev tau|| = 2 mu
    # ||e_dev||, yielding when it reaches SQRT23 * sigma_Y.
    e_yield = SQRT23 * ys / (2.0 * mu[p])
    if norm_ed <= e_yield or norm_ed < 1.0e-12:
        return  # elastic step, F unchanged

    scale = e_yield / norm_ed
    e1n = e_mean + ed1 * scale
    e2n = e_mean + ed2 * scale
    s1n = wp.exp(e1n)
    s2n = wp.exp(e2n)

    # Reconstruct F = U diag(s1n, s2n) V^T with U = R(phi), V^T = R(theta)^T.
    cphi = wp.cos(phi)
    sphi = wp.sin(phi)
    cth = wp.cos(theta)
    sth = wp.sin(theta)
    U = wp.mat22(cphi, -sphi, sphi, cphi)
    Sn = wp.mat22(s1n, 0.0, 0.0, s2n)
    Vt = wp.mat22(cth, sth, -sth, cth)  # R(theta)^T
    F[p] = U * Sn * Vt

    # Equivalent plastic strain increment (J2, Frobenius convention), guarded so
    # inverted / over-compressed shock-front particles don't book a garbage
    # increment that would spuriously spall at the damage milestone (see
    # MAX_DALPHA). F is still returned above; only the accumulation is guarded.
    if not inverted:
        alpha[p] = alpha[p] + wp.min(SQRT23 * (norm_ed - e_yield), MAX_DALPHA)


@wp.func
def _stress_invariants(F: wp.mat22, mu: float, lam: float):
    """Von Mises and max tensile principal of the fixed-corotated Cauchy stress.

    Cauchy sigma = (1/J) P(F) F^T with the same fixed-corotated P used in _p2g
    and the host-side ``_von_mises`` readout. Returns wp.vec2(vm, s_max_principal)
    in MPa. Used by the brittle fracture trigger. ``J`` is floored positive so a
    momentarily inverted/over-compressed shock-front element can't divide-by-zero
    or flip principal signs into a spurious tensile reading.
    """
    J = wp.determinant(F)
    R = _polar_r(F)
    lp = lam * (J - 1.0) * J
    PFt = 2.0 * mu * (F - R) * wp.transpose(F) + wp.mat22(lp, 0.0, 0.0, lp)
    invJ = 1.0 / wp.max(J, 1.0e-6)
    sxx = PFt[0, 0] * invJ
    syy = PFt[1, 1] * invJ
    sxy = PFt[0, 1] * invJ
    syx = PFt[1, 0] * invJ
    # Plane-stress vM invariant, consistent with the host `stress` column.
    vm = wp.sqrt(wp.max(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * syx, 0.0))
    # 2D principal stresses of the (near-symmetric) Cauchy tensor.
    avg = 0.5 * (sxx + syy)
    rad = wp.sqrt(wp.max(0.25 * (sxx - syy) * (sxx - syy) + sxy * syx, 0.0))
    return wp.vec2(vm, avg + rad)  # (von Mises, max principal — tension positive)


@wp.kernel
def _update_damage(
    F: wp.array(dtype=wp.mat22),
    mu: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    yield_k: wp.array(dtype=float),
    brittle: wp.array(dtype=float),
    reactive: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    dthr: wp.array(dtype=float),
    damage: wp.array(dtype=float),
):
    """Latch a particle as spalled once it fails — ductile or brittle path.

    Two failure modes (root §6). **Ductile** metals (``brittle`` off): ``alpha``
    (equivalent plastic strain, accumulated in ``_return_mapping`` and guarded by
    ``MAX_DALPHA``) crossing the material's ``damage_threshold`` — the milestone-3
    path, unchanged. **Brittle** materials (ceramics; ``brittle`` on): a *stress*
    trigger instead — von Mises Cauchy stress reaching ``yield_strength`` (shatter
    under the penetrator) or max tensile principal stress reaching
    ``BRITTLE_TENSILE_FRAC * yield_strength`` (tensile cracking at free surfaces).
    Brittle materials thus fail with ~zero plastic flow, the defining brittle
    signature, rather than acting as an indestructible ductile wall.

    Once set, ``damage`` stays 1.0 — fracture is irreversible. For non-reactive
    material this is the only writer of ``damage`` (reactive filler is skipped
    here and latched instead by ``_update_reactive``); ``_p2g`` reads it to drop a
    failed particle's cohesion (it becomes a free fragment: mass + momentum, no
    stress). Fixed particle count — nothing is created or destroyed, only flagged
    (contract §4).
    """
    p = wp.tid()
    if reactive[p] > 0.5:
        return  # reactive filler owns its own `damage` latch (see _update_reactive)
    if damage[p] >= 0.5:
        return  # already failed; latched

    if brittle[p] > 0.5:
        ys = yield_k[p]
        if ys > 0.0:
            inv = _stress_invariants(F[p], mu[p], lam[p])
            if inv[0] >= ys or inv[1] >= BRITTLE_TENSILE_FRAC * ys:
                damage[p] = 1.0
    else:
        if alpha[p] >= dthr[p] and dthr[p] > 0.0:
            damage[p] = 1.0


@wp.kernel
def _update_reactive(
    F: wp.array(dtype=wp.mat22),
    reactive: wp.array(dtype=float),
    ign_comp: wp.array(dtype=float),
    burn_time: wp.array(dtype=float),
    burn: wp.array(dtype=float),
    damage: wp.array(dtype=float),
    dt: float,
):
    """Drive the reactive filler's ignition + burn state (ERA/NERA, milestone 5).

    Per reactive particle, after G2P, in three latched stages:

      * **burning** (``burn > 0``) — count the detonation pulse down in physical
        time (ms), by ``dt``, flooring at 0. Tracking real time (not a substep
        count) keeps the pulse duration independent of ``grid_resolution`` /
        substeps-per-frame.
      * **spent** (``burn == 0``, ``damage`` latched) — do nothing; never
        re-ignite.
      * **unignited** — ignite when the impact shock compresses the filler past
        ``ignition_compression`` (``det(F)`` drops below it). Ignition sets
        ``burn = burn_time`` and latches ``damage = 1`` (marks the particle
        reacting/consumed; also the viewer flag and the re-ignition guard).

    ``burn`` is read by ``_p2g`` one substep later to add the detonation
    overpressure — a negligible one-substep latency. Non-reactive particles are
    skipped, so the non-reactive KE decks are unaffected.
    """
    p = wp.tid()
    if reactive[p] < 0.5:
        return
    if burn[p] > 0.0:
        burn[p] = wp.max(burn[p] - dt, 0.0)  # burning: age the pulse
        # A detonating gas has no elastic reference configuration — reset F to
        # identity so the (skipped-by-`_return_mapping`) filler F can't drift to
        # huge/inverted values under the sustained overpressure and overflow the
        # host `_von_mises` readout. The stress readout is masked for reactive
        # particles anyway (damage>0.5 -> 0), so this is purely hygiene; the
        # detonation momentum lives in `burn`/`_p2g`, not in F.
        F[p] = wp.mat22(1.0, 0.0, 0.0, 1.0)
        return
    if damage[p] >= 0.5:
        # Spent debris: still advected through G2P every substep but skipped by
        # `_return_mapping`, so its (unused, stress-masked) F would otherwise
        # drift to inf and overflow the host readout. Pin it to identity — like
        # burning filler, spent debris carries no elastic memory.
        F[p] = wp.mat22(1.0, 0.0, 0.0, 1.0)
        return  # spent: ignited once already, never re-ignite
    ic = ign_comp[p]
    if ic > 0.0:
        if wp.determinant(F[p]) < ic:  # shock arrived -> ignite
            burn[p] = burn_time[p]
            damage[p] = 1.0


@wp.kernel
def _p2g(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    F: wp.array(dtype=wp.mat22),
    mass: wp.array(dtype=float),
    vol0: wp.array(dtype=float),
    mu: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    damage: wp.array(dtype=float),
    reactive: wp.array(dtype=float),
    det_pressure: wp.array(dtype=float),
    burn: wp.array(dtype=float),
    grid_v: wp.array2d(dtype=wp.vec2),
    grid_m: wp.array2d(dtype=float),
    origin: wp.vec2,
    inv_dx: float,
    dx: float,
    dt: float,
):
    p = wp.tid()
    Xp = (x[p] - origin) * inv_dx
    base = wp.vec2i(int(wp.floor(Xp[0] - 0.5)), int(wp.floor(Xp[1] - 0.5)))
    fx = Xp - wp.vec2(float(base[0]), float(base[1]))
    wx = _bspline_w(fx[0])
    wy = _bspline_w(fx[1])

    # Stress term P(F) F^T scattered as APIC momentum (root §6, PHYSICS §3). The
    # scaling (-dt * vol0 * 4 * inv_dx^2) is shared by every branch below.
    coeff = -dt * vol0[p] * 4.0 * inv_dx * inv_dx
    affine = mass[p] * C[p]
    if reactive[p] > 0.5:
        # Reactive filler runs its OWN state machine, keyed off burn/damage and
        # never the ductile-spall gate (see the reactive note above):
        #   burning   -> isotropic detonation overpressure that flings the plates
        #   unignited -> soft fixed-corotated elastic (a never-igniting filler,
        #                ignition_compression=0, stays here: the NERA bulge)
        #   spent     -> cohesion-free debris (no stress)
        if burn[p] > 0.0:
            # Negative-diagonal Kirchhoff term = internal overpressure pushing
            # material OUTWARD (an EOS-free stand-in for detonation-product gas).
            # Sign verified empirically: the sandwiching plates fly apart, not in.
            pd = det_pressure[p]
            affine = coeff * wp.mat22(-pd, 0.0, 0.0, -pd) + affine
        elif damage[p] < 0.5:
            affine = coeff * _fixed_corotated_pft(F[p], mu[p], lam[p]) + affine
        # else spent: no stress term
    elif damage[p] < 0.5:
        # A spalled particle (damage>=0.5) is a free fragment: it drops its
        # stress term entirely — it can no longer hold tension or shear, so a
        # perforation channel / back-face spall spray opens up. It KEEPS its mass
        # and APIC momentum so grid momentum is conserved and it still collides
        # (grid-shared contact) instead of ghosting through the rod.
        affine = coeff * _fixed_corotated_pft(F[p], mu[p], lam[p]) + affine

    for i in range(3):
        for j in range(3):
            dpos = (wp.vec2(float(i), float(j)) - fx) * dx
            weight = wx[i] * wy[j]
            gi = base[0] + i
            gj = base[1] + j
            wp.atomic_add(grid_v, gi, gj, weight * (mass[p] * v[p] + affine * dpos))
            wp.atomic_add(grid_m, gi, gj, weight * mass[p])


@wp.kernel
def _grid_op(
    grid_v: wp.array2d(dtype=wp.vec2),
    grid_m: wp.array2d(dtype=float),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    m = grid_m[i, j]
    if m > 0.0:
        vel = grid_v[i, j] / m
        # Free-SLIP domain walls, a few cells thick: only the component heading
        # INTO the wall is killed, tangential motion is untouched. (This was
        # mislabelled "sticky reflecting" — sticky would zero the whole velocity
        # and reflecting would negate the normal component. It does neither.)
        #
        # Slip is the useful reading: a slip wall is a MIRROR PLANE. Combined
        # with armor slabs seeded across the full domain height (see `_seed`),
        # the target behaves as a plate that CONTINUES beyond the frame — armor
        # on a vehicle — rather than a finite block floating in vacuum whose free
        # top/bottom edges flare outward. The mirror does imply an image of the
        # projectile one domain-height away, so a deck must be tall enough that
        # the event finishes before the rod or its spray nears a wall; that is a
        # per-deck sizing duty, not something this kernel can enforce.
        #
        # No gravity: over a ~0.1 ms window it is utterly negligible next to
        # impact stresses.
        bound = 3
        vx = vel[0]
        vy = vel[1]
        if i < bound and vx < 0.0:
            vx = 0.0
        if i > nx - bound and vx > 0.0:
            vx = 0.0
        if j < bound and vy < 0.0:
            vy = 0.0
        if j > ny - bound and vy > 0.0:
            vy = 0.0
        grid_v[i, j] = wp.vec2(vx, vy)


@wp.kernel
def _g2p(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.mat22),
    F: wp.array(dtype=wp.mat22),
    grid_v: wp.array2d(dtype=wp.vec2),
    origin: wp.vec2,
    inv_dx: float,
    dt: float,
    lo: wp.vec2,
    hi: wp.vec2,
):
    p = wp.tid()
    Xp = (x[p] - origin) * inv_dx
    base = wp.vec2i(int(wp.floor(Xp[0] - 0.5)), int(wp.floor(Xp[1] - 0.5)))
    fx = Xp - wp.vec2(float(base[0]), float(base[1]))
    wx = _bspline_w(fx[0])
    wy = _bspline_w(fx[1])

    new_v = wp.vec2(0.0, 0.0)
    new_C = wp.mat22(0.0, 0.0, 0.0, 0.0)
    for i in range(3):
        for j in range(3):
            dpos = wp.vec2(float(i), float(j)) - fx
            weight = wx[i] * wy[j]
            g = grid_v[base[0] + i, base[1] + j]
            new_v += weight * g
            new_C += (4.0 * inv_dx * weight) * wp.outer(g, dpos)

    v[p] = new_v
    C[p] = new_C
    # Keep every particle at least one cell inside the domain. This is a memory-
    # safety guard, not a physics choice: both transfer kernels index the grid at
    # `floor(Xp - 0.5) + {0,1,2}` with no bounds check, so a particle within half
    # a cell of the low edge gives base = -1 and the P2G stencil scatters OUT OF
    # BOUNDS. The old 10%-of-height seeding margin hid this; now that armor spans
    # the full domain height (see `_seed`) particles sit at the wall for the whole
    # bake, so the guard has to be explicit. The slip wall in `_grid_op` already
    # removes wall-normal velocity, so in practice this clamp almost never binds —
    # it exists so that "almost never" cannot corrupt memory.
    x[p] = wp.vec2(
        wp.clamp(x[p][0] + dt * new_v[0], lo[0], hi[0]),
        wp.clamp(x[p][1] + dt * new_v[1], lo[1], hi[1]),
    )
    F[p] = (wp.mat22(1.0, 0.0, 0.0, 1.0) + dt * new_C) * F[p]


@wp.kernel
def _clamp_reactive_v(
    v: wp.array(dtype=wp.vec2),
    reactive: wp.array(dtype=float),
    vmax: float,
):
    """Cap reactive-filler speed at a physical detonation-product scale.

    The `_p2g` detonation source is F-independent, so an unconfined light filler
    particle keeps gaining momentum from the pulse each substep and a thin tail
    runs away to ~14 km/s (unphysical + CFL hazard). Clamp reactive particles
    only — the rod and plates carry real momentum and are left untouched. See
    REACTIVE_VMAX. Runs after G2P (on the post-transfer particle velocity).
    """
    p = wp.tid()
    if reactive[p] < 0.5:
        return
    s = wp.length(v[p])
    if s > vmax:
        v[p] = v[p] * (vmax / s)


# ===========================================================================
# Host-side seeding + attribute extraction (pure numpy; kernels stay clean).
# ===========================================================================


def _lame(E: float, nu: float) -> tuple[float, float]:
    """Lamé parameters (mu, lambda) from Young's modulus and Poisson ratio.

    Plane-strain / 3D convention: lambda = E nu / ((1+nu)(1-2nu)).
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def _fill_rect(x0: float, x1: float, y0: float, y1: float, spacing: float):
    """Regular lattice of points filling a rectangle, at cell centers."""
    nx = max(1, int(round((x1 - x0) / spacing)))
    ny = max(1, int(round((y1 - y0) / spacing)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _nose_halfwidth(s, nose_len: float, radius: float, shape: str):
    """Half-width (mm) of the rod at axial distance ``s`` behind its tip.

    ``s`` is a numpy array measured along the rod axis, 0 at the tip. Past the
    nose the rod is at full ``radius``, so this is the whole silhouette, not
    just the nose. Profiles are illustrative (root §10).
    """
    if shape == "blunt" or nose_len <= 0.0:
        return np.full_like(s, radius)
    t = np.clip(s / nose_len, 0.0, 1.0)
    if shape == "conical":
        return radius * t
    # Tangent ogive: a circular arc of radius `rho` meeting the shank
    # tangentially at the nose base, so there is no corner where nose meets
    # body. rho is fixed by requiring halfwidth(0)=0 and halfwidth(nose_len)=r.
    rho = (radius * radius + nose_len * nose_len) / (2.0 * radius)
    return np.sqrt(rho * rho - (nose_len * (1.0 - t)) ** 2) - (rho - radius)


def _seed(scenario, dx: float, spacing: float):
    """Seed the projectile and armor stack into particle arrays (numpy).

    Geometry is illustrative (root §10): the projectile is a rectangle aimed
    +x at the front face of the armor stack; each armor layer is a slab spanning
    the full domain height, laid front-to-back with any standoff gap.

    Returns a dict of numpy arrays: pos, vel, mu, lam, yield, dthr, brittle,
    reactive, det_pressure, burn_time, ign_comp, mass, vol0, mat_id.
    """
    dom = scenario.domain
    proj = scenario.projectile
    # Impact height. Defaults to mid-domain — what every normal-incidence deck
    # wants, and bit-for-bit what the pre-existing seeding did. Oblique decks
    # override it: the rod plunges in -y as it advances, so it needs most of the
    # domain BELOW the impact point and only rod-clearance above it. Centring
    # those decks would force a domain ~2x taller (and ~2x the particles) to buy
    # headroom the rod never uses.
    y_center = (
        proj.impact_y if proj.impact_y is not None
        else 0.5 * (dom.ymin + dom.ymax)
    )

    # Armor front face sits mid-domain; layers march +x from there.
    armor_front = dom.xmin + 0.5 * (dom.xmax - dom.xmin)
    # Armor slabs span the FULL domain height, inset by two cells. Together with
    # the slip (mirror) walls in `_grid_op` this makes each slab read as a plate
    # that CONTINUES beyond the frame — armor on a vehicle — instead of a finite
    # block floating in vacuum. Previously slabs stopped 10% of the height short
    # of each wall, so the plate had free top and bottom edges that flared
    # outward into the void and let debris escape around them.
    #
    # The inset is two cells, not zero: `_p2g`/`_g2p` index at
    # `floor(Xp-0.5)+{0,1,2}`, so a particle within half a cell of the low edge
    # scatters out of bounds. Two cells keeps the outermost row safely clear
    # while leaving a sub-mm gap (dx<1mm) in place of the old ~10mm void — small
    # enough that the slab is effectively wall-to-wall.
    inset = 2.0 * dx

    (pos_list, vel_list, mu_list, lam_list, yield_list, dthr_list, brittle_list,
     reactive_list, detp_list, burnt_list, igncomp_list,
     mass_list, vol_list, mid_list) = (
        [], [], [], [], [], [], [], [], [], [], [], [], [], [],
    )
    p_vol = spacing * spacing  # 2D "volume" (area) per particle

    def add_region(pts, mat, vx, vy):
        n = pts.shape[0]
        mu, lam = _lame(mat.youngs_modulus, mat.poisson_ratio)
        pos_list.append(pts)
        vel_list.append(np.tile([vx, vy], (n, 1)))
        mu_list.append(np.full(n, mu))
        lam_list.append(np.full(n, lam))
        yield_list.append(np.full(n, mat.yield_strength))
        dthr_list.append(np.full(n, mat.damage_threshold))
        brittle_list.append(np.full(n, 1.0 if mat.brittle else 0.0))
        reactive_list.append(np.full(n, 1.0 if mat.reactive else 0.0))
        detp_list.append(np.full(n, mat.detonation_pressure))
        burnt_list.append(np.full(n, mat.burn_time))
        igncomp_list.append(np.full(n, mat.ignition_compression))
        mass_list.append(np.full(n, mat.density * p_vol))
        vol_list.append(np.full(n, p_vol))
        mid_list.append(np.full(n, float(mat.material_id)))

    # Projectile: leading (+x) tip a small gap before the armor front. At
    # obliquity the rod RECTANGLE is rotated so its long axis stays parallel to
    # its velocity — a real APFSDS flies nose-first, so this is a yawed-zero
    # oblique strike, not a broadside. Only the RELATIVE angle between the rod
    # axis and the plate normal matters, so rotating the rod against fixed
    # vertical slabs is frame-equivalent to tilting the slabs against a
    # horizontal rod — and it leaves the validated armor seeding (milestones
    # 1-5) completely untouched. This is where reactive armor earns its keep
    # (milestone 6): the detonation-flung plates gain a velocity component
    # perpendicular to the tilted rod and sweep laterally across it. See
    # PHYSICS §3.1.
    gap = 3.0 * dx
    tip_x = armor_front - gap
    rod = _fill_rect(
        tip_x - proj.length, tip_x,
        y_center - 0.5 * proj.diameter, y_center + 0.5 * proj.diameter,
        spacing,
    )
    # Carve the nose: keep only points inside the tapering silhouette. Done here,
    # in rod-local axis-aligned coords (+x is the rod axis, tip leads at tip_x)
    # and BEFORE the rotation below, so the mask is a plain 1D compare rather
    # than a rotated-frame one. The rotation is about the tip, which the carve
    # leaves in place. A real APFSDS is pointed; a flat-faced rod was the
    # geometric tell that this is a sim. See `Projectile.nose_shape` for what
    # this does and does not buy physically.
    rod = rod[
        np.abs(rod[:, 1] - y_center) <= _nose_halfwidth(
            tip_x - rod[:, 0], proj.nose_len, 0.5 * proj.diameter, proj.nose_shape,
        )
    ]
    ang = math.radians(proj.angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    # Rotate the rod by -ang about its tip (tip_x, y_center) so local +x maps to
    # the velocity direction (cos, -sin): tip stays put and LEADS into the plate
    # (down-and-right), the body trails up-and-left. angle_deg=0 gives ca=1,sa=0
    # -> exact identity, so every normal-incidence deck seeds bit-for-bit as
    # before.
    rx = rod[:, 0] - tip_x
    ry = rod[:, 1] - y_center
    rod = np.stack([
        tip_x + ca * rx + sa * ry,
        y_center - sa * rx + ca * ry,
    ], axis=1)
    add_region(rod, materials.get(proj.material),
               proj.velocity * ca, -proj.velocity * sa)

    # Armor stack, front to back.
    x_cursor = armor_front
    for layer in scenario.armor:
        x_cursor += layer.standoff
        slab = _fill_rect(
            x_cursor, x_cursor + layer.thickness,
            dom.ymin + inset, dom.ymax - inset,
            spacing,
        )
        add_region(slab, materials.get(layer.material), 0.0, 0.0)
        x_cursor += layer.thickness

    return {
        "pos": np.concatenate(pos_list).astype(np.float32),
        "vel": np.concatenate(vel_list).astype(np.float32),
        "mu": np.concatenate(mu_list).astype(np.float32),
        "lam": np.concatenate(lam_list).astype(np.float32),
        "yield": np.concatenate(yield_list).astype(np.float32),
        "dthr": np.concatenate(dthr_list).astype(np.float32),
        "brittle": np.concatenate(brittle_list).astype(np.float32),
        "reactive": np.concatenate(reactive_list).astype(np.float32),
        "det_pressure": np.concatenate(detp_list).astype(np.float32),
        "burn_time": np.concatenate(burnt_list).astype(np.float32),
        "ign_comp": np.concatenate(igncomp_list).astype(np.float32),
        "mass": np.concatenate(mass_list).astype(np.float32),
        "vol0": np.concatenate(vol_list).astype(np.float32),
        "mat_id": np.concatenate(mid_list).astype(np.float32),
    }


def _von_mises(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Von Mises equivalent of the Cauchy stress (MPa), for the `stress` column.

    Cauchy sigma = (1/J) P(F) F^T with fixed-corotated P. This is the
    plane-*stress* vM invariant (drops sigma_zz) used in a plane-strain sim —
    fine for a plausibility readout; it nudges the bulk median slightly above
    yield. With plasticity active this reads *approximately* capped near each
    material's yield, not exactly: the small-strain elastic offset and 2D
    deviatoric convention account for the bulk, and a thin tail over-reads at
    extreme volumetric compression (small J) at the shock front, because
    fixed-corotated has no EOS. That tail is a pre-existing property of the
    elastic model (latent since milestone 1), not a plasticity artifact; tame it
    viewer-side with a percentile clamp on the colormap (see ``_return_mapping``).
    """
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    J = a * d - b * c
    xx = a + d
    yy = c - b
    den = np.sqrt(xx * xx + yy * yy)
    den = np.where(den < 1e-9, 1.0, den)
    cs, sn = xx / den, yy / den
    # (F - R) F^T
    fr00, fr01, fr10, fr11 = a - cs, b + sn, c - sn, d - cs
    m00 = fr00 * a + fr01 * b
    m01 = fr00 * c + fr01 * d
    m10 = fr10 * a + fr11 * b
    m11 = fr10 * c + fr11 * d
    lp = lam * (J - 1.0) * J
    pft00 = 2.0 * mu * m00 + lp
    pft01 = 2.0 * mu * m01
    pft10 = 2.0 * mu * m10
    pft11 = 2.0 * mu * m11 + lp
    Js = np.where(np.abs(J) < 1e-9, 1.0, J)
    s00, s01, s10, s11 = pft00 / Js, pft01 / Js, pft10 / Js, pft11 / Js
    return np.sqrt(np.maximum(s00 * s00 - s00 * s11 + s11 * s11 + 3.0 * s01 * s10, 0.0))


# ===========================================================================
# Bake driver
# ===========================================================================


def bake(scenario, writer, device: str = "cuda:0") -> None:
    """Run the elastic MLS-MPM substep loop and dump render frames.

    Wiring is in run.py; the physics is here. Sets ``writer.particle_count``
    after seeding (run.py constructs the writer with a placeholder 0).
    """
    sp = scenario.solver
    dom = scenario.domain

    # --- units: convert the two SI-seconds deck fields to ms once (root §7) ---
    dt_deck_ms = sp.dt * 1.0e3
    total_ms = sp.total_time * 1.0e3

    # --- grid geometry (mm-ms-g) ---
    grid_res = sp.grid_resolution
    inv_dx = grid_res / (dom.xmax - dom.xmin)
    dx = 1.0 / inv_dx
    n_side = max(1, int(round(math.sqrt(sp.particles_per_cell))))
    spacing = dx / n_side

    seed = _seed(scenario, dx, spacing)
    n = seed["pos"].shape[0]
    writer.particle_count = n  # <-- the writer was built with a placeholder 0

    # --- CFL: pick a stable substep dt, decoupled from the (optimistic) deck dt ---
    c_max = 0.0
    for name in {scenario.projectile.material, *(l.material for l in scenario.armor)}:
        mat = materials.get(name)
        mu, lam = _lame(mat.youngs_modulus, mat.poisson_ratio)
        c_p = math.sqrt((lam + 2.0 * mu) / mat.density)  # p-wave speed, mm/ms
        c_max = max(c_max, c_p)
    dt_cfl = CFL * dx / c_max
    dt_sim = min(dt_deck_ms, dt_cfl)

    frame_dt_ms = total_ms / sp.frame_count
    substeps = max(1, math.ceil(frame_dt_ms / dt_sim))
    dt = frame_dt_ms / substeps  # even division so frame times land exactly

    if dt < dt_deck_ms:
        print(
            f"[mpm] deck dt={dt_deck_ms:.3e} ms exceeds CFL limit "
            f"{dt_cfl:.3e} ms (c_max={c_max:.0f} mm/ms, dx={dx:.4f} mm); "
            f"using dt={dt:.3e} ms, {substeps} substeps/frame"
        )
    print(
        f"[mpm] {n} particles, grid {grid_res} (dx={dx:.4f} mm), "
        f"{sp.frame_count} frames x {substeps} substeps"
    )

    # --- upload to Warp ---
    x = wp.array(seed["pos"], dtype=wp.vec2, device=device)
    v = wp.array(seed["vel"], dtype=wp.vec2, device=device)
    C = wp.zeros(n, dtype=wp.mat22, device=device)
    # F starts at identity.
    F0 = np.tile(np.eye(2, dtype=np.float32).reshape(1, 2, 2), (n, 1, 1))
    F = wp.array(F0, dtype=wp.mat22, device=device)
    mass = wp.array(seed["mass"], dtype=float, device=device)
    vol0 = wp.array(seed["vol0"], dtype=float, device=device)
    mu = wp.array(seed["mu"], dtype=float, device=device)
    lam = wp.array(seed["lam"], dtype=float, device=device)
    yield_k = wp.array(seed["yield"], dtype=float, device=device)
    dthr = wp.array(seed["dthr"], dtype=float, device=device)  # damage threshold
    brittle = wp.array(seed["brittle"], dtype=float, device=device)  # 1 brittle, 0 ductile
    reactive = wp.array(seed["reactive"], dtype=float, device=device)  # 1 ERA/NERA filler
    det_pressure = wp.array(seed["det_pressure"], dtype=float, device=device)  # MPa pulse
    burn_time = wp.array(seed["burn_time"], dtype=float, device=device)  # ms pulse duration
    ign_comp = wp.array(seed["ign_comp"], dtype=float, device=device)  # det(F) ignition thresh
    alpha = wp.zeros(n, dtype=float, device=device)  # equiv. plastic strain
    damage = wp.zeros(n, dtype=float, device=device)  # 0 intact, 1 spalled/reacting (latched)
    burn = wp.zeros(n, dtype=float, device=device)  # ms of detonation pulse remaining

    nx = grid_res + 3
    ny = int(math.ceil((dom.ymax - dom.ymin) * inv_dx)) + 3
    grid_v = wp.zeros((nx, ny), dtype=wp.vec2, device=device)
    grid_m = wp.zeros((nx, ny), dtype=float, device=device)
    origin = wp.vec2(float(dom.xmin), float(dom.ymin))
    # One-cell keep-out band for the _g2p position clamp (see the kernel).
    clamp_lo = wp.vec2(float(dom.xmin + dx), float(dom.ymin + dx))
    clamp_hi = wp.vec2(float(dom.xmax - dx), float(dom.ymax - dx))

    mat_id = seed["mat_id"]

    def substep():
        grid_v.zero_()
        grid_m.zero_()
        wp.launch(_p2g, dim=n, device=device, inputs=[
            x, v, C, F, mass, vol0, mu, lam, damage, reactive, det_pressure, burn,
            grid_v, grid_m, origin, inv_dx, dx, dt])
        wp.launch(_grid_op, dim=(nx, ny), device=device, inputs=[
            grid_v, grid_m, nx, ny])
        wp.launch(_g2p, dim=n, device=device, inputs=[
            x, v, C, F, grid_v, origin, inv_dx, dt, clamp_lo, clamp_hi])
        # Bound the reactive filler's post-transfer speed so the F-independent
        # detonation source can't accelerate unconfined debris to a CFL-breaking
        # ~14 km/s (reactive particles only; see REACTIVE_VMAX / _clamp_reactive_v).
        wp.launch(_clamp_reactive_v, dim=n, device=device, inputs=[
            v, reactive, REACTIVE_VMAX])
        # von Mises radial return caps deviatoric stress at yield -> plastic
        # flow (mushrooming/cratering) instead of pure elastic rebound. It runs
        # on spalled particles too: it pins their deviatoric F to the yield
        # surface so F can't blow up and NaN the stress readout. Reactive filler
        # is skipped (owns its own state machine).
        wp.launch(_return_mapping, dim=n, device=device, inputs=[
            F, mu, yield_k, reactive, alpha])
        # Latch damage: ductile metals once accumulated plastic strain crosses
        # threshold; brittle ceramics once the stress state reaches strength
        # (von Mises or tensile principal). _p2g then treats failed particles as
        # free fragments (spall / shatter). Reactive filler is skipped here.
        wp.launch(_update_damage, dim=n, device=device, inputs=[
            F, mu, lam, yield_k, brittle, reactive, alpha, dthr, damage])
        # Reactive impulse (ERA/NERA): ignite filler on shock arrival, age the
        # detonation pulse. Writes `burn` (read by next substep's _p2g) and the
        # reactive `damage` latch. No-op for non-reactive particles.
        wp.launch(_update_reactive, dim=n, device=device, inputs=[
            F, reactive, ign_comp, burn_time, burn, damage, dt])

    def dump_frame():
        pos = x.numpy()
        vel = v.numpy()
        Fn = F.numpy().reshape(n, 2, 2)
        dmg = damage.numpy()
        vel_mag = np.linalg.norm(vel, axis=1)
        stress = _von_mises(Fn, seed["mu"], seed["lam"])
        # A spalled particle carries no stress (it lost cohesion); zero the
        # readout so the colormap isn't skewed by a broken fragment's stale F.
        stress = np.where(dmg > 0.5, 0.0, stress)
        frame = np.column_stack([
            pos[:, 0], pos[:, 1], vel_mag, stress,
            dmg,  # 0 intact, 1 spalled (latched past damage_threshold)
            mat_id,
        ]).astype(np.float32)
        writer.write_frame(frame)

    # Frame 0 is the seeded state at t=0; then step and dump for the rest.
    dump_frame()
    for _ in range(sp.frame_count - 1):
        for _s in range(substeps):
            substep()
        wp.synchronize_device(device)
        dump_frame()
