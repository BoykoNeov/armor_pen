"""MLS-MPM transfer kernels (NVIDIA Warp) — the physics core.

STATUS: milestone 2 — **elastic + von Mises plasticity** MLS-MPM. Two materials
(KE rod + armor plate) are seeded from the scenario and run through a P2G /
grid-update / G2P substep cycle with fixed-corotated hyperelasticity, followed
by a perfectly-plastic von Mises radial return (``_return_mapping``) that caps
deviatoric stress at each material's yield and lets metal flow. The rod now
**mushrooms** and the plate **craters/bulges plastically** — but there is still
**no perforation hole**: material separation needs the *damage/spall* milestone
(build order, root §9), where particles detach past ``damage_threshold``. A
cratered plate with no clean hole is the correct milestone-2 result, not a bug.

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


@wp.kernel
def _return_mapping(
    F: wp.array(dtype=wp.mat22),
    mu: wp.array(dtype=float),
    yield_k: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
):
    """Perfectly-plastic von Mises radial return in log-strain (Hencky) space.

    Applied per particle after G2P. SVD the (elastic) deformation gradient,
    take principal log-strains, split off the volumetric part, and radially
    return the deviatoric part onto the yield surface. Plastic flow is
    isochoric (the volumetric log-strain is untouched) — correct for metals.
    Accumulates equivalent plastic strain into ``alpha`` (feeds the damage
    milestone; unused this milestone). No hardening: perfectly plastic.
    """
    p = wp.tid()
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

    # Fixed-corotated Kirchhoff stress term P(F) F^T (root §6, PHYSICS §3).
    J = wp.determinant(F[p])
    R = _polar_r(F[p])
    PFt = 2.0 * mu[p] * (F[p] - R) * wp.transpose(F[p]) + wp.mat22(
        lam[p] * (J - 1.0) * J, 0.0, 0.0, lam[p] * (J - 1.0) * J
    )
    stress = (-dt * vol0[p] * 4.0 * inv_dx * inv_dx) * PFt
    affine = stress + mass[p] * C[p]

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
        # Sticky reflecting domain walls (a few cells thick). No gravity: over a
        # ~0.04 ms window it is utterly negligible next to impact stresses.
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
    x[p] = x[p] + dt * new_v
    F[p] = (wp.mat22(1.0, 0.0, 0.0, 1.0) + dt * new_C) * F[p]


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


def _seed(scenario, dx: float, spacing: float):
    """Seed the projectile and armor stack into particle arrays (numpy).

    Geometry is illustrative (root §10): the projectile is a rectangle aimed
    +x at the front face of the armor stack; each armor layer is a full-height
    (with margin) slab, laid front-to-back with any standoff gap.

    Returns a dict of numpy arrays: pos, vel, mu, lam, mass, vol0, mat_id.
    """
    dom = scenario.domain
    proj = scenario.projectile
    height = dom.ymax - dom.ymin
    y_center = 0.5 * (dom.ymin + dom.ymax)

    # Armor front face sits mid-domain; layers march +x from there.
    armor_front = dom.xmin + 0.5 * (dom.xmax - dom.xmin)
    margin = 0.1 * height  # keep the plate off the domain walls

    pos_list, vel_list, mu_list, lam_list, yield_list, mass_list, vol_list, mid_list = (
        [], [], [], [], [], [], [], [],
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
        mass_list.append(np.full(n, mat.density * p_vol))
        vol_list.append(np.full(n, p_vol))
        mid_list.append(np.full(n, float(mat.material_id)))

    # Projectile: leading (+x) tip a small gap before the armor front.
    gap = 3.0 * dx
    tip_x = armor_front - gap
    rod = _fill_rect(
        tip_x - proj.length, tip_x,
        y_center - 0.5 * proj.diameter, y_center + 0.5 * proj.diameter,
        spacing,
    )
    ang = math.radians(proj.angle_deg)
    add_region(rod, materials.get(proj.material),
               proj.velocity * math.cos(ang), -proj.velocity * math.sin(ang))

    # Armor stack, front to back.
    x_cursor = armor_front
    for layer in scenario.armor:
        x_cursor += layer.standoff
        slab = _fill_rect(
            x_cursor, x_cursor + layer.thickness,
            dom.ymin + margin, dom.ymax - margin,
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
    alpha = wp.zeros(n, dtype=float, device=device)  # equiv. plastic strain

    nx = grid_res + 3
    ny = int(math.ceil((dom.ymax - dom.ymin) * inv_dx)) + 3
    grid_v = wp.zeros((nx, ny), dtype=wp.vec2, device=device)
    grid_m = wp.zeros((nx, ny), dtype=float, device=device)
    origin = wp.vec2(float(dom.xmin), float(dom.ymin))

    mat_id = seed["mat_id"]

    def substep():
        grid_v.zero_()
        grid_m.zero_()
        wp.launch(_p2g, dim=n, device=device, inputs=[
            x, v, C, F, mass, vol0, mu, lam, grid_v, grid_m, origin, inv_dx, dx, dt])
        wp.launch(_grid_op, dim=(nx, ny), device=device, inputs=[
            grid_v, grid_m, nx, ny])
        wp.launch(_g2p, dim=n, device=device, inputs=[
            x, v, C, F, grid_v, origin, inv_dx, dt])
        # von Mises radial return caps deviatoric stress at yield -> plastic
        # flow (mushrooming/cratering) instead of pure elastic rebound.
        wp.launch(_return_mapping, dim=n, device=device, inputs=[
            F, mu, yield_k, alpha])

    def dump_frame():
        pos = x.numpy()
        vel = v.numpy()
        Fn = F.numpy().reshape(n, 2, 2)
        vel_mag = np.linalg.norm(vel, axis=1)
        stress = _von_mises(Fn, seed["mu"], seed["lam"])
        frame = np.column_stack([
            pos[:, 0], pos[:, 1], vel_mag, stress,
            np.zeros(n, dtype=np.float32),  # damage: 0 until the damage milestone
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
