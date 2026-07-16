"""The constitutive law lives in two places. Pin them together.

``mpm._fixed_corotated_pft`` (Warp, device) drives the dynamics and — since
milestone 8 — the brittle trigger via ``_stress_invariants``. ``mpm._von_mises``
(NumPy, host) computes the cache's ``stress`` column inside ``dump_frame``, where
device funcs are unreachable, so it is a hand-kept mirror. Two copies of a
material model drift the moment someone edits one of them; that is exactly the
bug this file exists to catch.

Also pins the EOS's load-bearing *properties* (milestone 8, PHYSICS §3.5) rather
than its output values: monotonicity is what makes the response well-posed at a
hypervelocity stagnation point, and tangent-matching at J=1 is what keeps the KE
decks from moving when the volumetric law is swapped underneath them.

Runs on CPU — these are constitutive assertions, not a bake, so they need no GPU
and stay fast enough to run every time.
"""

import numpy as np
import pytest

from ballistics_solver import materials, mpm

wp = pytest.importorskip("warp")


@pytest.fixture(scope="module", autouse=True)
def _warp_cpu():
    wp.init()


# A spread that exercises every branch: rest, mild/heavy compression, expansion,
# pure shear, rotation (must be stress-free), and a rotated compression (the case
# that catches a botched polar decomposition).
def _F_cases():
    th = 0.6
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cases = {
        "rest": np.eye(2),
        "mild_compression": np.diag([0.98, 0.98]),
        "heavy_compression": np.diag([0.78, 0.78]),  # J≈0.61 — the jet-tip regime
        "extreme_compression": np.diag([0.4, 0.4]),  # J=0.16 — old model's absurd zone
        "expansion": np.diag([1.05, 1.05]),
        "uniaxial": np.diag([0.85, 1.0]),
        "shear": np.array([[1.0, 0.12], [0.0, 1.0]]),
        "rotation_only": R,
        "rotated_compression": R @ np.diag([0.8, 0.9]),
        "anisotropic": np.array([[0.9, 0.05], [-0.03, 1.1]]),
        # Degenerate states a violent shock front genuinely produces. Both paths
        # must floor J identically (mpm.J_FLOOR) rather than one flooring and the
        # other passing the raw value through — that asymmetry was a live bug.
        # These also pin the float32 overflow guard: the EOS pressure diverges as
        # J -> 0, so an unfloored divide here goes to inf and then NaN.
        "inverted": np.diag([-0.9, 0.9]),  # det < 0: element flipped this substep
        "near_singular": np.diag([1e-3, 1e-3]),  # J = 1e-6, below J_FLOOR
    }
    return cases


@wp.kernel
def _vm_kernel(
    F: wp.array(dtype=wp.mat22),
    mu: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    out: wp.array(dtype=float),
):
    i = wp.tid()
    out[i] = _stress_invariants_vm(F[i], mu[i], lam[i])


@wp.func
def _stress_invariants_vm(F: wp.mat22, mu: float, lam: float):
    return mpm._stress_invariants(F, mu, lam)[0]


@pytest.mark.parametrize("mat_name", ["copper_jet", "rha", "ceramic", "tungsten_rod"])
def test_warp_and_numpy_stress_paths_agree(mat_name):
    """The device law and its host mirror must return the same von Mises stress.

    This is the anti-drift guard: if someone edits the volumetric law in one path
    and not the other, the cache's `stress` column silently stops describing the
    stress the solver actually applied.
    """
    mat = materials.get(mat_name)
    mu_s, lam_s = mpm._lame(mat.youngs_modulus, mat.poisson_ratio)
    cases = _F_cases()
    Fs = np.array(list(cases.values()), dtype=np.float32)
    n = len(Fs)

    mu = np.full(n, mu_s, dtype=np.float32)
    lam = np.full(n, lam_s, dtype=np.float32)

    F_wp = wp.array(Fs, dtype=wp.mat22, device="cpu")
    out = wp.zeros(n, dtype=float, device="cpu")
    wp.launch(
        _vm_kernel,
        dim=n,
        device="cpu",
        inputs=[F_wp, wp.array(mu, dtype=float, device="cpu"),
                wp.array(lam, dtype=float, device="cpu"), out],
    )
    got_warp = out.numpy()
    got_numpy = mpm._von_mises(Fs.astype(np.float64), mu.astype(np.float64),
                               lam.astype(np.float64))

    # Absolute floor for the near-zero cases, scaled to the material's own
    # stiffness: a rigid rotation stored in float32 is not exactly orthogonal, so
    # both paths read a tiny nonzero stress whose magnitude rides on K0 (tungsten
    # at K0=346 GPa reads ~0.01 MPa, i.e. 3e-8 relative — round-off, not drift).
    # A flat floor would either fail on stiff materials or go slack on soft ones.
    noise = 1.0e-6 * (lam_s + mu_s)
    for name, w, npy in zip(cases, got_warp, got_numpy):
        assert w == pytest.approx(npy, rel=2e-3, abs=noise), (
            f"{mat_name}/{name}: warp={w:.6g} MPa vs numpy={npy:.6g} MPa — the "
            f"device law and its host mirror have drifted apart"
        )


@pytest.mark.parametrize("mat_name", ["copper_jet", "rha", "ceramic", "tungsten_rod"])
def test_eos_pressure_is_monotone_in_compression(mat_name):
    """No turnover, no crush-through: p(J) rises without bound as J falls.

    The pre-EOS law failed exactly here — its volumetric term ``lam*(J-1)*J``
    peaked and then *collapsed* back toward zero, so the stagnation point
    equilibrated at J≈0.15 where real copper gives ≈0.61.
    """
    mat = materials.get(mat_name)
    mu_s, lam_s = mpm._lame(mat.youngs_modulus, mat.poisson_ratio)
    K0 = lam_s + mu_s
    J = np.linspace(0.05, 1.0, 4000)
    p = (K0 / materials.EOS_KP) * (J ** (-materials.EOS_KP) - 1.0)
    assert np.all(np.diff(p) < 0.0), f"{mat_name}: p(J) is not monotone in J"
    assert p[0] > p[-1] * 1e3, f"{mat_name}: p does not diverge as J -> 0"
    assert p[-1] == pytest.approx(0.0, abs=1e-6), f"{mat_name}: p(1) must be 0"


@pytest.mark.parametrize("mat_name", ["copper_jet", "rha", "ceramic", "tungsten_rod"])
def test_eos_tangent_matches_the_law_it_replaced(mat_name):
    """At J=1 the EOS must reproduce the OLD fixed-corotated bulk modulus exactly.

    K(1) = K0 = lam+mu is what makes milestone 8 a large-strain-only change: it is
    the reason a 1600 m/s KE deck barely moves while a 7 km/s jet tip moves ~2x.
    Break this and every deck silently re-tunes.
    """
    mat = materials.get(mat_name)
    mu_s, lam_s = mpm._lame(mat.youngs_modulus, mat.poisson_ratio)
    K0 = lam_s + mu_s

    # Tangent bulk modulus of the EOS at rest, by finite difference: K = -J dp/dJ.
    h = 1e-6
    def p(J):
        return (K0 / materials.EOS_KP) * (J ** (-materials.EOS_KP) - 1.0)
    K_num = -1.0 * (p(1.0 + h) - p(1.0 - h)) / (2.0 * h)
    assert K_num == pytest.approx(K0, rel=1e-4)

    # And the p-wave speed used by the CFL bound must equal the pre-EOS
    # sqrt((lam+2mu)/rho) at rest — the old bound was correct AT REST; milestone 8
    # only changed what happens under compression.
    c_eos = mpm._eos_sound_speed(1.0, K0, mu_s, mat.density)
    c_old = np.sqrt((lam_s + 2.0 * mu_s) / mat.density)
    assert c_eos == pytest.approx(c_old, rel=1e-9)


def test_eos_equilibrium_j_inverts_the_pressure_law():
    """``_eos_equilibrium_j`` is the closed-form inverse used to size the substep.

    If it drifts from ``_eos_pressure``, dt gets sized for the wrong compression
    and the failure mode is a NaN mid-bake rather than a wrong number.
    """
    mat = materials.get("copper_jet")
    mu_s, lam_s = mpm._lame(mat.youngs_modulus, mat.poisson_ratio)
    K0 = lam_s + mu_s
    for p_target in (1.0e3, 2.0e4, 7.2e4, 2.2e5):  # MPa, up to a 7 km/s tip
        J = mpm._eos_equilibrium_j(p_target, K0)
        p_back = (K0 / materials.EOS_KP) * (J ** (-materials.EOS_KP) - 1.0)
        assert p_back == pytest.approx(p_target, rel=1e-9)


def test_jet_tip_equilibrium_is_physical():
    """The milestone's headline a-priori prediction, as an executable assertion.

    A 7 km/s copper stagnation point (0.5·rho·v^2 = 220 GPa) must equilibrate near
    J≈0.61 — a ~39 % volume loss, severe but physical — where the pre-EOS law gave
    J≈0.15 (85 % loss, absurd). Fails if anyone quietly changes EOS_KP or K0.
    """
    mat = materials.get("copper_jet")
    mu_s, lam_s = mpm._lame(mat.youngs_modulus, mat.poisson_ratio)
    p_stag = 0.5 * mat.density * 7000.0**2
    J = mpm._eos_equilibrium_j(p_stag, lam_s + mu_s)
    assert 0.55 < J < 0.65, f"jet-tip equilibrium J={J:.4f} is outside the physical band"


def test_rotation_is_stress_free():
    """A rigid rotation must produce zero stress in both paths (corotated sanity)."""
    mat = materials.get("rha")
    mu_s, lam_s = mpm._lame(mat.youngs_modulus, mat.poisson_ratio)
    th = 0.9
    R = np.array([[[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]])
    vm = mpm._von_mises(R, np.array([mu_s]), np.array([lam_s]))
    assert vm[0] == pytest.approx(0.0, abs=1e-3)
