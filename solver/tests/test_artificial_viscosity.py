"""Pin the artificial-viscosity term's load-bearing PROPERTIES, not its values.

AV (milestone 11) ships DEFAULT OFF, and the property that makes that claim true
— zero coefficients => identically zero stress — is the first thing tested here,
because it is what guarantees the 30 already-baked decks are untouched. If it
breaks, every cache in the repo silently stops matching its documented numbers.

The other two are physics guards:
  * q must vanish in EXPANSION. An artificial viscosity that resists material
    coming apart would glue a spall crack open — precisely the physics this repo
    exists to show.
  * q must be a POSITIVE pressure in compression, i.e. a NEGATIVE-diagonal
    Kirchhoff stress, matching the EOS branch it sits beside. A sign flip here
    would pump energy into the shock instead of removing it.

Runs on CPU — constitutive assertions, no bake, no GPU.
"""

import numpy as np
import pytest

from ballistics_solver import materials, mpm

wp = pytest.importorskip("warp")


@pytest.fixture(scope="module", autouse=True)
def _warp_cpu():
    wp.init()


@wp.func
def _av_xx(F: wp.mat22, C: wp.mat22, mu: float, lam: float, rho0: float,
           l: float, c_q: float, c_l: float):
    return mpm._av_tau(F, C, mu, lam, rho0, l, c_q, c_l)[0, 0]


@wp.kernel
def _av_kernel(
    F: wp.array(dtype=wp.mat22),
    C: wp.array(dtype=wp.mat22),
    mu: float,
    lam: float,
    rho0: float,
    l: float,
    c_q: float,
    c_l: float,
    out: wp.array(dtype=float),
):
    i = wp.tid()
    out[i] = _av_xx(F[i], C[i], mu, lam, rho0, l, c_q, c_l)


def _run(Fs, Cs, mat_name="copper_jet", c_q=1.5, c_l=0.5, l=0.39):
    mat = materials.get(mat_name)
    mu_s, lam_s = mpm._lame(mat.youngs_modulus, mat.poisson_ratio)
    n = len(Fs)
    out = wp.zeros(n, dtype=float, device="cpu")
    wp.launch(
        _av_kernel, dim=n, device="cpu",
        inputs=[
            wp.array(np.asarray(Fs, dtype=np.float32), dtype=wp.mat22, device="cpu"),
            wp.array(np.asarray(Cs, dtype=np.float32), dtype=wp.mat22, device="cpu"),
            mu_s, lam_s, mat.density, l, c_q, c_l, out,
        ],
    )
    return out.numpy()


# A spread of compression rates (trace C < 0) and expansion rates (trace C > 0).
_F = [np.eye(2), np.diag([0.9, 0.9]), np.diag([0.7, 0.7])]
_C_COMPRESS = [np.diag([-500.0, -500.0]), np.diag([-3000.0, -600.0]),
               np.array([[-2000.0, 120.0], [-40.0, -900.0]])]
_C_EXPAND = [np.diag([500.0, 500.0]), np.diag([3000.0, 600.0]),
             np.array([[2000.0, 120.0], [-40.0, 900.0]])]


def test_zero_coefficients_give_identically_zero():
    """DEFAULT OFF must mean OFF — the guarantee that no baked deck moves.

    av_c_q = av_c_l = 0 is the shipped default. If this returns anything nonzero,
    every one of the repo's 30 caches silently diverges from its documented
    figures on the next rebake.
    """
    got = _run(_F * 3, _C_COMPRESS + _C_EXPAND + _C_COMPRESS, c_q=0.0, c_l=0.0)
    assert np.all(got == 0.0), f"zeroed AV still produced stress: {got}"


def test_expansion_produces_no_viscosity():
    """q = 0 whenever material is coming apart — never glue a spall crack open."""
    got = _run(_F, _C_EXPAND)
    assert np.all(got == 0.0), f"AV fired in expansion: {got}"


def test_compression_gives_positive_pressure():
    """Compression => positive pressure => NEGATIVE-diagonal Kirchhoff stress.

    Same sign convention as the EOS branch (tau_vol = -p*J*I). A flip would make
    the term an anti-damper.
    """
    got = _run(_F, _C_COMPRESS)
    assert np.all(got < 0.0), f"AV did not resist compression: {got}"


def test_viscosity_grows_with_compression_rate():
    """Monotone in |div v| — both the linear and quadratic terms are increasing."""
    rates = [-100.0, -500.0, -2000.0, -5000.0]
    got = _run([np.eye(2)] * len(rates), [np.diag([r, r]) for r in rates])
    mag = -got  # to positive pressure
    assert np.all(np.diff(mag) > 0.0), f"AV not monotone in compression rate: {mag}"


def test_quadratic_term_dominates_at_high_rate():
    """c_q scales as (div v)^2 and c_l as |div v|, so the quadratic wins as rate grows.

    Pins the two coefficients to their intended ROLES: without this, someone could
    swap them and every published coefficient would mean the other thing.
    """
    slow = np.diag([-50.0, -50.0])
    fast = np.diag([-5000.0, -5000.0])
    F = [np.eye(2)]
    q_only_slow = -_run(F, [slow], c_q=1.5, c_l=0.0)[0]
    l_only_slow = -_run(F, [slow], c_q=0.0, c_l=0.5)[0]
    q_only_fast = -_run(F, [fast], c_q=1.5, c_l=0.0)[0]
    l_only_fast = -_run(F, [fast], c_q=0.0, c_l=0.5)[0]
    # A 100x rate increase multiplies the linear term by 100 and the quadratic by
    # 100^2, so the quadratic/linear ratio must grow by ~100x.
    assert q_only_slow < l_only_slow, "quadratic should be negligible at low rate"
    assert (q_only_fast / l_only_fast) > 50.0 * (q_only_slow / l_only_slow)


def test_av_is_absent_from_the_constitutive_law():
    """`_fixed_corotated_pft` must NOT contain AV — the whole placement argument.

    AV is a NUMERICAL device. If it leaked into the constitutive law it would feed
    the brittle fracture triggers (shattering ceramic for a numerical reason) and
    break the Warp/NumPy two-path pin in test_stress_paths.py, since the host
    mirror `_von_mises` sees only F and cannot know div v. Guard: the law takes no
    velocity-gradient argument at all.
    """
    import inspect
    src = inspect.getsource(mpm._fixed_corotated_pft.func)
    assert "_av_tau" not in src, "artificial viscosity leaked into the constitutive law"
    assert "div_v" not in src, "constitutive law gained a rate dependence"
