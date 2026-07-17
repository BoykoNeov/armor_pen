"""The Mie-Grueneisen energy equation's guards — pinned as PROPERTIES.

Milestone 13 solves the energy balance in closed form (``mpm._g2p``), which is
possible because MG is linear in ``e``:

    rho0*e1*(1 + Gamma0*dJ/2) = rho0*e0 - [(p_cold(J1) + p0)/2 + q]*dJ

That denominator FACTOR, ``1 + Gamma0*dJ/2``, is the whole subject of this file.
It is 1 for dJ=0, falls as a substep compresses, vanishes at dJ = -2/Gamma0, and
goes NEGATIVE past that — which silently flips the sign of e. The ERA deck rode
that mode to e = -1.7e6 J/kg (against a physical ~1e5) and then to NaN across 98 %
of its particles, while the CFL audit printed "OK".

Two things are pinned here, and they pull in opposite directions:

  * the guard must FIRE before the factor can flip sign — else e inverts;
  * the guard must NOT fire on a resolved substep — else it is not a bug fix, it
    is a physics change, and it moves PHYSICS §3.10's piston and all 30 decks.

Both are asserted for EVERY material in the library, not for the one that
happened to be measured: rha is where the divergence was caught, but the energy
equation runs on every particle, and a threshold that is safe for Gamma0=1.93 is
not automatically safe for Gamma0=1.0.

Host-side and CPU-only: these are properties of the constants and the algebra, so
they need no GPU and run every time. They deliberately do NOT re-derive the
factor's formula — the point is the constants, and a second copy of the algebra
would just drift.
"""

import numpy as np
import pytest

from ballistics_solver import materials, mpm

wp = pytest.importorskip("warp")

# A generous bound on a resolved substep's volume change. The real decks are far
# below this: the ERA audit's own worst reading, div_v = -1.887e5 /ms at
# dt ~ 1.9e-7 ms, is dJ ~ -0.036, and that was a deck already BREACHING its CFL
# budget by 1.16x. 0.05 therefore over-states a healthy substep and the assertion
# is correspondingly conservative.
RESOLVED_DJ = 0.05


def _factor(g0: float, dJ: float) -> float:
    """The closed-form solve's denominator factor, ``1 + Gamma0*dJ/2``."""
    return 1.0 + g0 * dJ * 0.5


def _all_materials():
    """Every material in the library — the energy equation runs on all of them."""
    return list(materials.LIBRARY.values())


def test_the_guard_is_inert_on_every_resolved_substep_for_every_material():
    """A resolved substep must not trip the guard — for ANY material.

    This is the property that protects every existing result. The guard DROPS a
    substep's compression work when it fires, so a guard that fired in band would
    quietly stop conserving energy on decks that are behaving perfectly well.
    """
    for mat in _all_materials():
        g0 = mpm._mg_params(mat)["g0"]
        worst = _factor(g0, -RESOLVED_DJ)  # compression is the falling direction
        assert worst > mpm.MG_E_FACTOR_MIN, (
            f"{mat.name}: factor {worst:.4f} at a resolved dJ=-{RESOLVED_DJ} is at "
            f"or below MG_E_FACTOR_MIN={mpm.MG_E_FACTOR_MIN} — the guard would fire "
            f"on a HEALTHY substep and drop its work"
        )


def test_the_guard_fires_before_the_factor_can_flip_sign_for_every_material():
    """The guard's threshold must sit strictly above the factor's zero.

    If the threshold were at or below 0, the factor could reach 0 (division blows
    up) or go negative (e silently INVERTS) with the guard never firing. e < 0 is
    unphysical here by construction: from rest, compression gives dJ<0 with
    p_cold>0 and tension gives dJ>0 with p_cold<0, so rho0*de = -p*dJ is positive
    either way, and q >= 0 only ever adds.
    """
    assert mpm.MG_E_FACTOR_MIN > 0.0, "a non-positive threshold cannot guard a zero"
    for mat in _all_materials():
        g0 = mpm._mg_params(mat)["g0"]
        dJ_zero = -2.0 / g0  # where the factor vanishes
        dJ_fire = 2.0 * (mpm.MG_E_FACTOR_MIN - 1.0) / g0  # where the guard fires
        assert dJ_fire > dJ_zero, (
            f"{mat.name}: guard fires at dJ={dJ_fire:.4f} but the factor already "
            f"vanished at dJ={dJ_zero:.4f} — it cannot catch the sign flip"
        )


def test_the_factor_only_vanishes_past_a_total_one_substep_collapse():
    """Sanity on the physics: the guard's regime is unreachable by real material.

    The factor vanishes at dJ = -2/Gamma0. For every material here Gamma0 <= 2, so
    that is a volume change of 100 % or more IN A SINGLE SUBSTEP — a degenerate or
    inverted element, not a compression. This is what licenses treating the guard
    as a resolution backstop (the J_FLOOR posture) rather than a physical limit.
    """
    for mat in _all_materials():
        g0 = mpm._mg_params(mat)["g0"]
        assert abs(-2.0 / g0) >= 1.0, (
            f"{mat.name}: Gamma0={g0} puts the factor's zero at dJ={-2.0/g0:.4f}, "
            f"i.e. inside a sub-100 % volume change — the guard would then be "
            f"capping real physics rather than catching a degenerate element"
        )


def test_the_previous_guard_form_could_not_fire_and_must_not_come_back():
    """REGRESSION PIN, and it pins the REASONING, not just a number.

    The guard used to read:

        denom = rho0[p] * (1.0 + mg.g0 * dJ * 0.5)
        if wp.abs(denom) > 1.0e-12:   # <- tests the rho0-SCALED product

    Its comment promised it would "guard it rather than emit inf". It could not:
    rho0 for steel is 7.85e-3, so a 1e-12 threshold on the PRODUCT only fires once
    the factor is below ~1.3e-10. Everything between that and the sign flip sailed
    through amplified. Driven down nine pathological substeps it fired ZERO times
    and returned e = -6.0e13 — it emitted garbage where an inf would at least have
    been caught downstream.

    The lesson generalises past this one line: a threshold on a quantity carrying
    physical units is a threshold on the UNITS as much as on the number. The
    factor is dimensionless; rho0*factor is not. Guard the dimensionless one.
    """
    rho0 = materials.get("rha").density
    old_threshold = 1.0e-12

    # The factor at which the OLD form would finally have fired.
    old_fires_at = old_threshold / rho0
    assert old_fires_at < 1.0e-9, (
        "premise check: the old guard should only fire at an absurdly small factor"
    )

    # The sign flip it was written to catch: factor <= 0, i.e. rho0*factor <= 0.
    # At the moment of the flip the OLD test is |rho0*factor| > 1e-12, which is
    # TRUE for any factor of meaningful size — so the update proceeded.
    for factor in (-0.88, -0.0615, 0.001, 0.05):
        assert abs(rho0 * factor) > old_threshold, (
            f"the old guard did not fire at factor={factor}"
        )
        # ... while the current guard does, for every one of them.
        assert factor <= mpm.MG_E_FACTOR_MIN, (
            f"factor={factor} must be caught by MG_E_FACTOR_MIN="
            f"{mpm.MG_E_FACTOR_MIN}"
        )


def test_the_energy_equation_floors_J_exactly_as_the_pressure_does():
    """`dJ` and `p` must be conjugate: both on the floored state, or neither.

    Every pressure path floors J at J_FLOOR internally (``_mg_p_cold``). The energy
    equation used to take dJ from the RAW determinant, which charges the work
    -p*dJ at a pressure the model never had — below the floor the cold curve
    saturates while dJ keeps counting real volume change.

    Pinned by reading the source: the floor is three lines inside a Warp kernel
    with no host mirror to compare against, and a test that re-implemented the
    kernel would be pinning the copy. This asserts the shape that matters — that
    the determinants feeding dJ are floored — and fails loudly if someone removes
    it.
    """
    import inspect

    src = inspect.getsource(mpm._g2p.func)
    energy = src.split("energy equation (milestone 13")[1]
    assert "J_old = wp.max(J_old_raw, _J_FLOOR)" in energy, (
        "the energy equation's J_old is no longer floored — dJ and p are then "
        "computed on different states and the work term is charged at a pressure "
        "the model does not have"
    )
    assert "J_new = wp.max(J_new_raw, _J_FLOOR)" in energy, (
        "the energy equation's J_new is no longer floored (see J_old above)"
    )


# --- The REPORT's roundoff verdict --------------------------------------------
# The clamp's worst violation is only meaningful against `e`'s own scale. These pin
# the verdict, not the clamp: a wrong verdict does not corrupt a bake, it tells you
# to throw away a good one (or to ship a bad one), which is worse per-incident
# because it is believed.

# Measured on the milestone-13 rebake (MG + wall fix + AV on). The oblique deck is
# the one that matters: an absolute `abs(e_worst) < 1.0` verdict FAILED it while
# passing every other deck, including the jet whose `e` is 8.5x LARGER.
REBAKE = [
    # deck,                    e_max,    worst clamped e
    ("heat_vs_composite",      9.021e06, -3.450e-02),
    ("apfsds_vs_era",          1.503e06, -4.476e-01),
    ("apfsds_vs_nera",         1.003e07, -6.184e-01),
    ("apfsds_vs_era_oblique",  1.065e06, -1.178e+00),
]
REL_LIMIT = 1.0e-4  # must match the report in `bake`
F32_EPS = 1.19e-7


@pytest.mark.parametrize("deck,e_max,worst", REBAKE)
def test_the_roundoff_verdict_passes_every_deck_that_actually_baked_clean(deck, e_max,
                                                                          worst):
    """All four decks below are known-good bakes; the verdict must say so.

    Each ran to completion with the J floor and the resolution guard at EXACTLY 0 and
    passed validate_cache including §6.8 finiteness. Whatever the verdict is, it may
    not condemn these.
    """
    rel = abs(worst) / e_max
    assert rel < REL_LIMIT, (
        f"{deck}: verdict condemns a bake that ran clean — {rel:.2e} of e_max, "
        f"{rel / F32_EPS:.1f} float32 eps"
    )
    assert rel < 20 * F32_EPS, (
        f"{deck}: {rel / F32_EPS:.1f} eps is more cancellation than the energy "
        f"solve's handful of operations can explain — if this is real, the verdict "
        f"is not the thing to change"
    )


def test_an_ABSOLUTE_roundoff_threshold_is_anti_correlated_with_the_risk():
    """The regression pin. An absolute J/kg threshold must never come back.

    This is the decorative-guard defect again (see the guard-form test above): a
    threshold on a quantity carrying units is a threshold on the UNITS as much as on
    the number. Here it is not merely useless — it is INVERTED. It condemns the deck
    with the smallest `e` and clears the deck with the largest, because the
    cancellation it is trying to detect scales with `e` and the threshold does not.
    """
    by_deck = {d: (e_max, worst) for d, e_max, worst in REBAKE}
    condemned = [d for d, (e_max, w) in by_deck.items() if abs(w) >= 1.0]
    assert condemned, (
        "the shipped rebake data no longer reproduces the inversion this test pins"
    )
    # Derive the exemplar from the data — do NOT name a deck here. The first draft
    # of this test hardcoded the wrong one (the largest-e deck is apfsds_vs_nera at
    # 1.00e7, not the jet at 9.02e6) and the assertion caught it, which is the
    # method working; a named constant would just have to be maintained.
    cleared = [(d, e) for d, (e, w) in by_deck.items() if abs(w) < 1.0]
    biggest_cleared = max(cleared, key=lambda r: r[1])
    for d in condemned:
        assert by_deck[d][0] < biggest_cleared[1], (
            f"an absolute 1.0 J/kg threshold condemned {d} (e_max "
            f"{by_deck[d][0]:.3g}) while clearing {biggest_cleared[0]} (e_max "
            f"{biggest_cleared[1]:.3g}, {biggest_cleared[1] / by_deck[d][0]:.1f}x "
            f"LARGER) — it fires on the smaller-e deck, which is exactly backwards"
        )
    # And the relative verdict gets both right, which is the whole point.
    for deck, e_max, worst in REBAKE:
        assert abs(worst) / e_max < REL_LIMIT, f"{deck} must pass the relative form"


def test_the_verdict_still_catches_a_real_negative_drive():
    """The other half: relative must not be so loose it passes the actual failure.

    The pathological paths that exposed the old decorative guard returned
    e = -6.0e13 against a physical ~1e5-1e7 J/kg, and the pre-fix ERA bake reported
    -inf. A verdict that cannot condemn those is decoration in the other direction.
    """
    for drive in (-6.0e13, -1.7e06, float("-inf")):
        for e_max in (1.0e05, 9.0e06):
            rel = abs(drive) / e_max
            assert rel >= REL_LIMIT, (
                f"a real negative drive e={drive:.3g} against e_max={e_max:.3g} "
                f"({rel:.2e}) slips past the {REL_LIMIT:.0e} verdict"
            )
