"""The a-priori CFL substep bound must be sized on the EOS's PHYSICAL branch.

WHY THIS FILE EXISTS. `EOS_CFL_J_MARGIN` multiplied `J` — a volume RATIO — to make
headroom: `Jd = 0.35 * J_eq`. That reads like "35 % of margin" and is not. `J` lives
in (0,1] with the EOS diverging as J -> 0, so scaling it is violently nonlinear, and
on ALL 30 DECKS, IN EVERY MATERIAL, the design state landed past the material's own
Mie-Grueneisen pole and below its guard switch. The bound was therefore read off the
guard's extrapolated Murnaghan `J^-4` backstop — the branch `J_FLOOR`'s own comment
calls "a degeneracy backstop, NOT a physical limit".

Concretely: `rha` equilibrates at J_eq=0.902 under `apfsds_vs_rha`'s impact — a 10 %
compression — and the bound designed for J=0.316, past rha's pole at J=1-1/s=0.329,
where it read a 137 000 mm/ms sound speed for a material whose real shocked speed is
~6 000. That is ~20x, and it is why milestone 13 measured decks using 5-22 % of their
own budget.

The defect was NEVER visible as a failure. Every bake validated clean and stayed
finite, because the error was in the SAFE direction — it bought a substep far smaller
than needed. That is precisely the shape of defect this repo keeps finding (see the
memory note on instruments that cannot see the failure): green because nothing looked,
not because the answer was right. A margin that over-predicts by 20x on today's decks
is not "conservative", it is UNCALIBRATED — and an uncalibrated bound is free to
under-predict on the next deck, which is what actually happened at margin 0.8 on
`heat_vs_composite`.

WHAT IS PINNED HERE, and why each is written the way it is:

  1. `test_design_state_is_on_the_physical_eos_branch` — the property that was false.
     Written against `_mg_params`'s `J_sw` and the material's OWN pole (`1 - 1/s`),
     derived from `materials.py`, NOT by re-running the sizing code's arithmetic. A
     test that recomputed `Jd` the way `bake` does would be satisfied by copying the
     bug. VERIFIED TO FAIL on the old formula before being trusted (see the module
     docstring in the boundary-wall test for the same posture).
  2. `test_impact_pressure_*` — the impedance match is the new scale, so its two
     textbook properties (symmetric-impact closed form; monotone in v) are pinned
     independently of the EOS.
  3. `test_margin_is_a_pressure_not_a_ratio` — a regression guard on the SEMANTICS.
     If someone reintroduces a multiplier on `J`, raising the constant would make the
     bound looser; here raising it must make the bound TIGHTER. That sign is the whole
     difference between the two designs, so it gets a test.

These are all host-side arithmetic — no GPU, no bake.
"""
from pathlib import Path

import pytest

from ballistics_solver import config, materials, mpm

# Every shipped deck, as (name, path) — same source of truth `test_config.py` uses.
# Globbed rather than listed so a new deck is covered the moment it lands, which is
# the point: the defect this file pins was uniform across all 30 and nobody looked.
SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
ALL_DECKS = sorted((p.stem, p) for p in SCENARIOS.glob("*.yaml"))


def _k0(mat):
    mu, lam = mpm._lame(mat.youngs_modulus, mat.poisson_ratio)
    return lam + mu


def _deck_design_j(scenario, name):
    """The design volume ratio `bake` would pick for material `name` in `scenario`.

    Mirrors the sizing chain's INPUTS (impact pressure, margin) but asserts against
    limits derived from `materials.py`, not against the chain's own output.
    """
    proj = materials.get(scenario.projectile.material)
    v = scenario.projectile.velocity
    names = {scenario.projectile.material, *(a.material for a in scenario.armor)}
    p_design = mpm._scenario_cfl_margin(scenario) * max(
        [mpm._impact_pressure(proj, materials.get(nm), v) for nm in names]
        + [mpm._impact_pressure(proj, proj, v)]
    )
    mat = materials.get(name)
    return mpm._eos_equilibrium_j(p_design, _k0(mat), mpm._mg_params(mat))


@pytest.mark.parametrize("deck_name,deck_path", ALL_DECKS)
def test_design_state_is_on_the_physical_eos_branch(deck_name, deck_path):
    """No deck may size its substep from the pole guard's extrapolated backstop.

    THE ONE EXCEPTION IS NAMED, NOT WAIVED. `apfsds_vs_nera` overrides the margin
    because its binding particles are a kinematic vise at `nera_filler`'s pole, and
    a design state inside the guard is the CORRECT answer there (PHYSICS §3.6.2 said
    so a priori). Every other deck sizing from the guard is the milestone-14 defect.
    """
    scenario = config.load_scenario(deck_path)
    names = {scenario.projectile.material, *(a.material for a in scenario.armor)}

    for name in sorted(names):
        mat = materials.get(name)
        mgp = mpm._mg_params(mat)
        Jd = _deck_design_j(scenario, name)
        pole = 1.0 - 1.0 / mat.shock.s  # where p_H's denominator (1 - s*eta) vanishes

        if deck_name == "apfsds_vs_nera" and name == "nera_filler":
            # The vise, priced in the deck. Designing INSIDE the guard is the CORRECT
            # answer here and is what the override buys: this filler measurably GOES
            # there (worst live J=0.5385 vs J_sw=0.550, 2-4 live particles), so a
            # bound that refused to look there would be sized for a state the deck
            # does not occupy.
            #
            # NOTHING IS ASSERTED ABOUT THE POLE, deliberately. It is tempting to
            # demand Jd > pole ({pole:.4f}) on the grounds that the Hugoniot fit has
            # no meaning past its own singularity — but the design state is not doing
            # physics, it is pricing a sound speed, and below J_sw the law IS the
            # guard's Murnaghan branch: pole-free, monotone, and the SAME branch
            # `_mg_sound` hands the kernel for a particle at that J. Designing where
            # the kernel computes is self-consistent by construction. The pole is a
            # feature of the formula the guard replaces, so it does not constrain
            # this. (The milestone-14 defect was never "past the pole" per se — it
            # was sizing 30 decks from a branch NONE of their materials reach.)
            assert Jd < mgp["J_sw"], (
                f"{deck_name}/{name}: the nera override exists to design INSIDE the "
                f"guard (Jd={Jd:.4f} should be < J_sw={mgp['J_sw']:.4f}). If this "
                f"fails the override is no longer doing its job."
            )
            continue

        assert Jd >= mgp["J_sw"], (
            f"{deck_name}/{name}: design J={Jd:.4f} is below the pole guard's "
            f"switch J_sw={mgp['J_sw']:.4f} (pole at {pole:.4f}), so the substep is "
            f"sized from the guard's extrapolated Murnaghan backstop rather than "
            f"from the Mie-Grueneisen law. That is the milestone-14 defect: it is "
            f"NOT a stability risk (it over-predicts, ~20x), it is an UNCALIBRATED "
            f"bound, and an uncalibrated bound is free to under-predict on the next "
            f"deck. Size from a PRESSURE (EOS_CFL_P_MARGIN), never from a multiple "
            f"of J."
        )


@pytest.mark.parametrize("deck_name,deck_path", ALL_DECKS)
def test_design_state_is_actually_compressed(deck_name, deck_path):
    """...and the other direction: the bound must still be a bound.

    The branch test above only stops the formula being too STIFF. A margin that
    designed for J~1 would pass it and size dt from the rest sound speed, which is
    the pre-milestone-8 bug. Pin that the design state is genuinely compressed.
    """
    scenario = config.load_scenario(deck_path)
    names = {scenario.projectile.material, *(a.material for a in scenario.armor)}
    for name in sorted(names):
        Jd = _deck_design_j(scenario, name)
        assert Jd < 0.98, (
            f"{deck_name}/{name}: design J={Jd:.4f} is essentially uncompressed, so "
            f"dt is being sized from the REST sound speed — the defect milestone 8 "
            f"fixed. The EOS stiffens under compression; the bound must anticipate it."
        )


@pytest.mark.parametrize("mat_name", ["tungsten_rod", "rha", "copper_jet", "ceramic"])
def test_impact_pressure_matches_the_symmetric_closed_form(mat_name):
    """Symmetric impact has an exact answer; the bisection must find it.

    When a material strikes ITSELF, symmetry forces the interface to the mean
    velocity, so each side sees particle velocity v/2 and the contact pressure is
    `p_H(v/2)` in closed form. No fitting, no tolerance games — this is the one case
    where the impedance match is analytic, so it is the one that can check the solver.
    """
    mat = materials.get(mat_name)
    for v in (1500.0, 3500.0, 7000.0):
        got = mpm._impact_pressure(mat, mat, v)
        want = mpm._hugoniot_pressure(mat, v / 2.0)
        assert got == pytest.approx(want, rel=1e-6), (
            f"{mat_name} at {v} mm/ms: symmetric impact must equal p_H(v/2)"
        )


@pytest.mark.parametrize("mat_name", ["rha", "copper_jet", "tungsten_rod"])
def test_impact_pressure_is_monotone_in_velocity(mat_name):
    """Faster impact, higher contact pressure. Cheap, and it catches a bracket flip."""
    proj = materials.get("tungsten_rod")
    mat = materials.get(mat_name)
    last = -1.0
    for v in (1500.0, 2500.0, 3500.0, 5000.0, 7000.0):
        p = mpm._impact_pressure(proj, mat, v)
        assert p > last, f"{mat_name}: impact pressure must rise with v"
        last = p


@pytest.mark.parametrize("deck_name,deck_path", ALL_DECKS)
def test_deck_design_pressure_exceeds_the_stagnation_scale(deck_name, deck_path):
    """The milestone-14 premise, pinned on the quantity the sizing actually uses.

    The old bound's scale was `1/2*rho_proj*v^2`. The claim is that the real contact
    shock is HIGHER, and higher by a velocity-dependent factor no constant absorbs.

    THIS IS ASSERTED DECK-WIDE, NOT PER PAIR, and the distinction is physical rather
    than bookkeeping. A single pair can sit BELOW p_stag: dense tungsten into rha at
    7 km/s matches at 0.87x, because a lower-impedance target cannot sustain the
    striker's stagnation pressure. What cannot fall below it is the deck's WORST
    pair, which always includes the projectile striking its own material (symmetric,
    p_H(v/2)) — and that is exactly what `bake` sizes from. An earlier cut of this
    test asserted the per-pair version and failed on rha and copper_jet at 7000; the
    test was wrong, not the code.
    """
    scenario = config.load_scenario(deck_path)
    proj = materials.get(scenario.projectile.material)
    v = scenario.projectile.velocity
    names = {scenario.projectile.material, *(a.material for a in scenario.armor)}
    p_imp = max(
        [mpm._impact_pressure(proj, materials.get(nm), v) for nm in names]
        + [mpm._impact_pressure(proj, proj, v)]
    )
    p_stag = 0.5 * proj.density * v * v
    assert p_imp > p_stag, (
        f"{deck_name}: deck-wide impact pressure {p_imp:.0f} MPa should exceed the "
        f"steady stagnation scale {p_stag:.0f} MPa the old bound sized from. If this "
        f"fails, milestone 14's premise — that the old scale was LOW, not merely "
        f"under-margined — is wrong for this deck."
    )


def test_margin_is_a_pressure_not_a_ratio():
    """Raising the margin must TIGHTEN the bound. The old constant did the reverse.

    This is the semantic regression guard. `EOS_CFL_J_MARGIN` scaled J, so a bigger
    value meant a LESS compressed design state and a LOOSER bound; `EOS_CFL_P_MARGIN`
    scales pressure, so a bigger value must mean a MORE compressed design state and a
    TIGHTER one. Anyone reintroducing a J-multiplier flips this sign.
    """
    mat = materials.get("rha")
    K0, mgp = _k0(mat), mpm._mg_params(mat)
    p_base = mpm._impact_pressure(materials.get("tungsten_rod"), mat, 1600.0)

    j_prev = 1.0
    for margin in (1.0, 2.0, 3.0, 6.0):
        Jd = mpm._eos_equilibrium_j(margin * p_base, K0, mgp)
        assert Jd < j_prev, (
            f"margin {margin}: design J={Jd:.4f} did not fall below the previous "
            f"{j_prev:.4f}. A PRESSURE margin must compress the design state as it "
            f"grows; if this fails the constant has been given ratio semantics again."
        )
        j_prev = Jd


def test_nera_is_the_only_deck_that_overrides_the_margin():
    """The override is a documented exception, not a habit.

    If a second deck needs it, that is a real signal — a confinement the shock bound
    cannot see — and it should be read, not absorbed. Failing here is the prompt to
    go read it.
    """
    overriders = []
    for name, path in ALL_DECKS:
        sc = config.load_scenario(path)
        if sc.solver.cfl_p_margin is not None:
            overriders.append(name)
    assert overriders == ["apfsds_vs_nera"], (
        f"decks overriding cfl_p_margin: {overriders}. Only apfsds_vs_nera should — "
        f"its binding particles are a kinematic vise, not a shock (PHYSICS §3.6.1). "
        f"A new one means a new deck has a confinement the global shock bound cannot "
        f"see. Read it before adding it to this list."
    )
