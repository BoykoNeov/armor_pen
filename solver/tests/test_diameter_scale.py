r"""Contract tests for milestone 18 — the diameter/scale decomposition, and RED first.

Same split as `test_standoff_dt.py`, for the same reason.

**The deck design** (`plan_substeps` + the deck schema, solver-side). Milestone 18's
third arm is a 6 mm jet at dx=0.75: the fat arm's diameter at the shipped arm's 8
cells. Three properties make it a measurement rather than a third confounded point,
and none of them can be checked from a cache (`docs/CACHE_FORMAT.md` records
`frame_dt` and nothing else about the clock or the grid):

  * all three arms run a BIT-IDENTICAL `dt`, so no read is `dt`-confounded — which
    is precisely what §3.14's two-route comparison could not say of itself, and why
    its residual needed a transferred correction;
  * the new arm is bound by its DECK `dt`. Left alone it is CFL-bound at twice the
    shipped substep, so the pin is load-bearing rather than decorative;
  * the scale move scales the DISCRETIZATION and NOT the problem. That asymmetry is
    the whole hypothesis under test: `cells = diameter/dx`, so §3.8's
    controlling-parameter claim is a claim of scale invariance, and it can only fail
    because the standoff, the plate and the damage threshold do not scale with the
    jet.

**The instrument** (`tools/measure_standoff.py`). Each new reading is pinned against
the defect that same assertion must catch:

  * `_dt_residual` re-derives §3.14's published split from the caches instead of
    quoting it, and keys `DT_ARMS` BY NAME. `_dt_decomposition` indexes the same list
    positionally, so a reordering there is a live hazard: it would silently re-key
    the published decomposition and stay green.
  * `route_difference` is per-fraction by construction, because the scale row varies
    3x across the matching window. A mean-only instrument reports one number and
    hides that it depends on where you stood (the milestone 16 x=160 lesson).
  * `cells` is a row count, so it CANNOT distinguish the two 8-cell arms. That is not
    a defect — it is why the scale row has to exist — and it is pinned so nobody
    "fixes" `cells` into carrying an absolute scale it was never meant to carry.

SEVEN MUTATIONS VERIFIED RED before any of this was trusted
([[instruments-that-cannot-see-the-failure]]) — harness at
`M:\claud_projects\temp\m18\red_check.py`, cited rather than shipped, as in §3.13:

  * `_dt_residual` re-keyed positionally off a shuffled `DT_ARMS`
  * `route_difference` collapsed to `[mean] * n`
  * the new deck's `dt:` pin deleted (`bound_by` silently reverts to "cfl")
  * the new deck's `diameter` back to 3 mm, and separately its grid back to 1440
  * the standoff scaled along with the discretization (which would make the scale
    row a similarity test, and it would have to read ~0)
  * `particles_per_cell` changed under the matched-particle claim

One harness lesson, and it is the same shape as the defects being hunted: the
`particles_per_cell` mutation first came back GREEN because a `replace(..., 1)` hit
the deck HEADER's prose quoting `particles_per_cell: 4`, not the YAML line. **A
mutation that does not land reads exactly like a test that does not care** — check
the mutant is really mutated before believing the test that survived it.

Run: cd solver && pytest
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ballistics_solver import config, mpm

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "measure_standoff.py"
_spec = importlib.util.spec_from_file_location("_measure_standoff_m18", _TOOL)
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)

_DECKS = Path(__file__).resolve().parents[1] / "scenarios"

SHIPPED = "standoff_{s}"
FAT = "standoff_conv_d6mm_{s}"
COARSE_FAT = "standoff_conv_d6mm_dx750_{s}"
SIDES = ("s00", "s90")


def _deck(name: str):
    return config.load_scenario(_DECKS / f"{name}.yaml")


def _plan(name: str) -> dict:
    return mpm.plan_substeps(_deck(name))


# --- the deck design --------------------------------------------------------


@pytest.mark.parametrize("s", SIDES)
def test_all_three_arms_run_a_bit_identical_substep(s):
    """No read in this milestone may be dt-confounded.

    Stated as equality between arms rather than against a literal: a hardcoded
    1.754386e-6 would be satisfied by three decks that all copied the same wrong
    number, and would go red for a reason that is not about the experiment the
    moment anything upstream of the CFL bound legitimately moves.
    """
    ship, fat, coarse = (_plan(n.format(s=s)) for n in (SHIPPED, FAT, COARSE_FAT))
    assert fat["dt_ms"] == ship["dt_ms"]
    assert coarse["dt_ms"] == ship["dt_ms"], (
        f"the coarse-fat arm runs dt={coarse['dt_ms']!r} against the shipped "
        f"{ship['dt_ms']!r}; every read in --diameter-decomposition becomes a "
        "two-variable comparison"
    )


@pytest.mark.parametrize("s", SIDES)
def test_the_new_arm_is_pinned_by_its_deck_dt(s):
    """PHYSICS §3.13/§3.14: isolate with the deck `dt`, NEVER with `cfl_p_margin`.

    Unpinned, dx=0.75 is CFL-bound at twice the shipped substep — so deleting the
    deck `dt` does not fail loudly, it silently turns the arm into a coarse-dt
    duplicate and the milestone measures dx AND dt again.
    """
    coarse = _plan(COARSE_FAT.format(s=s))
    assert coarse["bound_by"] == "deck"
    assert coarse["dt_cfl_ms"] > _plan(SHIPPED.format(s=s))["dt_ms"], (
        "the CFL bound here must be LOOSER than the shipped dt, or the pin is not "
        "doing anything and the arm would have landed on the shipped clock anyway"
    )


@pytest.mark.parametrize("s", SIDES)
def test_the_scale_move_scales_the_discretization_and_not_the_problem(s):
    """THE HYPOTHESIS UNDER TEST, as a relation between two decks.

    `cells = diameter/dx`, so a claim that cells controls the response is a claim
    that the response depends only on that RATIO. The shipped and coarse-fat arms
    are the same discretization scaled 2x — and the standoff, the plate and the
    domain are IDENTICAL between them. That asymmetry is the only reason the scale
    row can read anything but zero, so it is asserted, not assumed.
    """
    ship, coarse = _deck(SHIPPED.format(s=s)), _deck(COARSE_FAT.format(s=s))
    p_ship, p_coarse = _plan(SHIPPED.format(s=s)), _plan(COARSE_FAT.format(s=s))

    # scaled: both factors of the ratio, by the same 2x
    assert p_coarse["dx"] == pytest.approx(2.0 * p_ship["dx"])
    assert coarse.projectile.diameter == pytest.approx(2.0 * ship.projectile.diameter)

    # NOT scaled: everything the jet has to fly through and hit
    assert coarse.armor[0].standoff == ship.armor[0].standoff
    assert coarse.armor[0].thickness == ship.armor[0].thickness
    assert coarse.armor[0].material == ship.armor[0].material
    assert coarse.projectile.length == ship.projectile.length
    assert coarse.projectile.velocity == ship.projectile.velocity
    assert coarse.projectile.tail_velocity == ship.projectile.tail_velocity
    assert (coarse.domain.xmax, coarse.domain.ymax) == (ship.domain.xmax, ship.domain.ymax)


@pytest.mark.parametrize("s", SIDES)
def test_the_two_8_cell_arms_really_are_8_cells(s):
    """The premise of the comparison, from the deck side.

    Read as a RELATION (equal ratios) plus the absolute, so it cannot be satisfied
    by a deck that got both factors wrong in the same direction.
    """
    ship, coarse = _deck(SHIPPED.format(s=s)), _deck(COARSE_FAT.format(s=s))
    p_ship, p_coarse = _plan(SHIPPED.format(s=s)), _plan(COARSE_FAT.format(s=s))
    cells_ship = ship.projectile.diameter / p_ship["dx"]
    cells_coarse = coarse.projectile.diameter / p_coarse["dx"]
    assert cells_ship == pytest.approx(cells_coarse)
    assert cells_ship == pytest.approx(8.0)
    # ...and the FAT arm is the 16-cell route it is compared against
    fat = _deck(FAT.format(s=s))
    assert fat.projectile.diameter / _plan(FAT.format(s=s))["dx"] == pytest.approx(16.0)


@pytest.mark.parametrize("s", SIDES)
def test_particle_resolution_across_the_jet_is_matched_too(s):
    """Not only the cells. `particles_per_cell=4` puts 2 particles per cell per axis,
    so the 6 mm jet at dx=0.75 is seeded 16 particles across exactly as the shipped
    3 mm jet is at dx=0.375. Were it not, the scale row would carry a third variable
    and could not be read as a scale move at all.
    """
    for name in (SHIPPED, COARSE_FAT):
        d, p = _deck(name.format(s=s)), _plan(name.format(s=s))
        assert d.solver.particles_per_cell == 4
        assert d.projectile.diameter / (p["dx"] / 2.0) == pytest.approx(16.0)


# --- the instrument ---------------------------------------------------------


def test_the_residual_is_keyed_by_name_not_by_position(monkeypatch):
    """`_dt_decomposition` indexes DT_ARMS POSITIONALLY (DT_ARMS[0]..DT_ARMS[5]).

    `_dt_residual` re-derives the same published split, so if it inherited that
    convention a reordering would silently re-key §3.14's numbers and stay green.
    Reordering the list must change nothing.

    `_mean_ratio` is stubbed by cache name, so this needs no caches and pins the
    KEYING rather than the arithmetic.
    """
    values = {
        "standoff_s00": 1.2643,
        "standoff_conv_dt513_s00": 1.2407,
        "standoff_conv_dt684_s00": 1.2445,
        "standoff_conv_dx250_s00": 1.4573,
        "standoff_conv_dx188_s00": 1.4968,
        "standoff_conv_d6mm_s00": 1.5587,
    }
    monkeypatch.setattr(so, "_mean_ratio", lambda caches, c0, c9, stride: values[c0])

    before = so._dt_residual(Path("caches"))
    monkeypatch.setattr(so, "DT_ARMS", list(reversed(so.DT_ARMS)))
    after = so._dt_residual(Path("caches"))
    assert before == after

    gap, dt_term, residual = before
    # The published §3.14 split, reproduced from its own inputs. Not a re-run of the
    # formula: these are the arm means the caches actually carry.
    assert gap == pytest.approx(4.13, abs=0.01)
    assert dt_term == pytest.approx(-1.56, abs=0.01)
    assert residual == pytest.approx(2.50, abs=0.01)


def test_a_mean_hides_a_window_that_swings():
    """WHY `route_difference` IS PER FRACTION (the milestone 16 x=160 lesson).

    Two route comparisons with the SAME mean: one flat, one swinging 3x across the
    window — which is what milestone 18's scale row actually does. A mean-only
    instrument reports the same single number for both and gives no hint that one of
    them depends entirely on where in the window it was read.
    """
    base = [1.0, 1.0, 1.0, 1.0]
    flat = [0.93, 0.93, 0.93, 0.93]
    swinging = [0.8772, 0.9325, 0.9451, 0.9590]

    d_flat = so.route_difference(base, flat)
    d_swing = so.route_difference(base, swinging)

    mean = sum(d_flat) / len(d_flat)
    assert sum(d_swing) / len(d_swing) == pytest.approx(mean, abs=0.35)
    assert max(d_flat) - min(d_flat) == pytest.approx(0.0, abs=1e-9)
    assert max(d_swing) - min(d_swing) > 8.0, (
        "the swinging window must be visibly a range; if the instrument returned a "
        "mean these two comparisons would be indistinguishable"
    )
    # ...and every value is well above the floor, so the spread is not a noise story.
    assert all(abs(v) > so.DT_RESOLUTION_PCT for v in d_swing)


def test_route_difference_is_signed_and_relative_to_the_named_base():
    """Sign carries the claim (§3.14 reads +2.50 %, §3.18's scale row reads
    negative), so an accidental abs() or a swapped base inverts the conclusion."""
    assert so.route_difference([1.0], [1.1])[0] == pytest.approx(10.0)
    assert so.route_difference([1.1], [1.0])[0] == pytest.approx(-9.0909, abs=1e-3)


def test_cells_cannot_tell_the_two_8_cell_arms_apart(tmp_path):
    """NOT a defect — the reason the scale row exists.

    `cells` counts lattice rows, so a 16-row jet reads 8 cells whatever its absolute
    pitch. The two 8-cell arms are therefore identical to this instrument, which is
    exactly why §3.8's claim could survive unexamined for so long. Pinned so nobody
    "improves" `cells` into carrying an absolute scale it must not carry.
    """
    from test_standoff_dt import _synth  # the milestone 17 fixture, unchanged

    fine = _synth(tmp_path, "ship8", [0.0, 10.0, 20.0, 30.0], rows=16, pitch=0.1875,
                  diameter=3.0)
    coarse = _synth(tmp_path, "coarse8", [0.0, 10.0, 20.0, 30.0], rows=16, pitch=0.375,
                    diameter=6.0)
    assert so.measure(fine)["cells"] == so.measure(coarse)["cells"] == 8.0
