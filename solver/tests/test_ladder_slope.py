r"""Contract tests for milestone 20 — the 32-cell rung and the statistic it broke, RED first.

Same split as `test_route_difference.py`, for the same reason: what only the sizing
path can check, and what the instrument must refuse to print.

**The statistic.** §3.16 published an increment RATIO and was entitled to, because its
two rungs were equally spaced in `dx` (0.0625, 0.0625 mm) — over equal steps a
first-order error gives 1.00. This milestone's rung is dx=0.09375, a **0.03125 mm
step, half of the two before it**, where the same first-order error gives **0.50**. A
new ratio quoted against the old null would be one statistic under two conventions:
the exact defect §3.16 corrected in §3.15's headline, arriving one milestone later in
this repo's own table.

The repo's own guard fired on it. `test_route_difference.py`'s
`test_the_ladder_rungs_are_equally_spaced_in_dx` asserted the property and its
docstring named the failure: *"A fourth rung chosen for round cell counts rather than
for equal `dx` would silently break this."* It went red as designed, and the fix is
not to widen it — `_slope_table` changes the published column to a SLOPE
(increment/step, null 1.00 at any spacing), and the tests below pin that the slope is
genuinely step-agnostic while the raw ratio is not. Over equal steps the two are
identical, so §3.16's 0.35 / 0.10 / 0.65 are REPRODUCED, not superseded.

**The deck design** (`plan_substeps` + the deck schema). The claim is a RELATION
BETWEEN DECKS, never a literal — a hardcoded `dt` is satisfied by two decks copying
one wrong number, and goes red for reasons that are not about the experiment the
moment anything upstream of the CFL bound legitimately moves.

  * `heat_conv_dx094` / `heat_conv_dt456` — the family's SECOND exact `dt` pair. The
    `ceil` band that lands on 456 is measured by bisecting the real sizing path
    (§3.16's lesson: a 0.5 % nudge is not known to stay inside a band nobody measured).
  * The rung shares a 34 us window, which is new — M19's rung did not, and the tool
    withholds `depth_end` across unequal windows precisely because of that.

**The CFL margin.** The 32-cell arm is where §3.16 said the budget trend (63 -> 66 ->
73 -> 84 % of `c_max`) would go looking for `EOS_CFL_P_MARGIN`'s ceiling. It ships at
the global P=4 with NO per-deck override, and the test that pins the override
allowlist lives in `test_cfl_sizing.py`. What is pinned here is the ARGUMENT that
would license one if the audit ever demanded it: the global ~4.05 ceiling is
`era_filler`'s, and `era_filler` is not in this deck — this deck's own ceiling is far
above it. That is the whole reason `cfl_p_margin` is a deck field (§3.11).

MUTATIONS VERIFIED RED before any of this was trusted
([[instruments-that-cannot-see-the-failure]]) — harness at
`M:\claud_projects\temp\m20\red_check.py`, cited rather than shipped, as in §3.13,
§3.15 and §3.16.

Run: cd solver && pytest
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ballistics_solver import config, materials, mpm

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_DECKS = Path(__file__).resolve().parents[1] / "scenarios"


def _load(mod: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, _TOOLS / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


jg = _load("measure_jet_grid", "_measure_jet_grid_m20")

DX_ARM = "heat_conv_dx094"
DT_ARM = "heat_conv_dt456"
PREV_DX = "heat_conv_dx125"
SHIPPED = "heat_vs_composite"


def _deck(name: str):
    return config.load_scenario(_DECKS / f"{name}.yaml")


def _plan(name: str) -> dict:
    return mpm.plan_substeps(_deck(name))


# --- the statistic ----------------------------------------------------------


def test_the_slope_ratio_is_the_increment_ratio_when_the_steps_are_equal():
    """WHY THIS IS A CHANGE OF CONVENTION AND NOT A CHANGE OF ANSWER.

    Over equal steps the step cancels out of a ratio of slopes, so the new column
    equals the old one exactly. If it did not, §3.16's published 0.35 / 0.10 / 0.65
    would silently become three different numbers for the same three caches — two
    figures for one cache, which §3.16 itself calls a mismatch until someone says why.
    """
    vals = [-23.31, -36.00, -44.23]
    equal = [0.0625, 0.0625]
    tab = jg._slope_table(vals, equal)
    incs = tab["incs"]
    assert tab["ratios"][0] == pytest.approx(abs(incs[1] / incs[0]))
    assert tab["ratios"][0] == pytest.approx(0.65, abs=0.005)


def test_the_slope_is_step_agnostic_where_the_raw_increment_ratio_is_not():
    """THE WHOLE REASON THE STATISTIC CHANGED, as a measurement on a known answer.

    Build a ladder whose error is EXACTLY first order — e(h) = C*(h - h_shipped) — on
    the real rung spacing, so the right answer is known in advance: a first-order
    error is not settling at all, and any honest statistic must say so at every rung.

      * the SLOPE recovers C at every rung and its ratios are 1.00 throughout;
      * the RAW increment ratio reads 1.00 on the equal steps and **0.50** on the
        halved one — the same non-settling error reported as "decaying by half".

    0.50 is well under the 0.5 threshold the verdict block calls SETTLING, so the old
    statistic on the new rung would not merely be imprecise: it would flip the verdict
    on a sequence that has not moved.
    """
    dxs = [0.2500, 0.1875, 0.1250, 0.09375]
    steps = [dxs[k - 1] - dxs[k] for k in range(1, len(dxs))]
    C, h_ship = -37.0, 0.390625
    vals = [C * (h - h_ship) for h in dxs]

    # SIGN CONVENTION: `steps` are positive amounts of `dx` REMOVED while the values
    # move with decreasing h, so the printed slope is -C, i.e. pp per mm of
    # refinement. Constancy is the claim; the sign is bookkeeping, and asserting the
    # exact signed value is what keeps it from being quietly flipped.
    tab = jg._slope_table(vals, steps)
    for s in tab["slopes"]:
        assert s == pytest.approx(-C, rel=1e-12), (
            "a first-order error must give a CONSTANT slope at any spacing; "
            f"got {tab['slopes']}"
        )
    for r in tab["ratios"]:
        assert r == pytest.approx(1.0, rel=1e-12)

    # And the statistic it replaced, on the same numbers.
    incs = tab["incs"]
    raw = [abs(incs[k] / incs[k - 1]) for k in range(1, len(incs))]
    assert raw[0] == pytest.approx(1.0, rel=1e-12), "equal steps: the two agree"
    assert raw[1] == pytest.approx(0.5, rel=1e-12), (
        "the halved step makes the raw ratio read 0.50 for an error that is "
        "unchanged — this is the number the mode must not publish"
    )
    assert raw[1] < 0.5 or raw[1] == pytest.approx(0.5), "…and it lands on the verdict threshold"


def test_the_step_used_by_the_statistic_is_the_measured_dx_not_a_label():
    """A slope divided by the WRONG step is a ratio with a silent scale error.

    The mode derives `steps` from each arm's measured `dx`; feeding the same values
    with a mislabelled step must move the answer, or the division is decorative.
    """
    vals = [-23.31, -36.00, -44.23, -50.0]
    right = jg._slope_table(vals, [0.0625, 0.0625, 0.03125])
    wrong = jg._slope_table(vals, [0.0625, 0.0625, 0.0625])
    assert right["ratios"][1] != pytest.approx(wrong["ratios"][1]), (
        "the step is not reaching the ratio, so the column is an increment ratio "
        "wearing a slope's name"
    )
    assert right["ratios"][1] == pytest.approx(2.0 * wrong["ratios"][1])


def test_a_near_zero_denominator_is_withheld_rather_than_amplified():
    """§3.13's x=160 posture, carried into the new column.

    Two rungs landing on top of each other make the next ratio arbitrarily large, and
    printing it beside the real ones invites reading a divergence off an accident.
    """
    vals = [-4.23, -4.24, -5.29, -6.10]
    tab = jg._slope_table(vals, [0.0625, 0.0625, 0.03125])
    assert tab["ratios"][0] is None, "a 0.2 % increment is not a denominator"
    assert tab["ratios"][1] is not None, "…but the next one is a reading"


def test_the_withholding_rule_survived_being_rescaled_into_a_slope():
    """The guard tests the INCREMENT against the value it moved, NEVER the slope.

    A guard rewritten in terms of slopes changes meaning with the step: dividing by a
    small step inflates the slope, so the same physically-negligible increment stops
    being withheld simply because the rung below it was finer. Whether a rung moved is
    a property of the rung, not of the spacing used to report it.

    The fixture has to be able to SEE that: an increment of 1.2 % of its value is
    withheld either way at a 0.0625 mm step (slope 0.8), so the discriminating case is
    the same increment over a 0.03125 mm step, where a slope-based guard reads 1.6 and
    lets it through. This test was GREEN against that mutation on its first draft, for
    exactly the reason [[instruments-that-cannot-see-the-failure]] describes.
    """
    vals = [-4.23, -4.28, -5.29, -6.10]  # first increment is 1.2 % of the value
    fine = jg._slope_table(vals, [0.03125, 0.0625, 0.0625])
    coarse = jg._slope_table(vals, [0.2500, 0.0625, 0.0625])
    assert fine["ratios"][0] is None, (
        "a negligible increment stopped being withheld because the step shrank — the "
        "guard is reading a slope, not the increment"
    )
    assert coarse["ratios"][0] is None
    assert fine["slopes"][0] == pytest.approx(8.0 * coarse["slopes"][0]), (
        "…and the slopes really do differ by the step ratio, so the fixture is "
        "capable of telling the two guards apart"
    )


def test_a_withheld_ratio_never_produces_a_settling_verdict():
    """THE BUG THIS TEST EXISTS FOR, found on the real four-rung run.

    `_slope_table` returns `None` for a ratio it has suppressed as noise-amplified.
    The verdict block compared that against `>= 0.5` — via a NaN, which is False — and
    printed **SETTLING** for a quantity whose ratio the same function had declared
    unreadable two lines earlier. A verdict computed from evidence already ruled
    inadmissible is the strongest form of
    [[instruments-that-cannot-see-the-failure]], and it reached the real ladder output
    before it was caught.
    """
    vals = [31.58, 45.13, 43.77, 40.07]          # the real v_resid row
    tab = jg._slope_table(vals, [0.0625, 0.0625, 0.03125])
    assert tab["ratios"][-1] is None, "fixture must actually exercise the withheld path"
    verdict, _ = jg._settling_verdict(vals, tab["slopes"], tab["ratios"], [12, 16, 24, 32])
    assert "WITHHELD" in verdict
    assert "SETTLING" not in verdict.replace("NOT SETTLING", ""), (
        "a suppressed ratio was reported as a settling verdict"
    )


def test_a_turnover_is_only_called_one_once_the_reversal_repeats():
    """§3.16's own posture, enforced rather than restated: *"One reversal is one
    point."* It named the turnover and declined to claim it. The direction line claims
    it only when the post-reversal sign has held for a second increment."""
    order = [12, 16, 24, 32]
    once = [31.58, 45.13, 43.77]                  # §3.16's three rungs: one reversal
    tab = jg._slope_table(once, [0.0625, 0.0625])
    _, d = jg._settling_verdict(once, tab["slopes"], tab["ratios"], order)
    assert "one reversal is still one point" in d
    assert "TURNED OVER" not in d

    twice = [31.58, 45.13, 43.77, 40.07]          # §3.17's fourth rung: it repeats
    tab = jg._slope_table(twice, [0.0625, 0.0625, 0.03125])
    _, d = jg._settling_verdict(twice, tab["slopes"], tab["ratios"], order)
    assert "TURNED OVER" in d


def test_the_direction_line_survives_a_ratio_that_does_not():
    """The two questions fail independently, which is why they are two lines. The
    v_resid row has no readable ratio and a perfectly readable direction."""
    vals = [31.58, 45.13, 43.77, 40.07]
    tab = jg._slope_table(vals, [0.0625, 0.0625, 0.03125])
    verdict, direction = jg._settling_verdict(vals, tab["slopes"], tab["ratios"],
                                              [12, 16, 24, 32])
    assert "WITHHELD" in verdict and "TURNED OVER" in direction


def test_a_monotone_sequence_is_reported_as_monotone():
    vals = [-23.31, -36.00, -44.23, -48.23]       # the real `through` row
    tab = jg._slope_table(vals, [0.0625, 0.0625, 0.03125])
    verdict, direction = jg._settling_verdict(vals, tab["slopes"], tab["ratios"],
                                              [12, 16, 24, 32])
    assert "MONOTONE" in direction
    assert "STILL NOT SETTLING" in verdict, (
        f"slope ratio {tab['ratios'][-1]:.2f} against a null of 1.00 is not settling"
    )


def test_the_statistic_change_flips_this_milestones_headline_verdict():
    """WHY THE CONVENTION HAD TO CHANGE BEFORE THE BAKE, ON THE REAL NUMBERS.

    `through` reads increments −12.69, −8.24, −3.99 pp over steps 0.0625, 0.0625,
    0.03125 mm. The RAW increment ratio is 0.48 — under the 0.50 the verdict block
    calls settling — while the SLOPE ratio is 0.97 against a null of 1.00. Same four
    caches, opposite conclusions, and the statistic is the only thing that differs.

    So this is not a presentational tidy-up: publishing §3.16's column on §3.17's rung
    would have announced that mass-through had converged, of a quantity whose slope
    barely moved.
    """
    vals = [-23.31, -36.00, -44.23, -48.23]
    steps = [0.0625, 0.0625, 0.03125]
    tab = jg._slope_table(vals, steps)
    incs = tab["incs"]
    raw = abs(incs[-1] / incs[-2])
    slope_ratio = tab["ratios"][-1]
    assert raw < jg.SETTLING_THRESHOLD <= slope_ratio, (
        f"raw {raw:.2f} and slope {slope_ratio:.2f} no longer straddle the "
        f"{jg.SETTLING_THRESHOLD} threshold — the reconciliation block's "
        "'OPPOSITE VERDICTS' line is then claiming something untrue"
    )
    assert raw == pytest.approx(0.48, abs=0.01)
    assert slope_ratio == pytest.approx(0.97, abs=0.01)


# --- the deck design --------------------------------------------------------


def test_the_32_cell_rung_runs_a_bit_identical_substep():
    """What makes the rung a MEASURED dx-only effect rather than a difference of two
    opposing errors (§3.13's transferable rule)."""
    dx, dt = _plan(DX_ARM), _plan(DT_ARM)
    assert dx["dt_ms"] == dt["dt_ms"], (
        f"{DX_ARM} runs dt={dx['dt_ms']!r} against {DT_ARM}'s {dt['dt_ms']!r}; the "
        "rung is then dx AND the clock, which is the thing the partner exists to remove"
    )
    assert dx["substeps"] == dt["substeps"] == 456


def test_the_partner_is_pinned_by_its_deck_dt_and_the_pin_bites():
    """PHYSICS §3.11/§3.13: isolate with the deck `dt`, NEVER with `cfl_p_margin`.

    Unpinned, the partner sits at the shipped grid's own CFL bound — 110 substeps, a
    4x miss — so the pin is load-bearing rather than decorative. And the deck must be
    DECK-bound: a partner whose deck dt sat above the CFL bound would be silently
    CFL-sized and would track the grid instead of the experiment.
    """
    p = _plan(DT_ARM)
    assert p["bound_by"] == "deck"
    assert p["dt_deck_ms"] <= p["dt_cfl_ms"]

    d = _deck(DT_ARM)
    unpinned = mpm.plan_substeps(replace(d, solver=replace(d.solver, dt=5.0e-8)))
    assert unpinned["substeps"] == _plan(SHIPPED)["substeps"], (
        "without its deck dt this arm is just the shipped deck again"
    )
    assert unpinned["substeps"] != 456


def test_the_ceil_band_that_lands_on_456_is_narrow_and_the_deck_sits_inside_it():
    """`substeps = ceil(frame_dt/dt)`, so only a narrow window of deck `dt` hits 456.

    Bisected through the REAL sizing path rather than re-derived: re-deriving the
    arithmetic in a test is satisfied by copying a bug (the `test_cfl_sizing` lesson),
    and §3.16 found the band 0.29 % wide after assuming a 0.5 % nudge would stay in it.
    """
    d = _deck(DT_ARM)

    def subs(dt_s: float) -> int:
        return mpm.plan_substeps(replace(d, solver=replace(d.solver, dt=dt_s)))["substeps"]

    lo, hi = 1e-12, 1e-8
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if subs(m) > 456:
            lo = m
        else:
            hi = m
    low_edge = hi
    lo, hi = 1e-12, 1e-8
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if subs(m) >= 456:
            lo = m
        else:
            hi = m
    high_edge = lo

    assert low_edge < d.solver.dt < high_edge, (
        f"deck dt {d.solver.dt} is outside the band [{low_edge}, {high_edge}] that "
        "lands on 456 substeps; the exact pair is an inference again"
    )
    width = (high_edge - low_edge) / d.solver.dt
    assert width < 0.005, f"band is {width:.4%} wide — wider than §3.16 measured"


def test_the_rung_shares_a_frame_dt_and_this_time_a_window_too():
    """A window is a RECORDING LENGTH as long as `frame_dt` is untouched (§3.16), and
    `frame_dt` is the thing that must match. M19's rung ran unequal windows on purpose;
    this one does not, which is what lets `depth_end` be comparable within the rung —
    the tool's own guard is what decides that, from the caches, not this test."""
    for name in (DX_ARM, DT_ARM):
        s = _deck(name).solver
        assert s.total_time / s.frame_count == pytest.approx(2.0e-7), (
            f"{name} broke the repo's uniform frame_dt, so its window is a second "
            "variable rather than a recording length"
        )
    a, b = _deck(DX_ARM).solver, _deck(DT_ARM).solver
    assert a.total_time == b.total_time and a.frame_count == b.frame_count


def test_the_new_rung_halves_the_dx_step_which_is_why_the_statistic_changed():
    """The PREMISE of this whole milestone's statistic change, measured from the decks.

    If a later edit made the spacing equal again, the slope column would still be
    correct but the reconciliation prose would be describing a ladder that no longer
    exists.
    """
    dxs = [_plan(n)["dx"] for _, n, *_ in jg.LADDER]
    steps = [dxs[k - 1] - dxs[k] for k in range(1, len(dxs))]
    assert steps[0] == pytest.approx(steps[1]), "the 16- and 24-cell rungs were equal"
    assert steps[2] == pytest.approx(0.5 * steps[1], rel=1e-9), (
        f"the new rung's step is {steps[2]}, not half of {steps[1]}; the 0.50 null "
        "the mode reconciles against is then wrong"
    )


def test_the_seeded_lattice_gives_exactly_32_cells_across_the_jet():
    """§3.13 caught `diameter/dx` disagreeing with the seed once (15 rows = 7.5 cells
    where the domain implies 7.68), and §3.15 read cells off the real lattice for that
    reason. Here the SEEDING PATH is run rather than its arithmetic restated."""
    sc = _deck(DX_ARM)
    dx = _plan(DX_ARM)["dx"]
    n_side = int(round(sc.solver.particles_per_cell ** 0.5))
    seed = mpm._seed(sc, dx, dx / n_side)
    jet = seed["mat_id"] == materials.get(sc.projectile.material).material_id
    rows = np.unique(np.round(seed["pos"][jet, 1], 6))
    assert len(rows) / n_side == pytest.approx(32.0), (
        f"{len(rows)} seeded rows is {len(rows) / n_side} cells across, not 32"
    )
    assert float(np.diff(rows).min()) == pytest.approx(dx / n_side)


# --- the CFL margin: the argument that would license a per-deck override -----


def test_this_decks_own_cfl_ceiling_is_far_above_the_GLOBAL_one():
    """WHY `cfl_p_margin` IS A DECK FIELD (§3.11), computed rather than quoted.

    M14's hard ceiling near P=4.05 is `era_filler`'s: it designs to J=0.5504 against a
    pole-guard switch at 0.5500, and past the crossing the four ERA decks size from the
    guard's extrapolated backstop — M14's own defect. `era_filler` is not in this deck,
    and sweeping P on THIS deck's materials the first crossing is P≈11.5. So the remedy
    §3.16 named is real: the global constant never has to move.

    **FAR ABOVE THE GLOBAL CONSTANT IS NOT THE SAME AS COMFORTABLE.** The 32-cell bake
    breached at 1.60x and covering its measured `c_eff` needs **P≈10** — so the
    headroom against what this deck actually demands is **1.15x**, not 2.9x. That is
    the milestone's transferable finding (§3.17) and the reason a 48-cell rung is not
    reachable under this sizing scheme. This test pins the relation that still holds;
    it is not evidence of comfort.

    Derived from `materials.py` through the real sizing helpers; a literal here would
    go stale the way §3.8's table did.
    """
    sc = _deck(DX_ARM)
    names = sorted({sc.projectile.material, *(a.material for a in sc.armor)})
    assert "era_filler" not in names, "the global ceiling's material is not in this deck"

    proj = materials.get(sc.projectile.material)
    v = sc.projectile.velocity
    worst = max([mpm._impact_pressure(proj, materials.get(n), v) for n in names]
                + [mpm._impact_pressure(proj, proj, v)])

    def clears(P: float) -> bool:
        for n in names:
            m = materials.get(n)
            mu, lam = mpm._lame(m.youngs_modulus, m.poisson_ratio)
            mgp = mpm._mg_params(m)
            if mpm._eos_equilibrium_j(P * worst, lam + mu, mgp) < mgp["J_sw"]:
                return False
        return True

    assert clears(mpm.EOS_CFL_P_MARGIN), (
        "this deck already sizes past a pole switch at the shipped margin"
    )
    lo, hi = mpm.EOS_CFL_P_MARGIN, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if clears(mid):
            lo = mid
        else:
            hi = mid
    # Measured P=11.5 against a global constant of 4 and a global CEILING of ~4.05.
    # The threshold is deliberately loose: the claim is "far above", and pinning the
    # ceiling to two decimals would be pinning a material table, not an argument.
    assert lo > 2.5 * mpm.EOS_CFL_P_MARGIN, (
        f"this deck's own ceiling is P={lo:.2f}, not the multiple of the global "
        "constant the per-deck argument rests on"
    )
    assert lo > 2.0 * 4.05, (
        f"this deck's ceiling P={lo:.2f} is close to the GLOBAL one (~4.05), so a "
        "per-deck override would buy little and §3.11's remedy needs re-arguing"
    )


def test_the_32_cell_arm_ships_on_the_global_margin():
    """The budget number IS a deliverable (§3.17): 63 -> 66 -> 73 -> 84 % along this
    ladder, and a per-deck `cfl_p_margin` set a priori on the newest rung would have
    destroyed the measurement it was baked to take. If a future edit adds an override
    here, `test_cfl_sizing.py`'s allowlist goes red too — deliberately, so the act is
    argued rather than absorbed."""
    for name in (DX_ARM, DT_ARM):
        assert _deck(name).solver.cfl_p_margin is None


# --- the breach control -----------------------------------------------------


def test_the_control_differs_from_the_breached_arm_in_dt_alone():
    """A dt-only pair at fixed `dx` needs no partner — it IS the pair. If the two arms
    disagreed on the grid, the comparison would be the confounded thing this whole
    family exists to avoid."""
    arm, ctl = _deck(DX_ARM), _deck("heat_conv_dx094_dt770")
    assert arm.solver.grid_resolution == ctl.solver.grid_resolution
    assert arm.solver.total_time == ctl.solver.total_time
    assert arm.solver.frame_count == ctl.solver.frame_count
    a, c = _plan(DX_ARM), _plan("heat_conv_dx094_dt770")
    assert a["dx"] == c["dx"]
    assert c["substeps"] == 770 and a["substeps"] == 456
    assert c["bound_by"] == "deck", (
        "the control must be pinned by its DECK dt (§3.11), never by cfl_p_margin"
    )
    assert ctl.solver.cfl_p_margin is None


def test_the_control_runs_the_substep_a_margin_of_ten_would_have_produced():
    """§3.11 forbids isolating with `cfl_p_margin`, so the margin was used only as a
    HOST-SIDE CALCULATOR: P=10 clears the measured c_eff=102797, and the deck `dt`
    then lands on that same substep count. This pins the two paths agreeing, which is
    the only thing that makes the control's header claim true."""
    d = _deck(DX_ARM)
    via_margin = mpm.plan_substeps(replace(d, solver=replace(d.solver, cfl_p_margin=10.0)))
    assert via_margin["substeps"] == _plan("heat_conv_dx094_dt770")["substeps"]
    assert via_margin["c_max"] > 102797, (
        "P=10 no longer clears the c_eff the 32-cell bake actually reached, so the "
        "control is no longer sized to the thing it is named for"
    )


def test_every_reference_row_is_a_dt_only_pair_at_one_grid():
    """The reference is the family's own `dt` sensitivity, so each row must differ in
    the clock and nothing else — and must be at the SHIPPED grid, since a reference
    measured at the breached arm's own resolution would beg the question."""
    shipped = _plan(SHIPPED)["dx"]
    for coarse, fine, cs, fs in jg.BREACH_REFERENCE:
        a, b = _plan(coarse), _plan(fine)
        assert a["dx"] == pytest.approx(b["dx"]) == pytest.approx(shipped)
        assert a["substeps"] == cs and b["substeps"] == fs
        assert fs > cs, "a reference row must REFINE the clock"


def test_the_dt_pair_is_normalised_by_its_own_coarse_arm():
    """Different convention from `_rung_delta`, on purpose and worth pinning. A ladder
    rung divides by the SHIPPED arm to stay additive with §3.13's decomposition; a
    dt-only pair has no third arm in it and divides by its own baseline. Mixing the
    two would put a §3.13 convention on a number that is not part of that table."""
    coarse = {"residual_v_matched": 80.0, "frac_through_matched": 0.5}
    fine = {"residual_v_matched": 100.0, "frac_through_matched": 0.4}
    out = jg._dt_pair(coarse, fine, [])
    assert out[-2] == pytest.approx(25.0)   # (100-80)/80, NOT /100
    assert out[-1] == pytest.approx(-20.0)


def test_the_floors_are_the_ones_the_repo_measured():
    """Stated in advance and derived from measurements this repo actually took —
    §3.13's <= 0.0024 % on front-curve percentile readings, and the 0.11 % aggregate
    floor from repeat bakes. A threshold chosen after seeing the difference is not a
    threshold."""
    assert jg.FLOOR_CURVE == 0.0024
    assert jg.FLOOR_AGGREGATE == 0.11
    assert jg.FLOOR_CURVE < jg.FLOOR_AGGREGATE, (
        "an aggregate over a population cannot be noisier than a percentile of it"
    )


def test_the_reference_scaling_is_conservative():
    """The reference ratios are not the control's 1.69x, so they are scaled to it —
    LINEARLY, while §3.14 measured the dt term SATURATING. Linear scaling therefore
    OVERSTATES what the reference predicts, which makes an indictment of the breached
    arm harder to obtain, never easier. Pinned as a direction, not a value."""
    for _, _, cs, fs in jg.BREACH_REFERENCE:
        ratio = fs / cs
        scale = (770 / 456 - 1.0) / (ratio - 1.0)
        assert scale > 1.0, (
            f"reference ratio {ratio:.2f}x is wider than the control's 1.69x, so "
            "scaling would SHRINK it and the test would stop being conservative"
        )


# --- the instrument ---------------------------------------------------------


def test_the_ladder_refuses_a_missing_arm_and_names_it(tmp_path, capsys):
    """A rung is TWO bakes and the expensive one is hours. Reporting a ladder one rung
    short would publish a settling verdict computed from fewer points than its own
    prose claims — and the prose is what a reader quotes."""
    assert jg._dt_ladder(tmp_path) == 2
    out = capsys.readouterr().out
    assert DX_ARM in out and "missing cache" in out
