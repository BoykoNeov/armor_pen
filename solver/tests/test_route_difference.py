r"""Contract tests for milestone 19 — the third route point, the 24-cell rung, RED first.

Same split as `test_standoff_dt.py` and `test_diameter_scale.py`, for the same
reason: what a cache can check and what only the sizing path can.

**The deck design** (`plan_substeps` + the deck schema, solver-side). Two pairs, and
in both the claim is a RELATION BETWEEN DECKS rather than a literal — a hardcoded
`dt` would be satisfied by two decks that copied the same wrong number, and would go
red for a reason that is not about the experiment the moment anything upstream of the
CFL bound legitimately moves.

  * `standoff_conv_d4p5mm_dt513_*` — 4.5 mm at the SHIPPED dx=0.375, so 12 cells by
    the diameter route, pinned to `standoff_conv_dx250_*`'s substep so the pair is
    dt-free. That pin is load-bearing: unpinned this arm is CFL-bound at 342, and the
    12-cell route difference would then need the same transferred correction that
    §3.14's 16-cell one does — which is the thing this row exists to avoid.
  * `heat_conv_dx125` / `heat_conv_dt342` — the family's FIRST EXACT dt pair (342 vs
    342, where §3.13's 16-cell rung carries 228 vs 230). The partner runs a LONGER
    window on purpose, and the test asserts the thing that must match (`frame_dt`,
    the substep) beside the thing that must not (`total_time`), so the asymmetry is
    pinned as deliberate rather than surviving as an oversight.

**The instrument** (`tools/measure_standoff.py`, `tools/measure_jet_grid.py`). Each
reading is pinned against the defect that same assertion must catch:

  * every arm table is now NAME-KEYED. Milestone 18 documented `_dt_decomposition`
    indexing `DT_ARMS` positionally as a live hazard — one `sort()` from re-keying a
    published decomposition while staying green — and this milestone removes it
    rather than deferring it again. So the test is: shuffle every table, and every
    published number must be unchanged.
  * `--route-difference` computes dt-matching from the arms' own `dt_ms`, never from
    a hand-written flag. A row that is quietly mislabelled "matched" would present a
    dt-confounded comparison as a measurement.
  * `--dt-ladder` REFUSES to compare `depth_end` across arms with unequal recording
    windows. `window_us` is read from the cache, so the guard cannot be satisfied by
    a label.

MUTATIONS VERIFIED RED before any of this was trusted
([[instruments-that-cannot-see-the-failure]]) — harness at
`M:\claud_projects\temp\m19\red_check.py`, cited rather than shipped, as in §3.13
and §3.15.

Run: cd solver && pytest
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ballistics_solver import config, mpm

_TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load(mod: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, _TOOLS / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


so = _load("measure_standoff", "_measure_standoff_m19")
jg = _load("measure_jet_grid", "_measure_jet_grid_m19")

_DECKS = Path(__file__).resolve().parents[1] / "scenarios"

NEW_DIAM = "standoff_conv_d4p5mm_dt513_{s}"
DX250 = "standoff_conv_dx250_{s}"
SHIPPED = "standoff_{s}"
SIDES = ("s00", "s90")


def _deck(name: str):
    return config.load_scenario(_DECKS / f"{name}.yaml")


def _plan(name: str) -> dict:
    return mpm.plan_substeps(_deck(name))


# --- the deck design: the 12-cell route pair --------------------------------


@pytest.mark.parametrize("s", SIDES)
def test_the_12_cell_routes_run_a_bit_identical_substep(s):
    """The reason this row is a MEASUREMENT and §3.14's 16-cell one is an estimate.

    §3.14 had to transfer a dt term measured at 8 cells onto its 16-cell arm, i.e.
    assume away the dx x dt interaction §3.13 named as out of reach. This pair does
    not: both arms run the same clock, so their difference needs no correction.
    """
    new, dx250 = _plan(NEW_DIAM.format(s=s)), _plan(DX250.format(s=s))
    assert new["dt_ms"] == dx250["dt_ms"], (
        f"the 4.5 mm arm runs dt={new['dt_ms']!r} against dx250's "
        f"{dx250['dt_ms']!r}; the 12-cell route difference becomes dt-confounded "
        "and would need exactly the correction this row exists to avoid"
    )
    assert new["substeps"] == dx250["substeps"]


@pytest.mark.parametrize("s", SIDES)
def test_the_new_arm_is_pinned_by_its_deck_dt_and_the_pin_bites(s):
    """PHYSICS §3.13/§3.14: isolate with the deck `dt`, NEVER with `cfl_p_margin`.

    Deleting the pin does not fail loudly. At the shipped dx the CFL bound is far
    looser, so the arm would silently land on 342 substeps and the milestone would
    measure diameter AND clock again.
    """
    new = _plan(NEW_DIAM.format(s=s))
    assert new["bound_by"] == "deck"
    assert new["dt_cfl_ms"] > new["dt_ms"], (
        "the CFL bound must be LOOSER than the pinned dt, or the pin does nothing "
        "and the arm lands where it would have anyway"
    )


@pytest.mark.parametrize("s", SIDES)
def test_the_diameter_route_holds_the_shipped_dx(s):
    """"Fatten the jet AT THE SHIPPED dx" is what makes it §3.8's second route.

    If the grid moved too, the arm would be a third dx point rather than the other
    route to the same cell count, and the comparison would stop being a route
    comparison at all.
    """
    new, ship = _plan(NEW_DIAM.format(s=s)), _plan(SHIPPED.format(s=s))
    assert new["dx"] == ship["dx"]
    assert _deck(NEW_DIAM.format(s=s)).solver.grid_resolution == \
        _deck(SHIPPED.format(s=s)).solver.grid_resolution


@pytest.mark.parametrize("s", SIDES)
def test_both_12_cell_routes_seed_12_cells_and_the_same_particles_across(s):
    """Both halves of "12 cells by both routes", from the seeding arithmetic.

    `particles_per_cell=4` lays 2 particles per cell per axis, so 24 rows across in
    BOTH arms — matched particle resolution, which is what stops the diameter being
    a proxy for a third variable. §3.15 made the same match at 16 rows.
    """
    new, dx250 = _deck(NEW_DIAM.format(s=s)), _deck(DX250.format(s=s))
    pn, pd = _plan(NEW_DIAM.format(s=s)), _plan(DX250.format(s=s))
    assert new.solver.particles_per_cell == dx250.solver.particles_per_cell == 4
    assert new.projectile.diameter / pn["dx"] == pytest.approx(12.0)
    assert dx250.projectile.diameter / pd["dx"] == pytest.approx(12.0)
    assert new.projectile.diameter / (pn["dx"] / 2.0) == pytest.approx(24.0)
    assert dx250.projectile.diameter / (pd["dx"] / 2.0) == pytest.approx(24.0)


@pytest.mark.parametrize("s", SIDES)
def test_the_12_cell_route_pair_differs_in_exactly_two_coupled_factors(s):
    """`cells = diameter/dx`, so holding the ratio and moving one factor moves the
    other. Everything ELSE must be identical or the row is not a scale row.
    """
    new, dx250 = _deck(NEW_DIAM.format(s=s)), _deck(DX250.format(s=s))
    assert new.projectile.diameter != dx250.projectile.diameter
    assert new.solver.grid_resolution != dx250.solver.grid_resolution
    # the scale factor is set by the cell count, not chosen
    assert new.projectile.diameter / dx250.projectile.diameter == pytest.approx(1.5)

    assert new.domain == dx250.domain
    assert new.projectile.length == dx250.projectile.length
    assert new.projectile.material == dx250.projectile.material
    assert new.projectile.velocity == dx250.projectile.velocity
    assert new.projectile.tail_velocity == dx250.projectile.tail_velocity
    assert new.projectile.angle_deg == dx250.projectile.angle_deg
    assert new.solver.total_time == dx250.solver.total_time
    assert new.solver.frame_count == dx250.solver.frame_count
    assert [(a.material, a.thickness, a.standoff) for a in new.armor] == \
        [(a.material, a.thickness, a.standoff) for a in dx250.armor]


# --- the deck design: the 24-cell rung and its exact partner -----------------


def test_the_24_cell_rung_is_the_familys_first_exact_dt_pair():
    """342 vs 342, where §3.13's 16-cell rung is 228 vs 230.

    Asserted as equality between the two decks AND as an inequality on the older
    rung, so the claim "this one is exact and that one is not" is pinned rather than
    asserted in prose.
    """
    dx125, dt342 = _plan("heat_conv_dx125"), _plan("heat_conv_dt342")
    assert dx125["substeps"] == dt342["substeps"] == 342
    assert dx125["dt_ms"] == dt342["dt_ms"]

    dx188, dt_fine = _plan("heat_conv_dx188"), _plan("heat_conv_dt_fine")
    assert dx188["substeps"] != dt_fine["substeps"], (
        "the 16-cell rung is documented as a 0.9 % substep mismatch; if it became "
        "exact, §3.13's caveat and this milestone's 'first exact pair' both go stale"
    )


def test_the_dt_partner_holds_the_shipped_grid_and_the_dx_arm_moves_it():
    dt342, ship = _plan("heat_conv_dt342"), _plan("heat_vs_composite")
    assert dt342["dx"] == ship["dx"]
    assert _plan("heat_conv_dx125")["dx"] < ship["dx"]
    assert dt342["bound_by"] == "deck"
    assert _plan("heat_conv_dx125")["bound_by"] == "cfl", (
        "the dx arm must be CFL-bound, or refining the grid did not drag the clock "
        "and there was nothing to partner"
    )


def test_the_24_cell_rung_is_24_cells_by_the_seeding_lattice():
    dx125 = _deck("heat_conv_dx125")
    dx = _plan("heat_conv_dx125")["dx"]
    assert dx125.projectile.diameter / dx == pytest.approx(24.0)
    # 2 particles per cell per axis -> 48 rows across, twice the 16-cell arm's 32.
    assert dx125.projectile.diameter / (dx / 2.0) == pytest.approx(48.0)


def test_the_longer_window_moves_total_time_and_NOT_frame_dt():
    """THE ASYMMETRY, pinned as deliberate.

    `heat_conv_dt342` records for 34 us where its dx partner records for 30, because
    a finer dt breaks out LATER and the matched residual needs breakout + 4 us inside
    the window. That is legitimate only because `frame_dt` — the repo's uniform
    2.0e-7 s — is untouched, and because every quantity the tool reports is read off
    the front curve or off each arm's own breakout. If a future edit "tidied" this by
    changing `frame_count` alone, `frame_dt` would move and the arms would stop being
    comparable frame-for-frame while still looking like a pair.
    """
    dt342, dx125, ship = (_deck(n) for n in
                          ("heat_conv_dt342", "heat_conv_dx125", "heat_vs_composite"))

    def frame_dt(d):
        return d.solver.total_time / d.solver.frame_count

    assert frame_dt(dt342) == pytest.approx(2.0e-7)
    assert frame_dt(dt342) == pytest.approx(frame_dt(dx125)) == pytest.approx(frame_dt(ship))
    assert dt342.solver.total_time > dx125.solver.total_time
    assert dx125.solver.total_time == ship.solver.total_time, (
        "the dx arm must NOT be extended: it breaks out at 23.7 us with 2.10 us of "
        "margin, so the window buys it nothing and costs ~13 % more GPU on the "
        "family's most expensive bake (PHYSICS §3.16)"
    )
    assert dt342.solver.grid_resolution == ship.solver.grid_resolution


def test_the_ceil_window_is_narrow_and_the_pin_sits_inside_it():
    """§3.14: a deck dt landing one integer off turns a measured pair into an
    inference, and `ceil` makes the band that hits a given count small.

    The band edges are found by BISECTING the real sizing path, not by re-deriving
    `ceil(frame_dt / dt)` — re-deriving it would be satisfied by copying a bug, which
    is the mistake `test_cfl_sizing` exists to avoid. Measured, the band for 342 is
    ~0.3 % wide, so "5.856e-10 is mid-window" is a claim worth pinning: a value near
    an edge is one float of upstream drift away from 341 or 343, and the pair would
    silently stop being a pair.
    """
    from dataclasses import replace

    d = _deck("heat_conv_dt342")
    assert mpm.plan_substeps(d)["substeps"] == 342

    def substeps_at(dt):
        return mpm.plan_substeps(replace(d, solver=replace(d.solver, dt=dt)))["substeps"]

    def edge(lo, hi):
        """Bisect for the dt where the substep count stops being 342."""
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if substeps_at(mid) == 342:
                lo = mid
            else:
                hi = mid
        return lo

    dt = d.solver.dt
    hi_edge = edge(dt, dt * 1.05)      # larger dt -> fewer substeps
    lo_edge = edge(dt, dt * 0.95)      # smaller dt -> more substeps
    width = (hi_edge - lo_edge) / dt
    assert width < 0.005, f"band {width:.4%} is not narrow; §3.14's warning is moot"

    where = (dt - lo_edge) / (hi_edge - lo_edge)
    assert 0.2 < where < 0.8, (
        f"the deck dt sits at {where:.0%} of a {width:.3%}-wide band — that is an "
        "edge, not the mid-window value the deck header claims"
    )


# --- the instrument: name-keying, across every table ------------------------


def _stub_ratios(monkeypatch):
    """Stub `_mean_ratio` by cache name, so keying is pinned without any cache."""
    values = {
        "standoff_s00": 1.2643,
        "standoff_conv_dt513_s00": 1.2407,
        "standoff_conv_dt684_s00": 1.2445,
        "standoff_conv_dx250_s00": 1.4573,
        "standoff_conv_dx188_s00": 1.4968,
        "standoff_conv_d6mm_s00": 1.5587,
        "standoff_conv_d6mm_dx750_s00": 1.1740,
        "standoff_conv_d4p5mm_dt513_s00": 1.3674,
    }
    monkeypatch.setattr(so, "_mean_ratio", lambda caches, c0, c9, stride: values[c0])


def test_every_arm_table_survives_being_shuffled(monkeypatch):
    """THE HAZARD MILESTONE 18 NAMED AND THIS ONE REMOVED.

    Positional indexing into a published arm table is one `sort()` away from
    re-keying the numbers it publishes, silently and green. Reversing all three
    tables must change nothing that is derived from them.
    """
    _stub_ratios(monkeypatch)
    before = so._dt_residual(Path("caches"))
    monkeypatch.setattr(so, "DT_ARMS", list(reversed(so.DT_ARMS)))
    monkeypatch.setattr(so, "DIAM_ARMS", list(reversed(so.DIAM_ARMS)))
    monkeypatch.setattr(so, "ROUTE_ARMS", list(reversed(so.ROUTE_ARMS)))
    assert so._dt_residual(Path("caches")) == before

    gap, dt_term, residual = before
    assert gap == pytest.approx(4.13, abs=0.01)
    assert dt_term == pytest.approx(-1.56, abs=0.01)
    assert residual == pytest.approx(2.50, abs=0.01)


def test_arm_lookup_is_by_key_and_a_missing_key_is_loud(monkeypatch):
    """A silent fallback is worse than a crash here: it would substitute one arm's
    caches for another's and print a confident wrong decomposition."""
    _stub_ratios(monkeypatch)
    assert so._arm_ratio(Path("caches"), so.ROUTE_ARMS, "d4p5mm") == pytest.approx(1.3674)
    with pytest.raises(KeyError):
        so._arm_ratio(Path("caches"), so.ROUTE_ARMS, "no_such_arm")


def test_every_route_row_names_arms_that_exist():
    keys = {a.key for a in so.ROUTE_ARMS}
    for row in so.ROUTE_ROWS:
        assert row.fine in keys and row.fat in keys


def test_dt_matching_is_computed_from_the_arms_not_declared():
    """The 8- and 12-cell rows are dt-free and the 16-cell one is not, and that must
    be READ off the table rather than written beside it — a row mislabelled 'matched'
    would present a dt-confounded comparison as a measurement.
    """
    by_key = {a.key: a for a in so.ROUTE_ARMS}
    matched = {row.cells: by_key[row.fine].dt_ms == by_key[row.fat].dt_ms
               for row in so.ROUTE_ROWS}
    assert matched == {8: True, 12: True, 16: False}


def test_the_scale_factor_is_the_cell_ratio_not_a_free_choice():
    """`cells = diameter/dx`, so reaching N cells from the shipped 8 moves each
    factor by N/8. The 12-cell row is therefore a 1.5x row where the others are 2x —
    the confound §3.16 declares, encoded in the table rather than only in prose.
    """
    assert {row.cells: row.scale for row in so.ROUTE_ROWS} == {8: 2.0, 12: 1.5, 16: 2.0}
    by_key = {a.key: a for a in so.ROUTE_ARMS}
    for row in so.ROUTE_ROWS:
        if row.cells == 8:
            continue          # both 8-cell arms are scaled twins, not routes from it
        assert "0.3750" in by_key[row.fat].label, (
            "the diameter route must sit at the SHIPPED dx; if it moved, the row "
            "stops being §3.8's second route"
        )


# --- the instrument: the window guard ---------------------------------------


ATTRS = ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
         "internal_energy"]
COL = {a: i for i, a in enumerate(ATTRS)}


def _jet_cache(tmp: Path, name: str, front, *, frame_dt=2.0e-7, rows=16,
               pitch=0.1875) -> Path:
    """A synthetic composite-stack cache: one slab, a jet front advancing per frame.

    Hand-written rather than via `CacheWriter` for the usual reason (test_jet_grid,
    test_standoff_dt): these fixtures construct states the solver would never emit.
    """
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    nf = len(front)
    face, back = 150.0, 160.0
    sp = 0.5
    nx = int(round((back - face) / sp))
    ny = int(round(120.0 / sp))
    xs = face + (np.arange(nx) + 0.5) * (back - face) / nx
    ys = (np.arange(ny) + 0.5) * 120.0 / ny
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    armor = np.stack([gx.ravel(), gy.ravel()], axis=1)
    n_arm = armor.shape[0]

    per_row = 40
    n_jet = rows * per_row
    jy = 60.0 + pitch * np.repeat(np.arange(rows), per_row)

    fr = np.zeros((nf, n_arm + n_jet, len(ATTRS)), dtype=np.float64)
    for f in range(nf):
        fr[f, :n_arm, COL["pos_x"]] = armor[:, 0]
        fr[f, :n_arm, COL["pos_y"]] = armor[:, 1]
        fr[f, :n_arm, COL["material_id"]] = 1.0
        fr[f, n_arm:, COL["pos_x"]] = face + front[f]
        fr[f, n_arm:, COL["pos_y"]] = jy
        fr[f, n_arm:, COL["vel_mag"]] = 5000.0
        fr[f, n_arm:, COL["material_id"]] = 6.0
        k = int(round(n_jet * 0.5 * f / max(1, nf - 1)))
        fr[f, n_arm:n_arm + k, COL["damage"]] = 1.0

    (d / "manifest.json").write_text(json.dumps({
        "schema_version": 3,
        "scenario": name,
        "particle_count": fr.shape[1],
        "frame_count": nf,
        "attributes": ATTRS,
        "dtype": "float32",
        "frame_dt": frame_dt,
        "domain": {"xmin": 0, "xmax": 300, "ymin": 0, "ymax": 120},
        "units": "mm-ms-g",
        "materials": {"1": "rha", "6": "copper_jet"},
        "projectile": {"kind": "heat_jet", "material": "copper_jet",
                       "length": 120.0, "diameter": 3.0, "velocity": 7000,
                       "tail_velocity": 2000, "angle_deg": 0.0,
                       "nose_shape": "blunt"},
        "armor": [{"material": "rha", "thickness": 10.0, "standoff": 0.0}],
        "material_descriptions": {"1": "rha", "6": "jet"},
    }))
    fr.astype("<f4").tofile(d / "frames.bin")
    return d


def test_window_us_is_read_from_the_cache_not_assumed(tmp_path):
    short = _jet_cache(tmp_path, "short", np.linspace(0.0, 20.0, 31))
    long_ = _jet_cache(tmp_path, "long", np.linspace(0.0, 20.0, 51))
    a, b = jg.measure(short), jg.measure(long_)
    assert a["window_us"] == pytest.approx(30 * 2.0e-7 * 1e6)
    assert b["window_us"] == pytest.approx(50 * 2.0e-7 * 1e6)
    assert b["window_us"] > a["window_us"]


def test_depth_end_differs_between_windows_that_are_otherwise_identical(tmp_path):
    """WHY THE GUARD HAS TO EXIST, as a measurement rather than an argument.

    Two arms penetrating at the SAME rate, differing only in how long recording
    continued, report `depth_end` values that differ by tens of percent. A ladder
    that differenced them would publish the recording length as a grid effect. This
    is the same shape as `measure_standoff`'s depth-at-a-fixed-lab-time trap and the
    same shape as this tool's own reason to exist.
    """
    rate = np.arange(51) * 0.4
    short = _jet_cache(tmp_path, "s31", rate[:31])
    long_ = _jet_cache(tmp_path, "s51", rate)
    a, b = jg.measure(short), jg.measure(long_)
    # identical physics: the front is at the same place at every shared frame
    assert np.allclose(a["front"], b["front"][: len(a["front"])])
    assert b["depth_end"] > a["depth_end"] * 1.5
    # ...while a quantity read off the CURVE is unmoved by the extra recording.
    x = 155.0
    assert jg.arrival_us(a, x) == pytest.approx(jg.arrival_us(b, x))


def test_the_window_guard_covers_the_pairwise_path_not_just_the_ladder(tmp_path, capsys):
    """THE HAZARD IS TOOL-WIDE, so the guard must be too.

    `--dt-ladder` is not how anyone will first reach for `heat_conv_dt342`: §3.13
    assembled its own tables by invoking this tool PAIRWISE, so that is the path a
    reader will use, and it printed the `depth_end` spread with no window check at
    all. The guard therefore lives in `_table`, which every path goes through.
    """
    rate = np.arange(51) * 0.4
    a, b = jg.measure(_jet_cache(tmp_path, "w31", rate[:31])), \
        jg.measure(_jet_cache(tmp_path, "w51", rate))
    jg._table([a, b])
    out = capsys.readouterr().out
    assert "WITHHELD" in out
    assert f"{a['depth_end']:.1f} mm" not in out.split("THE TRAP")[1]

    # ...and the guard must NOT fire when the windows do match, or it would suppress
    # the trap section on every honest comparison in the repo.
    c = jg.measure(_jet_cache(tmp_path, "w31b", rate[:31] * 0.9))
    jg._table([a, c])
    assert "WITHHELD" not in capsys.readouterr().out


def test_the_ladder_names_arms_that_exist_and_interfaces_it_may_quote():
    """§3.13's sensitivity sweep: x=160 swings more than the effect there and x=190
    is marginal. If a future edit put either back into the quoted set, the ladder
    would start publishing a cell its own source says not to."""
    assert set(jg.LADDER_INTERFACES) == {215.0, 235.0}
    for cells, dx_cache, dt_cache, dx_sub, dt_sub in jg.LADDER:
        assert dx_cache.startswith("heat_conv_dx")
        assert dt_cache.startswith("heat_conv_dt")
    assert [row[0] for row in jg.LADDER] == [12.0, 16.0, 24.0, 32.0]


def test_the_rung_delta_is_normalised_by_the_baseline_not_the_partner(tmp_path):
    """§3.13 published every decomposed row against the SHIPPED arm, which is what
    makes dt-only + dx-only comparable to the joint row. Normalising a rung by its own
    partner gives a different number for the same measurement (+34.1 % where §3.13
    published +31.6 %), and two figures for one cache is a mismatch until someone says
    why. Pinned on synthetic arms whose numbers are chosen so the two conventions
    cannot coincide.
    """
    base = {"residual_v_matched": 100.0, "frac_through_matched": 1.0,
            "front": None, "t_us": None}
    dt_arm = {"residual_v_matched": 80.0, "frac_through_matched": 0.8}
    dx_arm = {"residual_v_matched": 120.0, "frac_through_matched": 0.5}
    out = jg._rung_delta(dx_arm, dt_arm, base, [])
    # (120 - 80) / 100 = +40 %, NOT (120 - 80) / 80 = +50 %
    assert out[-2] == pytest.approx(40.0)
    assert out[-1] == pytest.approx(-30.0)


def test_the_ladder_rungs_are_ordered_and_the_spacing_claim_is_computed():
    """WHAT MADE THE INCREMENT RATIO READABLE, AND WHAT HAPPENED TO IT.

    This test used to assert the rungs were EQUALLY SPACED in `dx` (12/16/24 cells is
    0.2500 / 0.1875 / 0.1250 — two steps of 0.0625 mm), because over equal steps a
    first-order error gives an increment ratio of 1.00, so a ratio well under 1 means
    something. Its own docstring warned: "A fourth rung chosen for round cell counts
    rather than for equal `dx` would silently break this."

    **That is exactly what milestone 20's 32-cell rung does** — dx=0.09375 is a
    0.03125 mm step, half of the two before it — so the guard fired as designed. The
    resolution is NOT to widen it: §3.17 changes the published statistic to a SLOPE
    (increment/step), whose null is 1.00 at any spacing, and `test_ladder_slope.py`
    pins that it is genuinely step-agnostic. What survives here is the half that is
    still a property of the LADDER rather than of the statistic: the rungs must run
    coarse to fine, and the mode must never CLAIM equal spacing it does not have.
    """
    dxs = [_plan(dx_cache)["dx"] for _, dx_cache, *_ in jg.LADDER]
    assert dxs == sorted(dxs, reverse=True), "rungs must be ordered coarse -> fine"

    steps = [dxs[k - 1] - dxs[k] for k in range(1, len(dxs))]
    equal = max(steps) - min(steps) < 1e-9
    assert not equal, (
        "the rungs are equally spaced again; that is not a failure, but the mode's "
        "unequal-spacing prose and the reconciliation block are then dead code and "
        "this test is no longer checking anything — re-read §3.17 before deleting it"
    )
    # The prose is COMPUTED from the steps, so it cannot go stale the way §3.8's table
    # did. A mode that types "EQUALLY SPACED" over these rungs is asserting a null of
    # 1.00 for a ratio whose real null is 0.50.
    src = (Path(jg.__file__).read_text(encoding="utf-8"))
    assert 'equal else " — NOT EQUAL."' in src, (
        "the equal-spacing sentence is no longer derived from the measured steps"
    )


def test_the_ladder_baseline_is_the_shipped_deck():
    assert jg.LADDER_BASELINE == "heat_vs_composite"
    assert all(jg.LADDER_BASELINE not in (dx, dt) for _, dx, dt, *_ in jg.LADDER), (
        "the baseline must not also be a rung, or a rung would be differenced "
        "against itself"
    )


def test_the_ladder_rungs_pair_a_dx_arm_with_its_own_substep_count():
    """The labels in LADDER are not data, so they are checked against the decks they
    name. A rung whose two arms do not share a substep is not a matched-dt row, and
    the mode's entire claim is that each row is `dx` alone."""
    for cells, dx_cache, dt_cache, dx_sub, dt_sub in jg.LADDER:
        assert mpm.plan_substeps(_deck(dx_cache))["substeps"] == dx_sub
        assert mpm.plan_substeps(_deck(dt_cache))["substeps"] == dt_sub
        assert abs(dx_sub - dt_sub) / dx_sub < 0.01, (
            f"rung {cells}: {dx_sub} vs {dt_sub} is more than a 1 % clock mismatch, "
            "so the row is not a dx-only measurement"
        )
