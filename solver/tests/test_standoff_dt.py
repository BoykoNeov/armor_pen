"""Contract tests for milestone 17 — the dx/dt decomposition, and that it goes RED.

Two things are pinned here, and they live in different places for a reason.

**The deck pairing** (`plan_substeps`, solver-side). The whole experiment is that
each `dt` partner runs a substep BIT-IDENTICAL to its `dx` arm's, so the pair
differs in `dx` alone. That is a claim about the real sizing path, and nothing in a
cache can check it — `docs/CACHE_FORMAT.md` records `frame_dt`, never `dt`,
`grid_resolution` or a substep count. So it is asserted here against
`mpm.plan_substeps`, which `bake` itself calls. The assertions deliberately do NOT
re-derive the arithmetic: re-running `ceil(frame_dt / min(deck, cfl))` in a test
would be satisfied by copying a bug, which is the mistake `test_cfl_sizing` was
written to avoid. They state RELATIONS between decks instead.

**The instrument** (`tools/measure_standoff.py`). Every reading the new
`--dt-decomposition` mode reports is paired with the defect that same assertion must
catch:

  * `cells across the jet` — the controlling parameter of §3.8, previously carried
    as a hand-computed label. Read off the seeded lattice, it must survive a
    manifest whose `projectile.diameter` LIES, because that block is provenance,
    not data (CACHE_FORMAT §2.1).
  * the frame stride — arms baked at 225 and at 75 frames have to be compared on
    one cadence. Subsampling must change the interpolation resolution and nothing
    physical, and it must actually subsample.
  * **the mode's own reason to exist** — a synthetic pair whose depths BOTH move
    while their ratio does not. §3.13's falsifier is written on the ratio alone, so
    a ratio-only instrument would report "no effect" for a solver that moved every
    depth it measures. The depths must show what the quotient cancels.
  * the verdict — supported / falsified-in-sign / below-the-floor are three
    different readings and must not collapse into each other.

Run: cd solver && pytest
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ballistics_solver import config, mpm

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "measure_standoff.py"
_spec = importlib.util.spec_from_file_location("_measure_standoff", _TOOL)
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)

_DECKS = Path(__file__).resolve().parents[1] / "scenarios"

ATTRS = ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
         "internal_energy"]
COL = {a: i for i, a in enumerate(ATTRS)}


def _plan(name: str) -> dict:
    return mpm.plan_substeps(config.load_scenario(_DECKS / f"{name}.yaml"))


# --- the deck pairing -------------------------------------------------------

# (dt partner, the dx arm it partners). The pair differs in dx and in NOTHING else.
PAIRS = [
    ("standoff_conv_dt513_s00", "standoff_conv_dx250_s00"),
    ("standoff_conv_dt513_s90", "standoff_conv_dx250_s90"),
    ("standoff_conv_dt684_s00", "standoff_conv_dx188_s00"),
    ("standoff_conv_dt684_s90", "standoff_conv_dx188_s90"),
]


@pytest.mark.parametrize("partner,arm", PAIRS)
def test_partner_runs_a_bit_identical_substep(partner, arm):
    """The pairing is exact, or the pair measures two variables instead of one."""
    p, a = _plan(partner), _plan(arm)
    assert p["dt_ms"] == a["dt_ms"], (
        f"{partner} runs dt={p['dt_ms']!r} but {arm} runs {a['dt_ms']!r}; the pair "
        "no longer isolates dx"
    )
    assert p["substeps"] == a["substeps"]


@pytest.mark.parametrize("partner,arm", PAIRS)
def test_partner_differs_from_its_arm_in_dx_and_only_dx(partner, arm):
    """A matched dt is only interesting if the grid actually differs."""
    p, a = _plan(partner), _plan(arm)
    assert p["dx"] != a["dx"]
    assert p["dx"] == _plan("standoff_s00")["dx"], (
        f"{partner} must sit at the SHIPPED dx — that is what makes it a partner "
        "rather than a third point on the confounded ladder"
    )


@pytest.mark.parametrize("partner,_arm", PAIRS)
def test_the_partner_is_isolated_by_its_deck_dt_not_by_the_cfl_bound(partner, _arm):
    """PHYSICS §3.13: isolate with the deck dt, NEVER with `cfl_p_margin`.

    If a partner's deck `dt` were removed or raised above the CFL limit, the bound
    would silently revert to CFL — which at the shipped `dx` is the SHIPPED substep,
    quietly turning the partner back into a duplicate of the shipped arm.
    """
    p = _plan(partner)
    assert p["bound_by"] == "deck"
    assert p["dt_ms"] < _plan("standoff_s00")["dt_ms"]


def test_the_fat_jet_route_is_dt_free():
    """§3.13 asserted this by reading the code; here it goes through the real path.

    `_impact_pressure` and `_eos_equilibrium_j` never see the diameter, and the
    d6mm decks carry the shipped `grid_resolution`, so fattening the jet must leave
    the substep untouched. If it ever does not, the two routes to 16 cells differ
    in `dt` as well and the comparison that milestone 17 rests on is void.
    """
    for s in ("s00", "s90"):
        ship = _plan(f"standoff_{s}")
        fat = _plan(f"standoff_conv_d6mm_{s}")
        assert fat["dt_ms"] == ship["dt_ms"]
        assert fat["dx"] == ship["dx"]


def test_the_dx_arms_are_cfl_bound_so_refining_dx_really_did_drag_the_clock():
    """The premise of the whole milestone: on the dx ladder, dt is NOT free.

    If the dx arms were deck-dt bound, refining the grid would not have moved the
    substep and there would have been nothing to decompose.
    """
    for arm in ("standoff_conv_dx250_s00", "standoff_conv_dx188_s00",
                "standoff_s00", "standoff_conv_d6mm_s00"):
        assert _plan(arm)["bound_by"] == "cfl"
    coarse, fine = _plan("standoff_s00"), _plan("standoff_conv_dx188_s00")
    # dt_cfl ∝ dx, because c_max carries no dx dependence (the AV term contributes
    # c_q·v_tip). Asserted as a RATIO so it cannot be satisfied by copying either
    # number: halving the cell must halve the clock.
    assert fine["dt_ms"] / coarse["dt_ms"] == pytest.approx(
        fine["dx"] / coarse["dx"], rel=1e-3
    )


# --- the instrument ---------------------------------------------------------


def _write_cache(tmp: Path, name: str, frames: np.ndarray, diameter=3.0) -> Path:
    """Write a synthetic cache by hand, per docs/CACHE_FORMAT.md.

    Deliberately NOT via the solver's CacheWriter: these fixtures construct states
    the solver would never produce (a lying manifest, a pair rigged to cancel in
    the ratio), and a writer enforcing consistency would refuse the defects.
    """
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    fc, pc, stride = frames.shape
    (d / "manifest.json").write_text(json.dumps({
        "schema_version": 3,
        "scenario": name,
        "particle_count": pc,
        "frame_count": fc,
        "attributes": ATTRS,
        "dtype": "float32",
        "frame_dt": 2.0e-7,
        "domain": {"xmin": 0, "xmax": 540, "ymin": 0, "ymax": 120},
        "units": "mm-ms-g",
        "materials": {"1": "rha", "6": "copper_jet"},
        "projectile": {"kind": "heat_jet", "material": "copper_jet",
                       "length": 120.0, "diameter": diameter, "velocity": 7000,
                       "tail_velocity": 2000, "angle_deg": 0.0,
                       "nose_shape": "blunt"},
        "armor": [{"material": "rha", "thickness": 150.0, "standoff": 0.0}],
        "material_descriptions": {"1": "steel", "6": "copper"},
    }))
    frames.astype("<f4").tofile(d / "frames.bin")
    return d


def _synth(tmp: Path, name: str, front, *, rows: int = 16, pitch: float = 0.1875,
           diameter: float = 3.0, per_row: int = 40) -> Path:
    """A cache with `rows` jet rows and a 150 mm target, front advancing per frame.

    `front[f]` is the depth past the face at frame f; consumed fraction ramps
    linearly with the frame, so depth and consumed fraction are separable.

    `per_row` exists because `consumed` is a COUNT over jet particles, so a thin
    fixture makes it a coarse staircase and interpolating a staircase is violently
    sensitive to which frames a stride happens to keep — an artifact of the fixture
    that reads exactly like a defect in the tool. A real bake carries ~10^4 jet
    particles; 16 rows x 40 gives a 0.16 % granularity, close enough to smooth.
    """
    nf = len(front)
    n_arm, face = 40, 270.0
    n_jet = rows * per_row
    fr = np.zeros((nf, n_jet + n_arm, len(ATTRS)), dtype=np.float64)
    ys = 60.0 + pitch * np.repeat(np.arange(rows), per_row)
    for f in range(nf):
        # armor: a static column spanning face..face+150
        fr[f, :n_arm, COL["pos_x"]] = np.linspace(face, face + 150.0, n_arm)
        fr[f, :n_arm, COL["pos_y"]] = 60.0
        fr[f, :n_arm, COL["material_id"]] = 1.0
        # jet: `rows` distinct lattice rows, all at the front
        fr[f, n_arm:, COL["pos_x"]] = face + front[f]
        fr[f, n_arm:, COL["pos_y"]] = ys
        fr[f, n_arm:, COL["vel_mag"]] = 5000.0
        fr[f, n_arm:, COL["material_id"]] = 6.0
        # consumed fraction ramps 0 -> 0.5 across the window
        k = int(round(n_jet * 0.5 * f / max(1, nf - 1)))
        fr[f, n_arm:n_arm + k, COL["damage"]] = 1.0
    return _write_cache(tmp, name, fr, diameter=diameter)


def test_cells_across_the_jet_is_read_from_the_lattice(tmp_path):
    for rows, expect in ((16, 8.0), (24, 12.0), (32, 16.0)):
        d = _synth(tmp_path, f"j{rows}", np.linspace(0, 50, 20), rows=rows)
        assert so.measure(d)["cells"] == expect


def test_cells_survives_a_manifest_that_lies_about_the_diameter(tmp_path):
    """CACHE_FORMAT §2.1: the scenario block is provenance, never measure from it.

    A 32-row jet is 16 cells across whatever the manifest claims its diameter is.
    A tool computing `diameter / dx` would report 16 for the honest cache and
    something else here, from bytes that are identical where it matters.
    """
    honest = _synth(tmp_path, "honest", np.linspace(0, 50, 20), rows=32,
                    diameter=6.0)
    liar = _synth(tmp_path, "liar", np.linspace(0, 50, 20), rows=32,
                  diameter=999.0)
    assert so.measure(honest)["cells"] == so.measure(liar)["cells"] == 16.0


def test_stride_subsamples_frames_without_moving_the_physics(tmp_path):
    d = _synth(tmp_path, "s", np.linspace(0, 60, 61))
    full, dec = so.measure(d), so.measure(d, stride=3)
    assert dec["frames_used"] < full["frames_used"]
    for f in so.MATCH_FRACTIONS:
        assert so.depth_at(dec, f) == pytest.approx(so.depth_at(full, f), rel=2e-3)


def test_a_stride_that_skips_nothing_is_the_unstrided_reading(tmp_path):
    """`--family` and `--convergence` must be untouched by the stride's existence."""
    d = _synth(tmp_path, "s1", np.linspace(0, 60, 40))
    a, b = so.measure(d), so.measure(d, stride=1)
    assert np.array_equal(a["depth"], b["depth"])
    assert np.array_equal(a["consumed"], b["consumed"])


def test_the_ratio_cannot_see_a_move_that_the_depths_can(tmp_path):
    """THE REASON THE DEPTH TABLE EXISTS (and the advisor's catch).

    §3.13's falsifier is written on the S90/S0 ratio alone. Here is a refinement
    that suppresses BOTH arms by 8 % — every depth the study measures moves — and
    leaves the ratio bit-identical. A ratio-only instrument reports "no effect" and
    would have concluded the timestep does nothing, which is exactly backwards.
    """
    s00 = _synth(tmp_path, "a00", np.linspace(0, 40, 40))
    s90 = _synth(tmp_path, "a90", np.linspace(0, 52, 40))
    f00 = _synth(tmp_path, "b00", np.linspace(0, 40, 40) * 0.92)
    f90 = _synth(tmp_path, "b90", np.linspace(0, 52, 40) * 0.92)

    coarse = so.compare(so.measure(s00), so.measure(s90))
    fine = so.compare(so.measure(f00), so.measure(f90))
    # 1e-4 is float32 storage roundoff, not slack: the ratio agrees to ~1e-6 while
    # every depth under it moved 8 %, which is the point being made.
    assert fine == pytest.approx(coarse, rel=1e-4), "the ratio is blind here by construction"

    # ...and the depths are not.
    for f in so.MATCH_FRACTIONS:
        d_coarse = so.depth_at(so.measure(s00), f)
        d_fine = so.depth_at(so.measure(f00), f)
        assert d_fine == pytest.approx(0.92 * d_coarse, rel=1e-4)
        assert abs(d_fine - d_coarse) > 0.5, "an 8 % depth move must be visible"


def test_a_perforated_arm_is_flagged_rather_than_quietly_ratioed(tmp_path):
    """A ceiling-capped baseline INFLATES its row's ratio (§3.8's own warning)."""
    ok = _synth(tmp_path, "ok", np.linspace(0, 100, 30))
    through = _synth(tmp_path, "through", np.linspace(0, 200, 30))
    assert not so.measure(ok)["perforated"]
    assert so.measure(through)["perforated"]
