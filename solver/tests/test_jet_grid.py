"""Contract tests for tools/measure_jet_grid.py — that it goes RED.

The tool exists because the OBVIOUS metric on this deck is blind (PHYSICS §3.13):
`heat_vs_composite` perforates, so depth at the end of the window stops measuring
penetration and starts measuring a free residual's flight. An instrument built to
dodge one blind spot is worth nothing if it has its own, so every reading it
reports is pinned here against a defect the same assertion must catch:

  * interface detection — a stack whose layers are separated by a VOID and a stack
    whose layers are merely a MATERIAL CHANGE with no gap. The deck has one of
    each, and a gap-only rule finds 4 of the 5 interfaces while looking perfectly
    healthy. Fed a bonded ceramic/steel contact, a gap-only tool must miss it.
  * the manifest is not consulted — a cache whose `armor` provenance block LIES
    about the geometry must still measure the true interfaces, because that block
    is provenance, not data (CACHE_FORMAT §2.1).
  * arrival time — a front that reaches an interface later must READ later, and a
    front that never reaches it must read NOT REACHED rather than the window end.
    Those two are the same assertion for a tool that clamps, and they are the
    difference between "arrived late" and "did not arrive".
  * the ceiling itself — the reason the tool exists. A synthetic pair that
    penetrates at visibly different rates but ends the window at the same place
    must show a DIFFERENCE in arrival time and NO difference in depth_end. If
    depth_end could see it, the tool would be unnecessary.
  * the lattice reading — 15 rows across a 3 mm jet is 7.5 cells, not the 7.68 that
    `domain/grid_resolution` implies, because `_fill_rect` rounds each object's
    lattice to fit it. The tool must report the seeded resolution.

`measure_jet_grid.py` lives in tools/ and imports neither half of the repo
(CLAUDE.md §3); loading it by path here keeps that true — the tool does not learn
about the solver, the solver's test suite just reads it.

Run: cd solver && pytest
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "measure_jet_grid.py"
_spec = importlib.util.spec_from_file_location("_measure_jet_grid", _TOOL)
jg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jg)

ATTRS = ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
         "internal_energy"]
COL = {a: i for i, a in enumerate(ATTRS)}

FRAME_DT = 2.0e-7  # s — the repo's uniform cadence


def _write_cache(
    tmp: Path,
    name: str,
    frames: np.ndarray,
    armor_block=None,
) -> Path:
    """Write a synthetic cache by hand, per docs/CACHE_FORMAT.md.

    Deliberately does NOT go through the solver's CacheWriter: the point of these
    fixtures is to construct states the solver would never produce (a front that
    stalls, a lying manifest), and a writer that enforced consistency would refuse
    exactly the defects being tested for.
    """
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    fc, pc, stride = frames.shape
    manifest = {
        "schema_version": 3,
        "scenario": name,
        "particle_count": pc,
        "frame_count": fc,
        "attributes": ATTRS,
        "dtype": "float32",
        "frame_dt": FRAME_DT,
        "domain": {"xmin": 0, "xmax": 300, "ymin": 0, "ymax": 120},
        "units": "mm-ms-g",
        "materials": {"1": "rha", "2": "ceramic", "6": "copper_jet"},
        "projectile": {"kind": "heat_jet", "material": "copper_jet",
                       "length": 120.0, "diameter": 3.0, "velocity": 7000,
                       "tail_velocity": 2000, "angle_deg": 0.0,
                       "nose_shape": "blunt"},
        "armor": armor_block if armor_block is not None else [
            {"material": "rha", "thickness": 10.0, "standoff": 0.0}],
        "material_descriptions": {"1": "rha", "2": "ceramic", "6": "jet"},
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    frames.astype("<f4").tofile(d / "frames.bin")
    return d


def _slab(x0: float, x1: float, mat: float, pitch: float) -> np.ndarray:
    """A static armor slab spanning the full domain height, on a lattice."""
    nx = max(1, int(round((x1 - x0) / pitch)))
    ny = max(1, int(round(120.0 / pitch)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = (np.arange(ny) + 0.5) * 120.0 / ny
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    p = np.zeros((gx.size, len(ATTRS)), dtype=np.float64)
    p[:, COL["pos_x"]] = gx.ravel()
    p[:, COL["pos_y"]] = gy.ravel()
    p[:, COL["material_id"]] = mat
    return p


def _jet(rows: int, pitch: float, x_tip: float, speed: float,
         cols: int = 40) -> np.ndarray:
    """A moving jet column: `rows` particle rows across, `cols` long, centred y=60."""
    ys = 60.0 + (np.arange(rows) - (rows - 1) / 2.0) * pitch
    xs = x_tip - np.arange(cols) * pitch
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    p = np.zeros((gx.size, len(ATTRS)), dtype=np.float64)
    p[:, COL["pos_x"]] = gx.ravel()
    p[:, COL["pos_y"]] = gy.ravel()
    p[:, COL["vel_mag"]] = speed
    p[:, COL["material_id"]] = 6.0
    return p


def _build(tmp: Path, name: str, front_x, *, rows: int = 15, pitch: float = 0.2,
           slab_pitch: float = 1.0, cols: int = 40,
           slabs=((150.0, 160.0, 1.0), (190.0, 215.0, 2.0), (215.0, 235.0, 1.0)),
           armor_block=None) -> Path:
    """A cache whose jet FRONT follows `front_x` frame by frame.

    The armor is static and identical every frame; only the jet translates. That
    is enough for every quantity this tool reads, and it keeps the fixture a
    statement about the INSTRUMENT rather than about any physics.
    The armor is laid on a COARSE lattice (`slab_pitch`) independent of the jet's.
    Nothing here depends on armor resolution — the tool's interface rules are all
    expressed in multiples of the lattice pitch it measures for itself — and a
    real-deck pitch would put ~165 000 static particles into every one of 150
    frames, making these fixtures cost more than the bakes they are about.
    """
    armor = np.concatenate([_slab(a, b, m, slab_pitch) for a, b, m in slabs])
    fc = len(front_x)
    jet0 = _jet(rows, pitch, front_x[0], 7000.0, cols)
    pc = armor.shape[0] + jet0.shape[0]
    frames = np.zeros((fc, pc, len(ATTRS)), dtype=np.float64)
    for f, x in enumerate(front_x):
        jet = _jet(rows, pitch, x, 7000.0, cols)
        frames[f] = np.concatenate([armor, jet])
    return _write_cache(tmp, name, frames, armor_block=armor_block)


# A front that advances 1 mm/frame from x=150: crosses 160 at frame 10, 190 at 40.
STEADY = 150.0 + np.arange(150) * 1.0


def test_finds_every_interface_including_the_bonded_one(tmp_path):
    """5 interfaces: 150/160 (void after), 190, 215 (BONDED), 235."""
    c = _build(tmp_path, "steady", STEADY)
    r = jg.measure(c)
    assert len(r["faces"]) == 5, r["faces"]
    # Tolerance is half the armor lattice pitch: the tool reads particle CENTRES,
    # which sit that far inside the true face. That offset is real in every cache
    # and is why `_table` times all arms to one shared face set (common-mode).
    for want, got in zip((150.0, 160.0, 190.0, 215.0, 235.0), r["faces"]):
        assert abs(got - want) <= 0.5 + 1e-6, (want, got)


def test_a_gap_only_rule_would_miss_the_bonded_contact(tmp_path):
    """RED CHECK for the assertion above.

    Segmenting on voids ALONE finds 150/160/190/235 and looks entirely healthy —
    4 interfaces, ascending, plausible. It silently loses x=215, the
    ceramic/backing contact, which is one of the four arrival times the study
    reports. This reproduces that rule and asserts it fails, so the test above is
    known to be checking something.
    """
    c = _build(tmp_path, "steady2", STEADY)
    manifest, frames = jg.load(c)
    col = {a: i for i, a in enumerate(manifest["attributes"])}
    f0 = np.asarray(frames[0])
    static = f0[:, col["vel_mag"]] <= jg.MOVING_MM_PER_MS

    xs = np.sort(f0[static, col["pos_x"]])
    sp = jg._spacing(xs)
    segs = np.split(np.arange(xs.size),
                    np.flatnonzero(np.diff(xs) > jg.VOID_SPACINGS * sp) + 1)
    gap_only = sorted({round(float(xs[s].min()), 2) for s in segs}
                      | {round(float(xs[s].max()), 2) for s in segs})
    merged = []
    for x in gap_only:
        if merged and x - merged[-1] <= jg.MERGE_SPACINGS * sp:
            merged[-1] = 0.5 * (merged[-1] + x)
        else:
            merged.append(x)

    assert len(merged) == 4, merged                      # the defect
    assert not any(abs(x - 215.0) < 1.0 for x in merged)  # 215 is gone
    assert len(jg.measure(c)["faces"]) == 5               # the tool is not fooled


def test_geometry_is_read_from_frame_zero_not_from_the_manifest(tmp_path):
    """A cache whose provenance block LIES must still measure the true stack."""
    lie = [{"material": "rha", "thickness": 999.0, "standoff": 42.0}]
    c = _build(tmp_path, "liar", STEADY, armor_block=lie)
    faces = jg.measure(c)["faces"]
    assert len(faces) == 5
    assert abs(faces[-1] - 235.0) <= 0.5 + 1e-6, faces  # half the lattice pitch
    # And the lie really is in the file — otherwise this test proves nothing.
    assert json.loads((c / "manifest.json").read_text())["armor"][0]["thickness"] == 999.0


def test_a_slower_front_reads_later(tmp_path):
    """The core reading. Half the rate must double the arrival time."""
    fast = _build(tmp_path, "fast", STEADY)
    slow = _build(tmp_path, "slow", 150.0 + np.arange(150) * 0.5)
    a = jg.arrival_us(jg.measure(fast), 190.0)
    b = jg.arrival_us(jg.measure(slow), 190.0)
    assert np.isfinite(a) and np.isfinite(b)
    assert b > a
    assert abs(b / a - 2.0) < 0.02, (a, b)


def test_never_reaching_is_not_late(tmp_path):
    """A front that stalls short must read NOT REACHED, never the window end.

    Clamping to the last frame is the natural bug, and it turns "did not arrive"
    into "arrived at 29.8 us" — a number that would sit in the table looking like
    a measurement and quietly flatten the very effect being measured.
    """
    stall = _build(tmp_path, "stall", np.minimum(150.0 + np.arange(150) * 1.0, 200.0))
    r = jg.measure(stall)
    assert np.isfinite(jg.arrival_us(r, 190.0))          # it did reach this one
    assert not np.isfinite(jg.arrival_us(r, 215.0))      # and not this one
    assert not np.isfinite(jg.arrival_us(r, 235.0))


def test_depth_at_window_end_cannot_see_what_arrival_time_can(tmp_path):
    """THE REASON THIS TOOL EXISTS — verified, not asserted in prose.

    Two arms that penetrate at visibly different rates but are at the SAME place
    when the window closes. Depth-at-end reads them as identical; arrival time
    separates them by ~40 %. If this test ever goes green on the depth assertion
    alone, the tool has no reason to exist.
    """
    # Arm A: steady 1.0 mm/frame to x=250 by frame 100, then coasts.
    a_front = np.minimum(150.0 + np.arange(150) * 1.0, 250.0)
    # Arm B: slower through the stack (0.7), then a fast free residual to the
    # same x=250 at the same final frame.
    b_front = np.concatenate([150.0 + np.arange(120) * 0.7,
                              np.linspace(234.0, 250.0, 30)])
    A = jg.measure(_build(tmp_path, "armA", a_front))
    B = jg.measure(_build(tmp_path, "armB", b_front))

    assert abs(A["depth_end"] - B["depth_end"]) < 0.5      # blind
    ta, tb = jg.arrival_us(A, 235.0), jg.arrival_us(B, 235.0)
    assert (tb - ta) / ta > 0.30, (ta, tb)                 # not blind


def test_lattice_reading_is_the_seeded_resolution(tmp_path):
    """15 rows across a 3 mm jet is 7.5 cells — what was seeded, not 3/dx."""
    c = _build(tmp_path, "lattice", STEADY, rows=15, pitch=0.2)
    r = jg.measure(c)
    assert r["rows_across"] == 15
    assert r["cells_across"] == pytest.approx(7.5)
    assert r["dx"] == pytest.approx(0.4, abs=1e-4)  # abs: the cache is float32
    # The naive route: dx from domain/grid_resolution = 300/768 = 0.390625 gives
    # 7.68 cells. Both describe the same jet; they differ by the rounding
    # `_fill_rect` does, and the tool must report the one that penetrated.
    assert abs(3.0 / 0.390625 - r["cells_across"]) > 0.1

    fine = jg.measure(_build(tmp_path, "lattice2", STEADY, rows=24, pitch=0.125))
    assert fine["rows_across"] == 24
    assert fine["cells_across"] == pytest.approx(12.0)


def test_the_residual_is_read_on_a_matched_clock(tmp_path):
    """The residual metric lives ENTIRELY after perforation, so it is the one most
    exposed to the lab-time confound §3.8 warns about: arms that break out at
    different times have had different amounts of free flight when a shared final
    frame arrives.

    Two arms with the SAME residual physics (identical free-flight velocity) that
    break out 5 us apart. The matched reading must call them equal. A final-frame
    reading sees different populations and need not — so the assertion is that the
    correction is WIRED, not that it happens to be small.
    """
    # Both coast at 1.0 mm/frame once through; B simply starts 25 frames later.
    a_front = np.concatenate([np.linspace(150.0, 236.0, 90), 236.0 + np.arange(60) * 1.0])
    b_front = np.concatenate([np.full(25, 150.0),
                              np.linspace(150.0, 236.0, 90), 236.0 + np.arange(35) * 1.0])
    # A LONG jet (80 mm), so "fraction through" is a curve rather than a step: a
    # short rigid block is entirely past the back face the instant its tip is, and
    # would make both readings 1.0 and the test vacuous.
    A = jg.measure(_build(tmp_path, "resA", a_front, cols=400))
    B = jg.measure(_build(tmp_path, "resB", b_front, cols=400))

    assert abs(B["t_break"] - A["t_break"]) > 4.0          # they really do differ
    # Matched: the same amount of jet is through, because the same time has passed.
    assert A["frac_through_matched"] == pytest.approx(B["frac_through_matched"], rel=0.02)
    # Unmatched: the later arm has had less flight, so less of it is through.
    assert B["frac_through"] < A["frac_through"] * 0.95


def test_a_matched_reading_that_falls_outside_the_window_is_withheld(tmp_path):
    """An arm that breaks out too late has NO matched reading — it must not be
    handed the final frame under a "matched" heading.

    Picking the nearest frame is the natural implementation and it clamps silently:
    the confound the matched metric exists to remove would come back wearing its
    label, which is worse than not having the metric.
    """
    # Breaks out at ~28 us of a 29.8 us window: less than RESIDUAL_ELAPSED_US left.
    late = np.concatenate([np.linspace(150.0, 234.0, 141), 234.0 + np.arange(9) * 1.0])
    r = jg.measure(_build(tmp_path, "late", late, cols=400))
    assert np.isfinite(r["t_break"])
    assert r["t_break"] + jg.RESIDUAL_ELAPSED_US > r["t_us"][-1]   # genuinely short
    assert not np.isfinite(r["residual_v_matched"])                # withheld
    assert np.isfinite(r["residual_v"])                            # not the same reading

    # The control: an arm with room to spare does get one.
    ok = jg.measure(_build(tmp_path, "early", STEADY, cols=400))
    assert np.isfinite(ok["residual_v_matched"])


def test_arms_that_are_not_the_same_scenario_are_refused(tmp_path):
    """A shared face set is only legitimate if the arms share a stack."""
    a = _build(tmp_path, "sameA", STEADY)
    b = _build(tmp_path, "diffB", STEADY,
               slabs=((150.0, 160.0, 1.0), (190.0, 215.0, 2.0), (215.0, 260.0, 1.0)))
    assert jg.main([str(a), str(b)]) == 2
    assert jg.main([str(a), str(_build(tmp_path, "sameB", STEADY))]) == 0
