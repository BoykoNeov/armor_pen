"""Contract tests for tools/measure_wall_contact.py — that it goes RED.

The tool's first two real readings were `apfsds_vs_rha`: zero arrivals on all
four walls, contamination +0.00 % on both aggregates. That is either a clean
deck or a dead instrument, and the two look identical from the outside. This
repo has shipped the second one often enough to have a memory note about it, so
every green reading here is paired with a defect the same assertion must catch:

  * arrivals — a control where nothing travels, against a twin where debris
    crosses into the band. Zero vs one, one particle apart.
  * the seeded-at-the-wall exclusion — armor laid ON the wall (which every deck
    does deliberately, §1.1) must NOT register, or the tool alarms on all 30
    caches and has measured nothing. Nor may a particle that merely starts
    *near* the wall and creeps in: the exclusion is a start-distance rule, and
    it has to hold at its own boundary.
  * contamination — a wall-touched particle left in the live penetrator set,
    so the +0.00 % above is a measured zero rather than an unwired path.
  * the §1.1.1 symmetry check — fed the exact signature of the historical
    failure (`rha` 0.88..119.61 against a 120 mm domain) and its fixed twin
    (0.88..119.12, mirroring to 0.88 vs 0.88).

`measure_wall_contact.py` lives in tools/ and imports neither half of the repo
(CLAUDE.md §3); loading it by path here keeps that true — the tool does not
learn about the solver, the solver's test suite just reads it.

Run: cd solver && pytest
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ballistics_solver.cache_writer import CacheWriter

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "measure_wall_contact.py"
_spec = importlib.util.spec_from_file_location("_measure_wall_contact", _TOOL)
wall = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wall)

ATTRS = ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
         "internal_energy"]
FRAME_DT = 2.0e-7          # s -> 0.2 us per frame, as the shipped decks use
PROJECTILE = {
    "kind": "kinetic", "material": "tungsten_rod", "length": 60.0,
    "diameter": 8.0, "velocity": 1600.0, "tail_velocity": None,
    "angle_deg": 0.0, "nose_shape": "conical",
}


def build(tmp_path, *, y, vel, mat, domain, damage=None, x=None,
          angle_deg=0.0, name="t"):
    """Write a synthetic cache from per-particle, per-frame tracks.

    `y` is (n_particles, n_frames); everything else broadcasts against it.
    """
    y = np.asarray(y, dtype=float)
    n_p, n_f = y.shape
    vel = np.broadcast_to(np.asarray(vel, dtype=float), (n_p, n_f))
    dmg = np.zeros((n_p, n_f)) if damage is None else np.broadcast_to(
        np.asarray(damage, dtype=float), (n_p, n_f))
    xs = np.broadcast_to(
        np.asarray(0.5 * (domain["xmin"] + domain["xmax"]) if x is None else x,
                   dtype=float), (n_p, n_f))
    mat = np.asarray(mat, dtype=float)

    out = tmp_path / name
    proj = dict(PROJECTILE, angle_deg=angle_deg)
    with CacheWriter(
        out, scenario=name, particle_count=n_p, attributes=ATTRS,
        frame_dt=FRAME_DT, domain=domain, units="mm-ms-g (see docs/PHYSICS.md)",
        materials={"0": "tungsten_rod", "1": "rha"}, projectile=proj,
        armor=[{"material": "rha", "thickness": 40.0, "standoff": 0.0}],
        material_descriptions={"0": "a dense rod", "1": "a steel plate"},
    ) as w:
        for f in range(n_f):
            fr = np.zeros((n_p, len(ATTRS)), dtype=np.float32)
            fr[:, ATTRS.index("pos_x")] = xs[:, f]
            fr[:, ATTRS.index("pos_y")] = y[:, f]
            fr[:, ATTRS.index("vel_mag")] = vel[:, f]
            fr[:, ATTRS.index("damage")] = dmg[:, f]
            fr[:, ATTRS.index("material_id")] = mat
            w.write_frame(fr)
    return out


DOMAIN = {"xmin": 0.0, "xmax": 100.0, "ymin": 0.0, "ymax": 50.0}

# Frame 8 is still clear of the 1.2 mm band, frame 9 is inside it: the first
# arrival is therefore frame 9 == 1.8 us, and asserting the exact frame is what
# separates "noticed something" from "measured when".
TRAVELS = [25.0] * 8 + [5.0, 0.4]
STAYS = [25.0] * 10


def _tracks(bottom_track):
    """Four particles, differing only in what the debris particle does.

    0  armor laid ON the y_lo wall (0.5 mm) and never moving — every deck does
       this on purpose, and it must never register.
    1  armor debris: `bottom_track`, either travelling to the wall or staying.
    2  a creeper starting 2.0 mm out (inside the 3.0 mm keepout) and reaching
       0.3 mm — the exclusion rule tested at its own boundary.
    3  the penetrator, mid-domain, untouched.
    """
    return dict(
        y=[[0.5] * 10, bottom_track, list(np.linspace(2.0, 0.3, 10)), STAYS],
        vel=[[0.0] * 10, [0.0] + [500.0] * 9, [0.0] * 10, [1600.0] * 10],
        mat=[1, 1, 1, 0],
    )


def test_control_nothing_travels_reads_zero(tmp_path):
    """The green half. Armor on the wall, a creeper, and a rod that stays."""
    c = build(tmp_path, domain=DOMAIN, name="control", **_tracks(STAYS))
    r = wall.measure(c)
    for w in wall.WALLS:
        assert r["walls"][w]["n_arrived"] == 0, w
    assert r["n_touched"] == 0


def test_debris_reaching_the_wall_goes_red(tmp_path):
    """The red half — one particle apart from the control above."""
    c = build(tmp_path, domain=DOMAIN, name="red", **_tracks(TRAVELS))
    r = wall.measure(c)
    y_lo = r["walls"]["y_lo"]
    assert y_lo["n_arrived"] == 1
    assert y_lo["first_us"] == pytest.approx(1.8)
    assert y_lo["by_material"]["rha"]["n"] == 1
    assert y_lo["max_v_at_arrival"] == pytest.approx(500.0)
    # The other three walls stay silent: a tool that flags a wall nothing
    # approached is as useless as one that misses the wall something hit.
    for w in ("x_lo", "x_hi", "y_hi"):
        assert r["walls"][w]["n_arrived"] == 0, w


def test_armor_seeded_on_the_wall_is_never_eligible(tmp_path):
    """Every deck lays armor wall-to-wall (§1.1). If that registered, the tool
    would alarm on all 30 caches and discriminate nothing."""
    c = build(tmp_path, domain=DOMAIN, name="seeded", **_tracks(TRAVELS))
    r = wall.measure(c)
    # 4 particles, 2 of them start inside the 3 mm keepout (the 0.5 mm armor and
    # the 2.0 mm creeper), so exactly 2 are eligible at y_lo.
    assert r["walls"]["y_lo"]["n_eligible"] == 2
    # ...and of those, only the traveller arrives — not the creeper.
    assert r["walls"]["y_lo"]["n_arrived"] == 1


def test_keepout_is_what_excludes_the_creeper(tmp_path):
    """Drop the keepout below the creeper's start and it becomes eligible — so
    the exclusion above is the RULE doing work, not the creeper failing to move.
    """
    c = build(tmp_path, domain=DOMAIN, name="creep", **_tracks(TRAVELS))
    assert wall.measure(c, keepout=1.5)["walls"]["y_lo"]["n_arrived"] == 2


def test_a_wide_sweep_radius_cannot_readmit_the_frame_zero_false_positive(tmp_path):
    """The keepout's whole job, walking back in through the sweep.

    At `keepout=3.0` and `D=5.0`, a particle seeded 3.5 mm out is "within 5 mm"
    at frame 0 having travelled nowhere. A swept count must therefore demand a
    start clear of the radius as well — otherwise the widest column reports the
    seeding, which is exactly what the tool's first run over the real caches
    did (a ~3000-particle column on every deck, all of it stationary armor).
    """
    c = build(
        tmp_path, domain=DOMAIN, name="wide",
        # 0: parked at 3.5 mm forever. 1: travels from 25 mm into the band.
        y=[[3.5] * 10, TRAVELS], vel=0.0, mat=[1, 1],
    )
    sweep = wall.measure(c)["walls"]["y_lo"]["sweep"]
    assert sweep[5.0] == 1, "the parked particle must not count as an arrival"
    # The traveller starts 25 mm out, so it clears every radius and registers
    # in every column — the sweep is not simply switched off.
    assert all(sweep[D] == 1 for D in wall.ARRIVE_SWEEP)


def test_contamination_is_measured_not_hardwired(tmp_path):
    """A wall-touched particle left LIVE in the penetrator set must move the
    aggregate. Without this, the +0.00 % the shipped decks report could equally
    be an unwired code path."""
    # Rod particle 1 flies to the wall and ends slow; rod particle 0 does not.
    c = build(
        tmp_path, domain=DOMAIN, name="contam",
        y=[STAYS, TRAVELS], vel=[[1600.0] * 10, [1600.0] * 9 + [100.0]],
        mat=[0, 0],
    )
    r = wall.measure(c)
    ct = r["contamination"]
    assert ct["proj_touched"] == 1
    assert ct["rod_resid_v"] == pytest.approx(850.0)      # mean(1600, 100)
    assert ct["rod_resid_v_clean"] == pytest.approx(1600.0)
    assert ct["rod_resid_v_delta_pct"] > 80.0


def test_symmetry_check_catches_the_historical_signature(tmp_path):
    """§1.1.1's cheap check, fed the exact numbers the dead high walls produced:
    `rha` spanning 0.88..119.61 in a 120 mm domain — held off the working bottom
    mirror, jammed onto the top position clamp."""
    dom = {"xmin": 0.0, "xmax": 100.0, "ymin": 0.0, "ymax": 120.0}
    dead = build(tmp_path, domain=dom, name="dead",
                 y=[[0.88] * 3, [119.61] * 3], vel=0.0, mat=[1, 1])
    s = wall.measure(dead)["symmetry"]
    assert s["applicable"]
    assert s["by_material"]["rha"]["asym_mm"] == pytest.approx(0.49, abs=1e-3)

    live = build(tmp_path, domain=dom, name="live",
                 y=[[0.88] * 3, [119.12] * 3], vel=0.0, mat=[1, 1])
    s = wall.measure(live)["symmetry"]
    assert s["by_material"]["rha"]["asym_mm"] == pytest.approx(0.0, abs=1e-3)


def test_symmetry_check_is_skipped_where_it_would_mean_nothing(tmp_path):
    """An oblique deck is not symmetric by construction — the rod is tilted and
    the impact is deliberately off-centre (§3.2) — so an asymmetry there carries
    no information about the walls."""
    c = build(tmp_path, domain=DOMAIN, name="oblique", angle_deg=55.0,
              **_tracks(TRAVELS))
    assert wall.measure(c)["symmetry"]["applicable"] is False


def test_stride_can_only_miss_an_arrival_never_invent_one(tmp_path):
    """The documented direction of the sampling bias, pinned. Frame 9 is the
    only frame inside the band, so an even stride steps straight over it."""
    c = build(tmp_path, domain=DOMAIN, name="stride", **_tracks(TRAVELS))
    assert wall.measure(c, stride=1)["walls"]["y_lo"]["n_arrived"] == 1
    assert wall.measure(c, stride=2)["walls"]["y_lo"]["n_arrived"] == 0
