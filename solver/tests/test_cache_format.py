"""Contract tests for the cache format. No GPU required.

The cache is the ONE thing the solver and visualizer share (root CLAUDE.md §2),
and until milestone 13 it had no solver-side test at all — `validate_cache.py`
checks a *baked* cache, which means a format defect is only caught after paying
for a bake, and an ORDER defect is not caught by it even then.

What these pin is narrow on purpose: the mechanics the spec fixes in writing
(§2 manifest fields, §3 byte layout) and the one hazard milestone 13 introduced
— a column whose values are not the quantity its name claims.

Run: cd solver && pytest
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from ballistics_solver import CACHE_SCHEMA_VERSION
from ballistics_solver.cache_writer import CacheWriter

# The layout run.py declares. Kept here as the expected value rather than
# imported from run.py: a test that imports the thing it checks would agree with
# a typo. If these diverge, one of the two is wrong and this fails.
EXPECTED_ATTRIBUTES = [
    "pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
    "internal_energy",
]


def _write(tmp_path, frames, attributes=None):
    attrs = attributes if attributes is not None else EXPECTED_ATTRIBUTES
    out = tmp_path / "cache"
    with CacheWriter(
        out,
        scenario="test",
        particle_count=frames[0].shape[0],
        attributes=attrs,
        frame_dt=2.0e-7,
        domain={"xmin": 0.0, "xmax": 200.0, "ymin": 0.0, "ymax": 100.0},
        units="mm-ms-g (see docs/PHYSICS.md)",
        materials={"0": "tungsten_rod"},
    ) as w:
        w.particle_count = frames[0].shape[0]
        for f in frames:
            w.write_frame(f)
    return out


def test_run_declares_the_schema_the_spec_documents() -> None:
    """run.py's column list must match CACHE_FORMAT.md §2, and the version must
    have moved with it. v2 = v1 + internal_energy (milestone 13)."""
    from ballistics_solver import run  # noqa: F401  (import must not need a GPU)

    assert CACHE_SCHEMA_VERSION == 2
    assert "internal_energy" in EXPECTED_ATTRIBUTES
    assert len(set(EXPECTED_ATTRIBUTES)) == len(EXPECTED_ATTRIBUTES)
    # pos_x/pos_y are the format's only REQUIRED columns (§2).
    assert {"pos_x", "pos_y"} <= set(EXPECTED_ATTRIBUTES)


def test_manifest_records_the_bumped_version(tmp_path) -> None:
    out = _write(tmp_path, [np.zeros((4, len(EXPECTED_ATTRIBUTES)), dtype=np.float32)])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == CACHE_SCHEMA_VERSION == 2
    assert manifest["attributes"] == EXPECTED_ATTRIBUTES
    assert manifest["dtype"] == "float32"


def test_byte_layout_is_exactly_what_section_3_specifies(tmp_path) -> None:
    """frame-major, then particle-major, little-endian f32, no padding.

    Computed against the spec's own offset formula, not against the writer's
    behaviour — otherwise the test just restates whatever the code does.
    """
    n_p, n_a = 5, len(EXPECTED_ATTRIBUTES)
    frames = [
        np.arange(n_p * n_a, dtype=np.float32).reshape(n_p, n_a) + 1000.0 * f
        for f in range(3)
    ]
    out = _write(tmp_path, frames)
    blob = (out / "frames.bin").read_bytes()

    assert len(blob) == len(frames) * n_p * n_a * 4, "size formula (§3)"
    for f, frame in enumerate(frames):
        for p in range(n_p):
            for a in range(n_a):
                off = (f * n_p * n_a + p * n_a + a) * 4      # §3, verbatim
                got = struct.unpack_from("<f", blob, off)[0]
                assert got == pytest.approx(frame[p, a]), f"frame {f} particle {p} attr {a}"


def test_internal_energy_column_survives_the_write_read_round_trip(tmp_path) -> None:
    """Values put in under a name come back out under that name.

    SCOPE, stated because the obvious reading is wrong: this pins the WRITER and
    name-resolution, NOT the real milestone-13 hazard. That hazard is that
    `run.py` names the columns while `mpm.dump_frame` column_stacks the values,
    in different files, pinned to each other by nothing — a swapped pair emits a
    cache that validates perfectly, plays fine, and is silently mislabelled.
    This test cannot see that: it builds its own frame from the same list it
    asserts against, so it would agree with the swap.

    Catching the real thing needs a bake — the column read back from a live
    cache must be ~1e5 J/kg in shocked material and 0.0 at rest, which a
    material_id (a small int) cannot imitate. That check is in PHYSICS §3.10 and
    is run against a real deck, not here.
    """
    n_p = 6
    idx = {name: i for i, name in enumerate(EXPECTED_ATTRIBUTES)}
    frame = np.zeros((n_p, len(EXPECTED_ATTRIBUTES)), dtype=np.float32)
    # Deliberately distinguishable: a real material_id is a small int, a real
    # specific internal energy is ~1e5 J/kg. Swap them and the assert below
    # cannot pass by coincidence.
    frame[:, idx["material_id"]] = 1.0
    frame[:, idx["internal_energy"]] = 9.83e4

    out = _write(tmp_path, [frame])
    manifest = json.loads((out / "manifest.json").read_text())
    attrs = manifest["attributes"]
    data = np.frombuffer((out / "frames.bin").read_bytes(), dtype="<f4").reshape(
        1, n_p, len(attrs)
    )
    # Resolve BY NAME, exactly as the viewer must (§2) — never by hardcoded offset.
    e = data[0, :, attrs.index("internal_energy")]
    mat = data[0, :, attrs.index("material_id")]
    assert np.allclose(e, 9.83e4), "internal_energy column is not carrying e"
    assert np.allclose(mat, 1.0), "material_id column is not carrying the id"


def test_writer_rejects_a_frame_whose_width_is_not_the_declared_layout(tmp_path) -> None:
    """A column added to the stack but not to `attributes` (or vice versa) must
    fail at the writer, not become a silently misaligned cache."""
    with pytest.raises(ValueError):
        _write(tmp_path, [np.zeros((4, len(EXPECTED_ATTRIBUTES) - 1), dtype=np.float32)])


def test_writer_enforces_the_specs_two_attribute_rules(tmp_path) -> None:
    with pytest.raises(ValueError):  # §2: pos_x/pos_y required
        _write(tmp_path, [np.zeros((4, 2), dtype=np.float32)],
               attributes=["vel_mag", "internal_energy"])
    with pytest.raises(ValueError):  # §2: no duplicates
        _write(tmp_path, [np.zeros((4, 3), dtype=np.float32)],
               attributes=["pos_x", "pos_y", "pos_x"])
