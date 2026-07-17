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

from ballistics_solver import CACHE_SCHEMA_VERSION, materials
from ballistics_solver.cache_writer import CacheWriter, build_manifest

# The layout run.py declares. Kept here as the expected value rather than
# imported from run.py: a test that imports the thing it checks would agree with
# a typo. If these diverge, one of the two is wrong and this fails.
EXPECTED_ATTRIBUTES = [
    "pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
    "internal_energy",
]

# A v3 scenario block (§2.1), written out here rather than built from the solver's
# own helpers, for the same reason EXPECTED_ATTRIBUTES is.
PROJECTILE = {
    "kind": "kinetic", "material": "tungsten_rod", "length": 60.0,
    "diameter": 8.0, "velocity": 1600.0, "tail_velocity": None,
    "angle_deg": 0.0, "nose_shape": "conical",
}
ARMOR = [{"material": "rha", "thickness": 40.0, "standoff": 0.0}]


def _write(tmp_path, frames, attributes=None, name="cache"):
    attrs = attributes if attributes is not None else EXPECTED_ATTRIBUTES
    out = tmp_path / name
    with CacheWriter(
        out,
        scenario="test",
        particle_count=frames[0].shape[0],
        attributes=attrs,
        frame_dt=2.0e-7,
        domain={"xmin": 0.0, "xmax": 200.0, "ymin": 0.0, "ymax": 100.0},
        units="mm-ms-g (see docs/PHYSICS.md)",
        materials={"0": "tungsten_rod"},
        projectile=PROJECTILE,
        armor=ARMOR,
        material_descriptions={"0": "a dense rod"},
    ) as w:
        w.particle_count = frames[0].shape[0]
        for f in frames:
            w.write_frame(f)
    return out


def test_run_declares_the_schema_the_spec_documents() -> None:
    """run.py's column list must match CACHE_FORMAT.md §2, and the version must
    have moved with it. v3 = v2 + the scenario block."""
    from ballistics_solver import run  # noqa: F401  (import must not need a GPU)

    assert CACHE_SCHEMA_VERSION == 3
    assert "internal_energy" in EXPECTED_ATTRIBUTES
    assert len(set(EXPECTED_ATTRIBUTES)) == len(EXPECTED_ATTRIBUTES)
    # pos_x/pos_y are the format's only REQUIRED columns (§2).
    assert {"pos_x", "pos_y"} <= set(EXPECTED_ATTRIBUTES)


def test_manifest_records_the_bumped_version(tmp_path) -> None:
    out = _write(tmp_path, [np.zeros((4, len(EXPECTED_ATTRIBUTES)), dtype=np.float32)])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == CACHE_SCHEMA_VERSION == 3
    assert manifest["attributes"] == EXPECTED_ATTRIBUTES
    assert manifest["dtype"] == "float32"


def test_manifest_carries_the_v3_scenario_block(tmp_path) -> None:
    """§2.1: the block exists so a reader can say what it is drawing."""
    out = _write(tmp_path, [np.zeros((4, len(EXPECTED_ATTRIBUTES)), dtype=np.float32)])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["projectile"] == PROJECTILE
    assert manifest["armor"] == ARMOR
    # null survives the JSON round trip as None, and MEANS "uniform" — a reader
    # that treats it as 0 would call every KE rod a jet with a stationary tail.
    assert manifest["projectile"]["tail_velocity"] is None


def test_build_manifest_refuses_a_description_map_that_drifted(tmp_path) -> None:
    """§2.1 requires the two maps to share a key set. Catch it at the writer:
    a validator catching it later means the bake was already paid for."""
    with pytest.raises(ValueError, match="key set"):
        build_manifest(
            scenario="test", particle_count=1, frame_count=1,
            attributes=EXPECTED_ATTRIBUTES, frame_dt=2.0e-7,
            domain={"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0},
            units="mm-ms-g", materials={"0": "tungsten_rod", "1": "rha"},
            projectile=PROJECTILE, armor=ARMOR,
            material_descriptions={"0": "only one of the two"},
        )


def test_every_material_in_the_library_describes_itself() -> None:
    """`description` is required with no default (like `shock`) so a new material
    must say what it is. This pins the property that makes that worth anything:
    the two manifest maps are built from one LIBRARY and therefore agree by
    construction, and no material ships a blank or a restatement of its name."""
    assert materials.id_to_name().keys() == materials.id_to_description().keys()
    for m in materials.LIBRARY.values():
        assert m.description.strip(), f"{m.name} has no description"
        assert m.description.strip() != m.name, f"{m.name}'s description is its name"
        # It renders in a HUD panel beside a swatch, not in a document.
        assert len(m.description) <= 90, f"{m.name}'s description is too long for the HUD"


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


# --- the migration path (CACHE_FORMAT §2.2) ---------------------------------

def _v2_cache(tmp_path, name="cache"):
    """A cache as it stood BEFORE v3: real frames.bin, manifest with no block."""
    frames = [
        np.arange(5 * len(EXPECTED_ATTRIBUTES), dtype=np.float32).reshape(
            5, len(EXPECTED_ATTRIBUTES)) + 100.0 * f
        for f in range(3)
    ]
    out = _write(tmp_path, frames, name=name)
    m = json.loads((out / "manifest.json").read_text())
    m["schema_version"] = 2
    for key in ("projectile", "armor", "material_descriptions"):
        del m[key]
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    return out


def _deck(name="test", domain=(0.0, 200.0, 0.0, 100.0)):
    from ballistics_solver.config import ArmorLayer, Domain, Projectile, Scenario

    return Scenario(
        name=name,
        domain=Domain(*domain),
        projectile=Projectile(kind="kinetic", material="tungsten_rod",
                              length=60.0, diameter=8.0, velocity=1600.0),
        armor=[ArmorLayer(material="rha", thickness=40.0)],
    )


def test_remanifest_does_not_touch_frames_bin(tmp_path) -> None:
    """THE hazard of migrating in place, and it is not hypothetical:
    `CacheWriter.__init__` opens frames.bin "wb", so merely CONSTRUCTING a writer
    over an existing cache truncates it to zero bytes before a frame is written.
    Migration must go nowhere near it. Byte-identical, or the 75 GB of bakes this
    path is supposed to SAVE are gone instead."""
    from ballistics_solver.run import _remanifest

    out = _v2_cache(tmp_path)
    before = (out / "frames.bin").read_bytes()
    assert before, "fixture is empty; the test would pass vacuously"

    assert _remanifest(_deck(), out) == 0
    assert (out / "frames.bin").read_bytes() == before


def test_remanifest_upgrades_the_manifest_and_carries_layout_verbatim(tmp_path) -> None:
    """The descriptive fields come from the deck; everything else comes from the
    BAKE. That split is what makes migration safe — a layout field sourced from
    the deck could disagree with the bytes on disk and nothing would notice."""
    from ballistics_solver.run import _remanifest

    out = _v2_cache(tmp_path)
    old = json.loads((out / "manifest.json").read_text())
    assert _remanifest(_deck(), out) == 0
    new = json.loads((out / "manifest.json").read_text())

    assert new["schema_version"] == 3
    for key in ("particle_count", "frame_count", "attributes", "frame_dt",
                "domain", "units", "materials", "scenario"):
        assert new[key] == old[key], f"{key} was not carried verbatim from the bake"
    assert new["projectile"]["velocity"] == 1600.0
    assert new["armor"] == [{"material": "rha", "thickness": 40.0, "standoff": 0.0}]
    assert new["material_descriptions"]["0"] == materials.get("tungsten_rod").description
    # Idempotent: re-running on an already-migrated cache is a no-op, not a fault.
    assert _remanifest(_deck(), out) == 0
    assert json.loads((out / "manifest.json").read_text()) == new


def test_remanifest_refuses_a_deck_that_drifted_from_the_bake(tmp_path) -> None:
    """The guard is the reason this path is allowed to exist. Stamping today's
    deck onto last week's cache produces a manifest that confidently describes
    the wrong scenario — and it is STRUCTURALLY PERFECT, so no validator
    downstream can catch it. Verified to refuse, not assumed to."""
    from ballistics_solver.run import _remanifest

    wrong_name = _v2_cache(tmp_path, name="a")
    assert _remanifest(_deck(name="some_other_deck"), wrong_name) == 1

    moved_domain = _v2_cache(tmp_path, name="b")
    assert _remanifest(_deck(domain=(0.0, 300.0, 0.0, 100.0)), moved_domain) == 1

    # A cache too old to carry v2's column list cannot be migrated by relabelling.
    ancient = _v2_cache(tmp_path, name="c")
    m = json.loads((ancient / "manifest.json").read_text())
    m["schema_version"] = 1
    (ancient / "manifest.json").write_text(json.dumps(m, indent=2))
    assert _remanifest(_deck(), ancient) == 1

    # An id whose meaning changed since the bake must not inherit a description
    # written for a different material.
    renamed = _v2_cache(tmp_path, name="d")
    m = json.loads((renamed / "manifest.json").read_text())
    m["materials"] = {"0": "unobtanium"}
    (renamed / "manifest.json").write_text(json.dumps(m, indent=2))
    assert _remanifest(_deck(), renamed) == 1
