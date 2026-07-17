"""Contract tests for tools/validate_cache.py — specifically, that it goes RED.

This file exists because of the repo's most-repeated defect: a checker is trusted
because it is green, when it is green because it is BLIND. `validate_cache.py`
grew rule 9 (the v3 scenario block, CACHE_FORMAT §2.1) and rule 9 was verified to
fail on every way the block can break — but that verification was a throwaway
script, which means a later refactor could neuter the rule and nothing would
notice. The verification belongs here, where it runs.

The controls matter as much as the failures. `tail_velocity: null` MUST stay
green: it means "uniform", and it is every KE deck in the repo — a checker that
rejects it would be worse than one that checks nothing.

`validate_cache.py` lives in tools/ and imports neither half of the repo
(CLAUDE.md §3); loading it by path here keeps that true — the tool does not learn
about the solver, the solver's test suite just reads it.

Run: cd solver && pytest
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ballistics_solver.cache_writer import CacheWriter

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "validate_cache.py"
_spec = importlib.util.spec_from_file_location("_validate_cache", _TOOL)
validate_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_cache)

ATTRS = ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
         "internal_energy"]
PROJECTILE = {
    "kind": "kinetic", "material": "tungsten_rod", "length": 60.0,
    "diameter": 8.0, "velocity": 1600.0, "tail_velocity": None,
    "angle_deg": 0.0, "nose_shape": "conical",
}


@pytest.fixture
def cache(tmp_path):
    """A minimal, genuinely valid v3 cache."""
    out = tmp_path / "cache"
    with CacheWriter(
        out,
        scenario="test", particle_count=3, attributes=ATTRS, frame_dt=2.0e-7,
        domain={"xmin": 0.0, "xmax": 200.0, "ymin": 0.0, "ymax": 100.0},
        units="mm-ms-g (see docs/PHYSICS.md)",
        materials={"0": "tungsten_rod", "1": "rha"},
        projectile=PROJECTILE,
        armor=[{"material": "rha", "thickness": 40.0, "standoff": 0.0}],
        material_descriptions={"0": "a dense rod", "1": "a steel plate"},
    ) as w:
        w.particle_count = 3
        w.write_frame(np.zeros((3, len(ATTRS)), dtype=np.float32))
    return out


def _revalidate(cache_dir: Path, mutate) -> None:
    """Apply `mutate` to the manifest, then validate. Raises CacheInvalid if the
    tool does its job."""
    path = cache_dir / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    mutate(m)
    path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    validate_cache.validate(cache_dir)


def test_the_fixture_is_actually_valid(cache) -> None:
    """The control. Without this, every assert below could pass because the
    baseline was broken all along and rule 9 never ran at all."""
    validate_cache.validate(cache)


def test_a_uniform_projectile_stays_valid(cache) -> None:
    """tail_velocity: null means UNIFORM — every KE deck in the repo. Rejecting it
    would be a far worse bug than anything rule 9 catches."""
    _revalidate(cache, lambda m: m["projectile"].update(tail_velocity=None))


def test_a_graded_projectile_stays_valid(cache) -> None:
    """...and a number means a velocity-graded jet. Both are legal (§2.1)."""
    _revalidate(cache, lambda m: m["projectile"].update(tail_velocity=2000.0))


@pytest.mark.parametrize("field", ["projectile", "armor", "material_descriptions"])
def test_a_missing_v3_field_is_refused(cache, field) -> None:
    with pytest.raises(validate_cache.CacheInvalid):
        _revalidate(cache, lambda m: m.pop(field))


@pytest.mark.parametrize("mutate, why", [
    (lambda m: m["projectile"].pop("velocity"), "projectile missing a key"),
    (lambda m: m["projectile"].update(velocity="fast"), "projectile key wrong type"),
    # bool is a subclass of int in Python, so a naive isinstance check waves this
    # through. Nothing emits it; a validator exists to refuse it anyway.
    (lambda m: m["projectile"].update(velocity=True), "a bool where a number goes"),
    (lambda m: m.update(armor=[]), "empty armor stack"),
    (lambda m: m.update(armor=["rha"]), "armor layer is not an object"),
    (lambda m: m["armor"][0].pop("thickness"), "armor layer missing a key"),
    (lambda m: m["material_descriptions"].pop("1"), "a material with no description"),
    (lambda m: m["material_descriptions"].update({"9": "ghost"}), "a description with no material"),
    (lambda m: m["material_descriptions"].update({"1": "   "}), "a blank description"),
    (lambda m: m["material_descriptions"].update({"1": 42}), "a description that is not text"),
    (lambda m: m.update(schema_version=2), "a stale schema version"),
])
def test_a_broken_scenario_block_is_refused(cache, mutate, why) -> None:
    """Rule 9, one break at a time. Each of these was watched going red before the
    green above was believed — that is the whole point of the file."""
    with pytest.raises(validate_cache.CacheInvalid):
        _revalidate(cache, mutate)


def test_the_description_map_must_track_the_material_list(cache) -> None:
    """The one cross-field invariant the tool CAN see. Both maps are emitted
    together from one LIBRARY, so they only drift if something upstream is wrong —
    which makes the drift worth refusing rather than tolerating."""
    with pytest.raises(validate_cache.CacheInvalid, match="key set"):
        _revalidate(cache, lambda m: m["materials"].update({"2": "ceramic"}))
