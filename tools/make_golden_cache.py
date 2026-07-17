#!/usr/bin/env python3
"""Regenerate the tiny golden fixture at visualizer/fixtures/tiny_golden_cache/.

This is the DOCUMENTED command referenced by CLAUDE.md §9 and CACHE_FORMAT §6.
The fixture is a canonical, committed 3-frame cache that:

  - lets the visualizer be built and tested with NO solver present, and
  - is an *independent* implementation of docs/CACHE_FORMAT.md (stdlib only,
    no solver import) — so `validate_cache.py` passing on it genuinely
    cross-checks the contract rather than testing the writer against itself.

    python tools/make_golden_cache.py

Regenerate only via this command, and whenever schema_version bumps (§6).
The scene is a trivial illustrative toy (a small tungsten rod approaching an
RHA plate) — it is not a physics result, just well-formed data.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

SCHEMA_VERSION = 3
ATTRIBUTES = ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
              "internal_energy"]
FRAME_COUNT = 3
DOMAIN = {"xmin": 0.0, "xmax": 200.0, "ymin": 0.0, "ymax": 100.0}
MATERIALS = {"0": "tungsten_rod", "1": "rha"}

# The v3 scenario block (§2.1). Hand-written rather than imported from the
# solver's materials.py / config.py — that is the whole point of this file (see
# the module docstring): the fixture is an INDEPENDENT implementation of the
# spec, so `validate_cache.py` passing on it cross-checks the contract instead of
# testing the solver against itself. Importing the library here to save typing
# three strings would quietly throw that away.
#
# It describes the toy scene below, which is illustrative and not a physics
# result — same as every other number in this file.
PROJECTILE = {
    "kind": "kinetic",
    "material": "tungsten_rod",
    "length": 24.0,
    "diameter": 8.0,
    "velocity": 1600.0,
    "tail_velocity": None,   # uniform — a rod, not a jet
    "angle_deg": 0.0,
    "nose_shape": "conical",
}
ARMOR = [{"material": "rha", "thickness": 40.0, "standoff": 0.0}]
# Same key set as MATERIALS — §2.1 requires it, and rule 9 checks it.
MATERIAL_DESCRIPTIONS = {
    "0": "Tungsten heavy alloy: very dense, tough KE long-rod penetrator.",
    "1": "Rolled homogeneous armor: the baseline steel plate.",
}

# 8 rod particles (material 0) + 12 plate particles (material 1) = 20 total.
ROD = 8
PLATE = 12
PARTICLE_COUNT = ROD + PLATE

OUT = (
    Path(__file__).resolve().parents[1]
    / "visualizer" / "fixtures" / "tiny_golden_cache"
)


def _particle(p: int, frame: int) -> list[float]:
    """Build one particle record for a given frame. Illustrative, not physical."""
    if p < ROD:  # tungsten rod, moving right toward the plate
        x = 30.0 + p * 3.0 + frame * 8.0
        y = 50.0
        vel = 1600.0
        stress = 200.0 * frame
        damage = 0.0
        mat = 0
        # Specific internal energy, J/kg (schema v2). 0 at t=0 = the reference
        # state, warming as the toy rod works — illustrative, like every other
        # number here, but the right ORDER (a real shocked metal reads ~1e5).
        energy = 4.0e4 * frame
    else:  # RHA plate column at x=120, taking a little damage over frames
        q = p - ROD
        x = 120.0
        y = 20.0 + q * 5.0
        vel = 0.0
        stress = 50.0 * frame + 5.0 * q
        damage = min(1.0, 0.15 * frame * math.exp(-abs(q - 6) / 3.0))
        mat = 1
        energy = 2.5e4 * frame * math.exp(-abs(q - 6) / 3.0)
    return [x, y, vel, stress, damage, float(mat), energy]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "frames.bin").open("wb") as blob:
        for frame in range(FRAME_COUNT):          # frame-major (§3)
            for p in range(PARTICLE_COUNT):       # then particle-major
                rec = _particle(p, frame)
                blob.write(struct.pack(f"<{len(ATTRIBUTES)}f", *rec))  # little-endian f32

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scenario": "tiny_golden_cache",
        "particle_count": PARTICLE_COUNT,
        "frame_count": FRAME_COUNT,
        "attributes": ATTRIBUTES,
        "dtype": "float32",
        "frame_dt": 2.0e-6,
        "domain": DOMAIN,
        "units": "mm-ms-g (see docs/PHYSICS.md)",
        "materials": MATERIALS,
        "projectile": PROJECTILE,
        "armor": ARMOR,
        "material_descriptions": MATERIAL_DESCRIPTIONS,
    }
    with (OUT / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"wrote golden fixture ({PARTICLE_COUNT} particles x {FRAME_COUNT} frames) to {OUT}")


if __name__ == "__main__":
    main()
