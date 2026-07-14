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

SCHEMA_VERSION = 1
ATTRIBUTES = ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id"]
FRAME_COUNT = 3
DOMAIN = {"xmin": 0.0, "xmax": 200.0, "ymin": 0.0, "ymax": 100.0}
MATERIALS = {"0": "tungsten_rod", "1": "rha"}

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
    else:  # RHA plate column at x=120, taking a little damage over frames
        q = p - ROD
        x = 120.0
        y = 20.0 + q * 5.0
        vel = 0.0
        stress = 50.0 * frame + 5.0 * q
        damage = min(1.0, 0.15 * frame * math.exp(-abs(q - 6) / 3.0))
        mat = 1
    return [x, y, vel, stress, damage, float(mat)]


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
    }
    with (OUT / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"wrote golden fixture ({PARTICLE_COUNT} particles x {FRAME_COUNT} frames) to {OUT}")


if __name__ == "__main__":
    main()
