"""CLI entry point: scenario.yaml -> cache dir.

    python -m ballistics_solver.run scenarios/apfsds_vs_rha.yaml --out ../caches/apfsds_vs_rha

STATUS: the plumbing (parse args, load scenario, set up the cache writer) is
real; the ``mpm.bake`` step is a stub (see mpm.py). This file wires the pieces
together so the intended flow is legible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import materials
from .config import load_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ballistics-bake",
        description="Bake a terminal-ballistics scenario to an on-disk cache.",
    )
    parser.add_argument("scenario", type=Path, help="path to a scenario YAML deck")
    parser.add_argument("--out", type=Path, required=True, help="output cache directory")
    parser.add_argument(
        "--cpu", action="store_true",
        help="allow the CPU backend (skips the GPU assert; slow, for debugging only)",
    )
    args = parser.parse_args(argv)

    device = "cpu" if args.cpu else "cuda:0"

    scenario = load_scenario(args.scenario)
    print(f"loaded scenario {scenario.name!r}: "
          f"{len(scenario.armor)} armor layer(s), "
          f"{scenario.solver.frame_count} frames requested")

    # Deferred imports so `--help` and scenario parsing work without Warp.
    from .cache_writer import CacheWriter
    from . import mpm

    # GPU sanity check first — a silent CPU fallback is a real hazard (§11).
    import warp as wp
    wp.init()
    if not args.cpu:
        mpm.assert_gpu(device)

    writer = CacheWriter(
        args.out,
        scenario=scenario.name,
        particle_count=0,  # set by mpm.bake once particles are seeded (§ below)
        attributes=["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id"],
        frame_dt=scenario.solver.total_time / scenario.solver.frame_count,
        domain={
            "xmin": scenario.domain.xmin, "xmax": scenario.domain.xmax,
            "ymin": scenario.domain.ymin, "ymax": scenario.domain.ymax,
        },
        units="mm-ms-g (see docs/PHYSICS.md)",
        materials=materials.id_to_name(),
    )

    with writer:
        mpm.bake(scenario, writer, device=device)  # seeds particles, sets writer.particle_count

    print(f"wrote cache to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
