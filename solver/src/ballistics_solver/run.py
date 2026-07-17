"""CLI entry point: scenario.yaml -> cache dir.

    python -m ballistics_solver.run scenarios/apfsds_vs_rha.yaml --out ../caches/apfsds_vs_rha

STATUS: the plumbing (parse args, load scenario, set up the cache writer) is
real; the ``mpm.bake`` step is a stub (see mpm.py). This file wires the pieces
together so the intended flow is legible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import materials
from .config import Scenario, load_scenario


def _scenario_block(scenario: Scenario) -> tuple[dict, list[dict]]:
    """Scenario -> the manifest's `projectile` / `armor` (CACHE_FORMAT §2.1).

    Lives here rather than in cache_writer.py to keep that module what its
    docstring claims: pure format serialization that takes primitives and knows
    nothing about the scenario schema. This is the one place the two meet.

    Everything below is an INPUT to the bake. Nothing here is a result, which is
    why §2.1 forbids measuring from it — `velocity` is the tip's speed at t=0, and
    the honest live one is the `vel_mag` column.
    """
    p = scenario.projectile
    projectile = {
        "kind": p.kind,
        "material": p.material,
        "length": p.length,
        "diameter": p.diameter,
        "velocity": p.velocity,
        # None -> JSON null, meaning "uniform". A number means a velocity-graded
        # projectile, which is the whole difference between a rod and a jet.
        "tail_velocity": p.tail_velocity,
        "angle_deg": p.angle_deg,
        "nose_shape": p.nose_shape,
    }
    armor = [
        {"material": a.material, "thickness": a.thickness, "standoff": a.standoff}
        for a in scenario.armor
    ]
    return projectile, armor


def _remanifest(scenario: Scenario, cache_dir: Path) -> int:
    """Rewrite an existing cache's manifest.json in place — CACHE_FORMAT §2.2.

    v3 added no column and moved no byte of frames.bin, so re-running the physics
    to migrate would re-roll every documented figure in this repo within its
    ~0.11 % scatter (MPM `atomic_add` ordering is not deterministic) to change a
    text label. This writes the manifest and touches nothing else.

    THE DRIFT GUARD IS THE POINT. A deck can change after its bake, and stamping
    today's deck onto last week's cache yields a manifest that confidently
    describes the wrong scenario — a defect no validator downstream could see,
    because the result is structurally perfect. So this refuses unless the deck
    still agrees with the bake, and a refusal means that deck needs a real rebake.
    """
    from .cache_writer import build_manifest, write_manifest

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"error: no manifest.json in {cache_dir}", file=sys.stderr)
        return 1
    old = json.loads(manifest_path.read_text(encoding="utf-8"))

    # A v1 cache predates the `internal_energy` column, so its attribute list
    # cannot be carried into a v3 manifest — that one needs a rebake, not a label.
    if old.get("schema_version") not in (2, 3):
        print(f"error: {cache_dir.name}: cannot migrate schema_version "
              f"{old.get('schema_version')!r}; rebake it", file=sys.stderr)
        return 1

    if old.get("scenario") != scenario.name:
        print(f"error: {cache_dir.name}: deck is {scenario.name!r} but the bake is "
              f"{old.get('scenario')!r} — wrong deck for this cache", file=sys.stderr)
        return 1

    keys = ("xmin", "xmax", "ymin", "ymax")
    dom_old = {k: float(old.get("domain", {}).get(k, float("nan"))) for k in keys}
    dom_deck = {
        "xmin": float(scenario.domain.xmin), "xmax": float(scenario.domain.xmax),
        "ymin": float(scenario.domain.ymin), "ymax": float(scenario.domain.ymax),
    }
    if dom_old != dom_deck:
        print(f"error: {cache_dir.name}: deck domain {dom_deck} != baked domain "
              f"{dom_old} — the deck changed since this bake; rebake it",
              file=sys.stderr)
        return 1

    # Every id the BAKE named must still mean the same material. A library that
    # renamed or re-numbered an id would otherwise get a description pinned to it
    # that describes something else. Ids added to the library since the bake are
    # correctly ignored: this manifest describes the ids it names, and §2.1 already
    # allows the map to carry ids the data never uses.
    live_names = {str(m.material_id): m.name for m in materials.LIBRARY.values()}
    baked = old.get("materials", {})
    drifted = {i: n for i, n in baked.items() if live_names.get(i) != n}
    if drifted:
        print(f"error: {cache_dir.name}: material ids {drifted} no longer match the "
              f"library — rebake it", file=sys.stderr)
        return 1

    projectile, armor = _scenario_block(scenario)
    new = build_manifest(
        # Layout and physics carried VERBATIM from the bake, so they cannot drift.
        # Only the descriptive fields below come from the deck.
        scenario=old["scenario"],
        particle_count=old["particle_count"],
        frame_count=old["frame_count"],
        attributes=old["attributes"],
        frame_dt=old["frame_dt"],
        domain=old["domain"],
        units=old["units"],
        materials=baked,
        projectile=projectile,
        armor=armor,
        material_descriptions={
            i: materials.get(n).description for i, n in baked.items()
        },
    )
    write_manifest(cache_dir, new)
    print(f"remanifested {cache_dir} (v{old['schema_version']} -> "
          f"v{new['schema_version']}; frames.bin untouched)")
    return 0


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
    parser.add_argument(
        "--remanifest", action="store_true",
        help="do NOT bake: rewrite an existing cache's manifest.json in place and "
             "leave frames.bin untouched (CACHE_FORMAT §2.2). Refuses if the deck "
             "has drifted from the bake. No GPU needed.",
    )
    args = parser.parse_args(argv)

    device = "cpu" if args.cpu else "cuda:0"

    scenario = load_scenario(args.scenario)
    print(f"loaded scenario {scenario.name!r}: "
          f"{len(scenario.armor)} armor layer(s), "
          f"{scenario.solver.frame_count} frames requested")

    # Before the Warp import and the GPU assert: migrating a manifest is pure
    # bookkeeping and must not need a GPU to run.
    if args.remanifest:
        return _remanifest(scenario, args.out)

    # Deferred imports so `--help` and scenario parsing work without Warp.
    from .cache_writer import CacheWriter
    from . import mpm

    # GPU sanity check first — a silent CPU fallback is a real hazard (§11).
    import warp as wp
    wp.init()
    if not args.cpu:
        mpm.assert_gpu(device)

    projectile, armor = _scenario_block(scenario)
    writer = CacheWriter(
        args.out,
        scenario=scenario.name,
        particle_count=0,  # set by mpm.bake once particles are seeded (§ below)
        attributes=["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id",
                    "internal_energy"],
        frame_dt=scenario.solver.total_time / scenario.solver.frame_count,
        domain={
            "xmin": scenario.domain.xmin, "xmax": scenario.domain.xmax,
            "ymin": scenario.domain.ymin, "ymax": scenario.domain.ymax,
        },
        units="mm-ms-g (see docs/PHYSICS.md)",
        materials=materials.id_to_name(),
        projectile=projectile,
        armor=armor,
        material_descriptions=materials.id_to_description(),
    )

    with writer:
        mpm.bake(scenario, writer, device=device)  # seeds particles, sets writer.particle_count

    print(f"wrote cache to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
