"""Cache writer — the solver's implementation of docs/CACHE_FORMAT.md.

Writes ``manifest.json`` + ``frames.bin`` (single-blob, v1). This is pure
format serialization: no physics, no GPU, no knowledge of the visualizer. If
this file and CACHE_FORMAT.md ever disagree, the spec wins (CLAUDE.md §9).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import CACHE_SCHEMA_VERSION


def build_manifest(
    *,
    scenario: str,
    particle_count: int,
    frame_count: int,
    attributes: list[str],
    frame_dt: float,
    domain: dict[str, float],
    units: str,
    materials: dict[str, str],
    projectile: dict,
    armor: list[dict],
    material_descriptions: dict[str, str],
) -> dict:
    """Assemble a manifest dict per CACHE_FORMAT §2. The ONE definition of shape.

    Both paths that emit a manifest go through here — :meth:`CacheWriter.close`
    after a bake, and ``run.py --remanifest`` when migrating an existing cache in
    place (CACHE_FORMAT §2.2). That is deliberate: two functions building the same
    JSON is how the migrated caches and the freshly-baked ones would quietly come
    to disagree about the format they both claim to implement.
    """
    if str(sorted(materials)) != str(sorted(material_descriptions)):
        # CACHE_FORMAT §2.1 requires the two maps to share a key set, and a
        # validator catching this after the fact is worse than never emitting it:
        # `materials.id_to_name`/`id_to_description` build both from one LIBRARY,
        # so a mismatch here means a caller assembled them by hand and got it wrong.
        raise ValueError(
            f"materials and material_descriptions must share a key set "
            f"(CACHE_FORMAT §2.1); got {sorted(materials)} vs "
            f"{sorted(material_descriptions)}"
        )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "scenario": scenario,
        "particle_count": particle_count,
        "frame_count": frame_count,
        "attributes": list(attributes),
        "dtype": "float32",
        "frame_dt": frame_dt,
        "domain": domain,
        "units": units,
        "materials": materials,
        # The v3 scenario block — provenance, not data (CACHE_FORMAT §2.1). It says
        # what was SEEDED; nothing here is a result, and nothing may be measured
        # from it.
        "projectile": projectile,
        "armor": armor,
        "material_descriptions": material_descriptions,
    }


def write_manifest(out_dir: str | Path, manifest: dict) -> None:
    """Serialize a manifest dict to ``<out_dir>/manifest.json``.

    Separate from :class:`CacheWriter` on purpose, and the reason is a real
    hazard: ``CacheWriter.__init__`` opens ``frames.bin`` with mode ``"wb"``, so
    merely CONSTRUCTING one over an existing cache truncates the blob to zero
    bytes before a single frame is written. The migration path (§2.2) must touch
    manifest.json and nothing else, so it needs a door that is not the writer.
    """
    with (Path(out_dir) / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


class CacheWriter:
    """Accumulate frames and emit a spec-compliant cache directory.

    Parameters mirror the manifest fields in CACHE_FORMAT §2. Frames are
    appended one at a time as ``(particle_count, len(attributes))`` float32
    arrays and streamed to ``frames.bin`` in frame-major order (§3).
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        scenario: str,
        particle_count: int,
        attributes: list[str],
        frame_dt: float,
        domain: dict[str, float],
        units: str,
        materials: dict[str, str],
        projectile: dict,
        armor: list[dict],
        material_descriptions: dict[str, str],
    ) -> None:
        if "pos_x" not in attributes or "pos_y" not in attributes:
            raise ValueError("attributes must include pos_x and pos_y (CACHE_FORMAT §2)")
        if len(set(attributes)) != len(attributes):
            raise ValueError("attributes must not contain duplicates")

        self.out_dir = Path(out_dir)
        self.scenario = scenario
        self.particle_count = particle_count
        self.attributes = list(attributes)
        self.frame_dt = frame_dt
        self.domain = domain
        self.units = units
        self.materials = materials
        self.projectile = projectile
        self.armor = armor
        self.material_descriptions = material_descriptions

        self._frames_written = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._blob = (self.out_dir / "frames.bin").open("wb")

    def write_frame(self, frame: np.ndarray) -> None:
        """Append one render frame: shape (particle_count, len(attributes))."""
        expected = (self.particle_count, len(self.attributes))
        if frame.shape != expected:
            raise ValueError(f"frame shape {frame.shape} != expected {expected}")
        # Little-endian float32, row-major over (particle, attribute) — §3.
        self._blob.write(np.ascontiguousarray(frame, dtype="<f4").tobytes())
        self._frames_written += 1

    def close(self) -> None:
        """Flush frames.bin and write the authoritative manifest.json."""
        self._blob.close()
        manifest = build_manifest(
            scenario=self.scenario,
            particle_count=self.particle_count,
            frame_count=self._frames_written,
            attributes=self.attributes,
            frame_dt=self.frame_dt,
            domain=self.domain,
            units=self.units,
            materials=self.materials,
            projectile=self.projectile,
            armor=self.armor,
            material_descriptions=self.material_descriptions,
        )
        write_manifest(self.out_dir, manifest)

    def __enter__(self) -> "CacheWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
