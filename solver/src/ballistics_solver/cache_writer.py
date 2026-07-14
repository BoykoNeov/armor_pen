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
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "scenario": self.scenario,
            "particle_count": self.particle_count,
            "frame_count": self._frames_written,
            "attributes": self.attributes,
            "dtype": "float32",
            "frame_dt": self.frame_dt,
            "domain": self.domain,
            "units": self.units,
            "materials": self.materials,
        }
        with (self.out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")

    def __enter__(self) -> "CacheWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
