#!/usr/bin/env python3
"""Validate a cache directory against docs/CACHE_FORMAT.md.

Language-neutral helper: depends on neither the solver nor the visualizer
(CLAUDE.md §3). Uses only the Python standard library so it runs anywhere.

    python tools/validate_cache.py caches/apfsds_vs_rha

Exit code 0 = valid, 1 = invalid, 2 = usage error. Implements the §6 checklist.
"""

from __future__ import annotations

import array
import json
import math
import struct
import sys
from pathlib import Path

# v2 (milestone 13) appended the `internal_energy` column. v1 is deliberately NOT
# kept: every cache the solver emits is rebaked to v2, so accepting v1 would only
# let a stale pre-milestone-13 cache validate clean and look current.
SUPPORTED_SCHEMA_VERSIONS = {2}
REQUIRED_FIELDS = {
    "schema_version": int,
    "scenario": str,
    "particle_count": int,
    "frame_count": int,
    "attributes": list,
    "dtype": str,
    "frame_dt": (int, float),
    "domain": dict,
    "units": str,
    "materials": dict,
}


class CacheInvalid(Exception):
    """Raised when a cache violates the format contract."""


def _check_manifest(manifest: dict) -> None:
    for field, typ in REQUIRED_FIELDS.items():
        if field not in manifest:
            raise CacheInvalid(f"manifest missing required field {field!r}")
        if not isinstance(manifest[field], typ):
            raise CacheInvalid(
                f"manifest field {field!r} has type {type(manifest[field]).__name__}, "
                f"expected {typ}"
            )

    if manifest["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        raise CacheInvalid(
            f"schema_version {manifest['schema_version']} not understood "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )
    if manifest["dtype"] != "float32":
        raise CacheInvalid(f"dtype must be 'float32', got {manifest['dtype']!r}")

    attrs = manifest["attributes"]
    if not attrs:
        raise CacheInvalid("attributes must be non-empty")
    if len(set(attrs)) != len(attrs):
        raise CacheInvalid("attributes must not contain duplicates")
    for required in ("pos_x", "pos_y"):
        if required not in attrs:
            raise CacheInvalid(f"attributes must include {required!r}")

    for n in ("particle_count", "frame_count"):
        if manifest[n] <= 0:
            raise CacheInvalid(f"{n} must be positive, got {manifest[n]}")
    if manifest["frame_dt"] <= 0:
        raise CacheInvalid("frame_dt must be positive")

    dom = manifest["domain"]
    for key in ("xmin", "xmax", "ymin", "ymax"):
        if key not in dom:
            raise CacheInvalid(f"domain missing {key!r}")
    if not (dom["xmax"] > dom["xmin"] and dom["ymax"] > dom["ymin"]):
        raise CacheInvalid(f"degenerate domain: {dom}")


def _check_binary(cache_dir: Path, manifest: dict) -> None:
    if manifest.get("frame_layout", "single_blob") != "single_blob":
        # Per-frame layout is allowed by the spec (§4) but not checked here yet.
        print("  note: non-single-blob frame_layout; binary size check skipped")
        return

    blob = cache_dir / "frames.bin"
    if not blob.is_file():
        raise CacheInvalid("frames.bin not found")

    expected = (
        manifest["frame_count"]
        * manifest["particle_count"]
        * len(manifest["attributes"])
        * 4  # sizeof(float32)
    )
    actual = blob.stat().st_size
    if actual != expected:
        raise CacheInvalid(
            f"frames.bin is {actual} bytes, expected exactly {expected} "
            f"(= frame_count * particle_count * len(attributes) * 4)"
        )


def _check_material_ids(cache_dir: Path, manifest: dict) -> None:
    """Best-effort: every material_id in the data has a materials entry.

    Samples the first frame only, to stay cheap on huge caches (§6.7).
    """
    if "material_id" not in manifest["attributes"] or manifest.get("frame_layout"):
        return
    attrs = manifest["attributes"]
    stride = len(attrs)
    mat_col = attrs.index("material_id")
    pc = manifest["particle_count"]
    known = set(manifest["materials"].keys())

    with (cache_dir / "frames.bin").open("rb") as fh:
        frame0 = fh.read(pc * stride * 4)
    values = struct.unpack(f"<{pc * stride}f", frame0)
    seen = {str(int(round(values[p * stride + mat_col]))) for p in range(pc)}
    missing = seen - known
    if missing:
        raise CacheInvalid(f"material_id(s) {sorted(missing)} have no materials entry")


def _check_finite(cache_dir: Path, manifest: dict) -> None:
    """CACHE_FORMAT §6.8 — no NaN, no +/-Inf, in any column, in any frame.

    THE POINT, because it is not obvious: a diverged bake is STRUCTURALLY PERFECT.
    Blowing up changes the values and never the layout, so every other check in
    this file passes on it — the manifest is well-formed, the blob is exactly the
    right size, and the material_id check samples frame 0, which is still clean.
    A real 550-frame cache with 97 % of its particles carrying NaN velocity from
    frame 276 on validated OK against rules 1-7. This is the rule that noticed.

    Reports the FIRST bad frame and the column, because in a divergence "when"
    is the diagnostic — the frame index times frame_dt is where to look.

    Scans every frame rather than sampling: a full pass over the biggest cache
    here is seconds against a bake measured in minutes, and sampling would trade
    away the first-frame number for savings nobody needs.
    """
    if manifest.get("frame_layout", "single_blob") != "single_blob":
        print("  note: non-single-blob frame_layout; finiteness check skipped")
        return

    attrs = manifest["attributes"]
    stride = len(attrs)
    pc = manifest["particle_count"]
    frame_bytes = pc * stride * 4

    with (cache_dir / "frames.bin").open("rb") as fh:
        for f in range(manifest["frame_count"]):
            buf = fh.read(frame_bytes)

            # FAST PATH, and it is exact rather than heuristic. In IEEE-754
            # binary32 every NaN and +/-Inf has an all-ones exponent, so its most
            # significant byte (little-endian: index 3 of each word) is s|1111111
            # = 0x7F or 0xFF. Contrapositive: if no MSB in the frame is 0x7F or
            # 0xFF, NOTHING in it can be non-finite — proven, not sampled. Both
            # `buf[3::4]` and `in` run at C speed, so the common case (a clean
            # frame) costs one strided slice and two byte searches.
            #
            # The converse does not hold: a finite float with exponent 254
            # (|v| ~ 1.7e38..3.4e38) also has MSB 0x7F. That is a false POSITIVE,
            # which only costs a slow confirmation below — never a false negative.
            msb = buf[3::4]
            if 0x7F not in msb and 0xFF not in msb:
                continue

            values = array.array("f")
            values.frombytes(buf)
            if sys.byteorder == "big":
                values.byteswap()  # the format is little-endian, always (§3)
            bad: dict[str, int] = {}
            for i, name in enumerate(attrs):
                n = sum(1 for v in values[i::stride] if not math.isfinite(v))
                if n:
                    bad[name] = n
            if not bad:
                continue  # exponent-254 false positive: genuinely finite

            t_us = f * manifest["frame_dt"] * 1e6
            raise CacheInvalid(
                f"non-finite values (NaN/Inf) first appear at frame {f} "
                f"(t={t_us:.3f} us), in "
                + ", ".join(f"{k}={v}/{pc}" for k, v in bad.items())
                + " — the bake DIVERGED. The cache is structurally perfect and "
                  "physically meaningless; do not ship it."
            )


def validate(cache_dir: Path) -> None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise CacheInvalid(f"no manifest.json in {cache_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CacheInvalid(f"manifest.json is not valid JSON: {exc}") from exc

    _check_manifest(manifest)
    _check_binary(cache_dir, manifest)
    _check_material_ids(cache_dir, manifest)
    _check_finite(cache_dir, manifest)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    cache_dir = Path(argv[0])
    if not cache_dir.is_dir():
        print(f"error: {cache_dir} is not a directory", file=sys.stderr)
        return 2

    try:
        validate(cache_dir)
    except CacheInvalid as exc:
        print(f"INVALID  {cache_dir}\n  {exc}")
        return 1
    print(f"OK       {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
