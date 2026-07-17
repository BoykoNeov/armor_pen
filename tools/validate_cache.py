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

# v3 added the scenario block (§2.1). Older versions are deliberately NOT kept:
# every cache in the repo is migrated to v3, so accepting v2 would only let a
# stale cache validate clean and look current. (v2 dropped v1 for the same
# reason, and v1 for the same reason before that.)
SUPPORTED_SCHEMA_VERSIONS = {3}
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
    "projectile": dict,
    "armor": list,
    "material_descriptions": dict,
}

# §2.1: `projectile` key -> accepted JSON type(s). `tail_velocity` is the odd one
# — null means "uniform", which is what every KE deck is, so None is a legitimate
# VALUE here and not a missing field.
PROJECTILE_FIELDS = {
    "kind": str,
    "material": str,
    "length": (int, float),
    "diameter": (int, float),
    "velocity": (int, float),
    "tail_velocity": (int, float, type(None)),
    "angle_deg": (int, float),
    "nose_shape": str,
}
ARMOR_LAYER_FIELDS = {
    "material": str,
    "thickness": (int, float),
    "standoff": (int, float),
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

    _check_scenario_block(manifest)


def _check_scenario_block(manifest: dict) -> None:
    """Rule 9 — the v3 scenario block is present and well-formed (§2.1).

    Structural only, and that is the honest ceiling: this file depends on neither
    half (CLAUDE.md §3), so it cannot know that `tungsten_rod` is dense or that
    1600 m/s is what the deck said. It checks the shape and the one cross-field
    invariant it CAN see — that `material_descriptions` and `materials` agree on
    their key set. The deck-vs-bake agreement that would catch a stale
    description is guarded where the deck is in scope: `run.py --remanifest`.
    """
    proj = manifest["projectile"]
    for key, typ in PROJECTILE_FIELDS.items():
        if key not in proj:
            raise CacheInvalid(f"projectile missing required key {key!r} (§2.1)")
        if not isinstance(proj[key], typ):
            raise CacheInvalid(
                f"projectile[{key!r}] has type {type(proj[key]).__name__}, "
                f"expected {typ}"
            )
    # bool is a subclass of int in Python, so the isinstance checks above would
    # wave through `"velocity": true`. Nothing emits that, but a numeric field
    # holding a bool is precisely the kind of thing a validator exists to refuse.
    for key, typ in PROJECTILE_FIELDS.items():
        if typ is not str and isinstance(proj[key], bool):
            raise CacheInvalid(f"projectile[{key!r}] must be a number, not a bool")

    armor = manifest["armor"]
    if not armor:
        raise CacheInvalid("armor must be a non-empty array (§2.1)")
    for i, layer in enumerate(armor):
        if not isinstance(layer, dict):
            raise CacheInvalid(f"armor[{i}] must be an object, got {type(layer).__name__}")
        for key, typ in ARMOR_LAYER_FIELDS.items():
            if key not in layer:
                raise CacheInvalid(f"armor[{i}] missing required key {key!r} (§2.1)")
            if not isinstance(layer[key], typ) or (
                typ is not str and isinstance(layer[key], bool)
            ):
                raise CacheInvalid(
                    f"armor[{i}][{key!r}] has type {type(layer[key]).__name__}, "
                    f"expected {typ}"
                )

    # The one invariant worth catching: both maps are emitted together from one
    # LIBRARY, so they only fall out of step if something upstream is wrong.
    descs = manifest["material_descriptions"]
    if set(descs) != set(manifest["materials"]):
        missing = sorted(set(manifest["materials"]) - set(descs))
        extra = sorted(set(descs) - set(manifest["materials"]))
        raise CacheInvalid(
            f"material_descriptions must have the same key set as materials "
            f"(§2.1); missing {missing}, unexpected {extra}"
        )
    for key, value in descs.items():
        if not isinstance(value, str) or not value.strip():
            raise CacheInvalid(
                f"material_descriptions[{key!r}] must be a non-empty string"
            )


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
