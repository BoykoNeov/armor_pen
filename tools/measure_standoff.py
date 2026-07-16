#!/usr/bin/env python3
"""Measure the shaped-charge STANDOFF effect from a cache (milestone 10).

Language-neutral helper (CLAUDE.md §3): reads caches per docs/CACHE_FORMAT.md and
imports nothing from the solver or the visualizer.

    python tools/measure_standoff.py caches/standoff_s00 caches/standoff_s90
    python tools/measure_standoff.py --family          # the shipped 4-deck family
    python tools/measure_standoff.py --convergence     # the 6-deck convergence study

WHAT IT MEASURES, AND WHY NOT THE OBVIOUS THING. A shaped-charge jet is seeded
velocity-graded, so each element flies at its own constant speed and the jet
extrapolates back to a VIRTUAL ORIGIN a distance

    Z0 = L * v_tip / (v_tip - v_tail)

behind the tip. The deck's standoff S adds to that: Z = Z0 + S. With v the velocity
of the element currently at the crater bottom, that element has flown Z + P, and the
crater deepens at u(v):

    v*t = Z + P(t),  dP/dt = u(v)  =>  dt/t = dv/(u(v) - v)  =>  t = t0 * G(v)
    =>  P = v*t - Z = Z * [ v*G(v)/V0 - 1 ]

**P is proportional to Z at matched v** — and the derivation never assumed ideal
hydrodynamics, so it holds for ANY u(v), Tate-with-strength included, and is
independent of jet diameter. See docs/PHYSICS.md §3.8.

Matched v == matched material element == matched CONSUMED FRACTION, which is what
this tool matches on: the fraction of jet particles that have latched `damage` is a
smooth aggregate, where a percentile of a Lagrangian label is noisy.

WHY NOT DEPTH AT A FIXED TIME. Depth at a fixed LAB time is an artifact of the
OPPOSITE SIGN — a longer standoff impacts later, so it penetrates for less of the
window and reads SHALLOWER. This tool reports that number too (`depth_end`), purely
so the trap is visible rather than inviting.

HOW IT AVOIDS BAKING IN ASSUMPTIONS:
  * The jet is identified as whatever is MOVING at frame 0 — not by material id.
  * The target face and back face are read from frame 0, before anything moves.
  * The crater is checked against the back face: a perforated target puts a CEILING
    on depth, which is the one quantity this study measures.
  * The penetration front is a high percentile, not the max, so one particle spat
    off the crater lip cannot define it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Penetration front as a percentile of jet x. Jet material at the crater bottom is
# actively being consumed, so this deliberately includes damaged particles:
# restricting to live ones would systematically lag the true front.
FRONT_PERCENTILE = 99.5

# Consumed fractions to match on. Kept below ~1/3 because the marker elements of
# interest sit in the leading third of the jet, and because the trailing jet never
# arrives in an affordable window (PHYSICS §3.4).
MATCH_FRACTIONS = (0.15, 0.20, 0.25, 0.30)


def load(cache_dir: Path):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("dtype") != "float32":
        raise SystemExit(f"unsupported dtype {manifest.get('dtype')!r}; expected float32")
    pc, fc = manifest["particle_count"], manifest["frame_count"]
    stride = len(manifest["attributes"])
    frames = np.memmap(
        cache_dir / "frames.bin", dtype="<f4", mode="r", shape=(fc, pc, stride)
    )
    return manifest, frames


def measure(cache_dir: Path) -> dict:
    manifest, frames = load(cache_dir)
    col = {a: i for i, a in enumerate(manifest["attributes"])}
    for need in ("pos_x", "vel_mag", "damage"):
        if need not in col:
            raise SystemExit(f"cache lacks required attribute {need!r}")

    f0 = np.asarray(frames[0])
    jet = f0[:, col["vel_mag"]] > 1.0          # armor is seeded at rest
    if not jet.any() or jet.all():
        raise SystemExit("need both a moving jet and a static target at frame 0")
    face_x = float(f0[~jet, col["pos_x"]].min())
    back_x = float(f0[~jet, col["pos_x"]].max())
    n_jet = int(jet.sum())

    fc = manifest["frame_count"]
    depth = np.full(fc, np.nan)
    consumed = np.zeros(fc)
    for f in range(fc):
        fr = np.asarray(frames[f])
        depth[f] = np.percentile(fr[jet, col["pos_x"]], FRONT_PERCENTILE) - face_x
        consumed[f] = float((fr[jet, col["damage"]] >= 0.5).sum()) / n_jet

    deepest = float(np.nanmax(depth))
    thickness = back_x - face_x
    return {
        "cache": cache_dir.name,
        "n_jet": n_jet,
        "face_x": face_x,
        "thickness": thickness,
        "depth": depth,
        "consumed": consumed,
        "depth_end": float(depth[-1]),
        "deepest": deepest,
        "perforated": bool(deepest >= thickness),
    }


def depth_at(r: dict, frac: float) -> float:
    """Depth interpolated at a matched consumed fraction."""
    if r["consumed"].max() < frac:
        return float("nan")
    return float(np.interp(frac, r["consumed"], r["depth"]))


def compare(a: dict, b: dict) -> list[float]:
    """Depth ratio b/a at each matched consumed fraction."""
    out = []
    for f in MATCH_FRACTIONS:
        da, db = depth_at(a, f), depth_at(b, f)
        out.append(db / da if np.isfinite(da) and np.isfinite(db) and da > 0 else np.nan)
    return out


def _report(r: dict) -> None:
    flag = "  *** PERFORATED — depth is CEILING-LIMITED, the measurement is void" if r["perforated"] else ""
    print(f"  {r['cache']:<28} target {r['thickness']:5.1f} mm   deepest {r['deepest']:6.1f} mm"
          f"   margin {r['thickness'] - r['deepest']:5.1f} mm{flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache_dirs", type=Path, nargs="*")
    ap.add_argument("--family", action="store_true", help="the shipped standoff_s* family")
    ap.add_argument("--convergence", action="store_true", help="the standoff_conv_* study")
    ap.add_argument("--caches", type=Path, default=Path("caches"))
    args = ap.parse_args(argv)

    if args.family:
        decks = [args.caches / f"standoff_s{s:02d}" for s in (0, 30, 60, 90)]
        rs = [measure(d) for d in decks]
        print("HALF-SPACE CHECK (a perforated target voids the measurement)")
        for r in rs:
            _report(r)
        print("\nDEPTH at matched consumed fraction (mm), and vs the S=0 baseline")
        print("  S    " + "".join(f"  f={f:.2f}" for f in MATCH_FRACTIONS) + "     ratio vs S=0   Z/Z0 predicted")
        for s, r in zip((0, 30, 60, 90), rs):
            ds = [depth_at(r, f) for f in MATCH_FRACTIONS]
            rat = np.nanmean(compare(rs[0], r))
            print(f"  {s:<4d} " + "".join(f" {d:6.1f}" for d in ds)
                  + f"       {rat:6.3f}         {(168.0 + s) / 168.0:6.3f}")
        print("\nTHE TRAP — depth at the end of the window (a FIXED LAB TIME):")
        print("  " + "   ".join(f"S={s}: {r['depth_end']:.1f} mm" for s, r in zip((0, 30, 60, 90), rs)))
        print("  It FALLS with standoff. That is not physics: a longer standoff impacts")
        print("  later and so penetrates for less of the window. Match on consumed")
        print("  fraction, never on lab time.")
        return 0

    if args.convergence:
        cfg = [
            ("3 mm jet, dx=0.375 (SHIPPED)", 8, "standoff_s00", "standoff_s90"),
            ("3 mm jet, dx=0.250", 12, "standoff_conv_dx250_s00", "standoff_conv_dx250_s90"),
            ("3 mm jet, dx=0.1875", 16, "standoff_conv_dx188_s00", "standoff_conv_dx188_s90"),
            ("6 mm jet, dx=0.375", 16, "standoff_conv_d6mm_s00", "standoff_conv_d6mm_s90"),
        ]
        print("CONVERGENCE: is the standoff shortfall numerical?")
        print("  The derivation is DIAMETER-INDEPENDENT, so every row predicts 1.536.")
        print("  Two independent routes to 16 cells across the jet: refine dx, or fatten the jet.\n")
        print("  configuration                  cells  " + "".join(f" f={f:.2f} " for f in MATCH_FRACTIONS) + "   mean")
        for name, cells, c0, c9 in cfg:
            a, b = measure(args.caches / c0), measure(args.caches / c9)
            rs = compare(a, b)
            print(f"  {name:<30} {cells:4d}  " + "".join(f"{r:7.4f} " for r in rs)
                  + f"  {np.nanmean(rs):6.4f}")
        print("\n  a-priori prediction (nothing fitted)                                 1.5357")
        print("\n  The shipped row is the WORST. It under-reads the effect ~2.3x on the")
        print("  excess. Report this as a monotone trend toward the prediction — NOT as a")
        print("  Richardson extrapolation, whose observed order here is ill-conditioned.")
        return 0

    if len(args.cache_dirs) != 2:
        ap.error("give two cache dirs (baseline first), or --family / --convergence")
    a, b = (measure(d) for d in args.cache_dirs)
    for r in (a, b):
        _report(r)
    rs = compare(a, b)
    print("\n  depth ratio at matched consumed fraction: "
          + "  ".join(f"f={f:.2f}: {r:.4f}" for f, r in zip(MATCH_FRACTIONS, rs)))
    print(f"  mean = {np.nanmean(rs):.4f}")
    return 1 if (a["perforated"] or b["perforated"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
