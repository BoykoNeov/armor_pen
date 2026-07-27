#!/usr/bin/env python3
"""Measure the shaped-charge STANDOFF effect from a cache (milestone 10).

Language-neutral helper (CLAUDE.md §3): reads caches per docs/CACHE_FORMAT.md and
imports nothing from the solver or the visualizer.

    python tools/measure_standoff.py caches/standoff_s00 caches/standoff_s90
    python tools/measure_standoff.py --family          # the shipped 4-deck family
    python tools/measure_standoff.py --convergence     # the 6-deck convergence study
    python tools/measure_standoff.py --dt-decomposition  # milestone 17: dx vs dt

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


def cells_across_jet(f0: np.ndarray, jet: np.ndarray, col: dict) -> float:
    """Cells across the jet, MEASURED from the seeded lattice at frame 0.

    `cells across the jet` is the controlling parameter of the whole §3.8 study, and
    it was previously carried as a hand-computed label (diameter / dx). It does not
    have to be: the seed is a lattice, `particles_per_cell=4` puts two particles per
    cell per axis, so counting distinct y-rows in the jet and halving reads it off
    the cache directly.

    Deliberately NOT `projectile.diameter / (domain / grid_resolution)`: `diameter`
    is provenance rather than data (CACHE_FORMAT §2.1), the manifest carries no
    `grid_resolution` at all, and `_fill_rect` rounds each object's lattice to fit
    it — so the arithmetic can disagree with the lattice actually seeded (PHYSICS
    §3.13 measured 15 rows = 7.5 cells where the domain implied 7.68).
    """
    ys = np.unique(np.round(f0[jet, col["pos_y"]].astype(np.float64), 4))
    return len(ys) / 2.0


def measure(cache_dir: Path, stride: int = 1) -> dict:
    """Read a cache into a depth-vs-consumed-fraction curve.

    `stride` subsamples FRAMES, and exists only so arms baked at different frame
    cadences can be compared on the same one (milestone 17: the shipped standoff
    decks dump 225 frames, the `standoff_conv_*` arms 75, i.e. every 3rd shipped
    frame time). It changes the interpolation resolution and nothing physical —
    measured at -0.113 % on the shipped S90/S0 ratio for 225 -> 75, two orders below
    the effects being decomposed. Default 1, so `--family` and `--convergence` are
    untouched.
    """
    manifest, frames = load(cache_dir)
    col = {a: i for i, a in enumerate(manifest["attributes"])}
    for need in ("pos_x", "pos_y", "vel_mag", "damage"):
        if need not in col:
            raise SystemExit(f"cache lacks required attribute {need!r}")

    f0 = np.asarray(frames[0])
    jet = f0[:, col["vel_mag"]] > 1.0          # armor is seeded at rest
    if not jet.any() or jet.all():
        raise SystemExit("need both a moving jet and a static target at frame 0")
    face_x = float(f0[~jet, col["pos_x"]].min())
    back_x = float(f0[~jet, col["pos_x"]].max())
    n_jet = int(jet.sum())

    idx = range(0, manifest["frame_count"], stride)
    depth = np.full(len(idx), np.nan)
    consumed = np.zeros(len(idx))
    for i, f in enumerate(idx):
        fr = np.asarray(frames[f])
        depth[i] = np.percentile(fr[jet, col["pos_x"]], FRONT_PERCENTILE) - face_x
        consumed[i] = float((fr[jet, col["damage"]] >= 0.5).sum()) / n_jet

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
        "cells": cells_across_jet(f0, jet, col),
        "frames_used": len(depth),
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


# --- milestone 17: separating dx from the dt that rides along with it ----------
#
# (label, s00 cache, s90 cache, substeps/frame, frame stride)
#
# `substeps` is a LABEL, not a measurement. No cache carries `dt`, `grid_resolution`
# or a substep count (CACHE_FORMAT §2 records `frame_dt` only), and this tool imports
# nothing from the solver (CLAUDE.md §3). The claim those labels encode — that each
# `dt` partner runs a substep BIT-IDENTICAL to its `dx` arm's, so the pair differs in
# `dx` alone — is pinned where it can be checked against the real sizing path, in
# `solver/tests/test_standoff_dt.py`. Do not read them as data.
#
# `cells across the jet` is NOT a label: it is measured off the seeded lattice
# (`cells_across_jet`), which is what makes the two 16-cell rows comparable at all.
#
# The stride puts every arm on the same 6.0e-7 s frame cadence; see `measure`.
DT_ARMS = [
    ("shipped        3 mm jet, dx=0.3750", "standoff_s00", "standoff_s90", 114, 3),
    ("dt partner     3 mm jet, dx=0.3750", "standoff_conv_dt513_s00", "standoff_conv_dt513_s90", 513, 1),
    ("dt partner     3 mm jet, dx=0.3750", "standoff_conv_dt684_s00", "standoff_conv_dt684_s90", 684, 1),
    ("dx arm         3 mm jet, dx=0.2500", "standoff_conv_dx250_s00", "standoff_conv_dx250_s90", 513, 1),
    ("dx arm         3 mm jet, dx=0.1875", "standoff_conv_dx188_s00", "standoff_conv_dx188_s90", 684, 1),
    ("fat jet        6 mm jet, dx=0.3750", "standoff_conv_d6mm_s00", "standoff_conv_d6mm_s90", 114, 1),
]

# Below this, a difference is not resolvable and must not be read as an effect: the
# repo's run-to-run scatter floor on an aggregate is 0.11 % (PHYSICS §3.2), and the
# 225 -> 75 frame decimation this mode applies to the shipped arm moves the ratio
# -0.113 %. Two independent reasons for the same number.
DT_RESOLUTION_PCT = 0.2


def _dt_decomposition(caches: Path) -> int:
    print("DT-DECOMPOSITION: is §3.8's two-route disagreement a TIMESTEP effect?")
    print("  §3.8 reaches 16 cells across the jet two ways and they disagree. §3.13")
    print("  predicted why: `dt` is CFL-bound, so refining `dx` refines the CLOCK with")
    print("  it, while fattening the jet does not touch `dt` at all. Each `dx` arm here")
    print("  has a substep-matched `dt` PARTNER at the SHIPPED `dx`, so that pair")
    print("  differs in `dx` alone and its difference is MEASURED, not inferred.\n")

    arms = {}
    print("  half-space check (a perforated baseline would INFLATE its row's ratio)")
    for label, c0, c9, sub, stride in DT_ARMS:
        a, b = measure(caches / c0, stride), measure(caches / c9, stride)
        arms[(label, sub)] = (a, b)
        for r in (a, b):
            _report(r)
    print()

    hdr = "".join(f" f={f:.2f} " for f in MATCH_FRACTIONS)
    print(f"  {'arm':<36} {'cells':>5} {'sub':>4}  {hdr}   mean")
    means = {}
    for label, c0, c9, sub, stride in DT_ARMS:
        a, b = arms[(label, sub)]
        rs = compare(a, b)
        means[(label, sub)] = float(np.nanmean(rs))
        print(f"  {label:<36} {a['cells']:5.1f} {sub:4d}  "
              + "".join(f"{r:7.4f} " for r in rs) + f"  {means[(label, sub)]:6.4f}")
    print("\n  a-priori prediction (nothing fitted)                                        1.5357")

    # The ratio can hide a dt effect that moves BOTH depths — a different finding
    # from "no dt effect", and §3.13's falsifier is written on the ratio alone.
    print("\n  DEPTHS at matched consumed fraction (mm) — the ratio can hide a common-mode")
    print("  move, so both arms are reported rather than only their quotient.")
    print(f"  {'arm':<36} {'sub':>4}  {'S=0: ' + hdr:<40}{'S=90: ' + hdr}")
    for label, c0, c9, sub, stride in DT_ARMS:
        a, b = arms[(label, sub)]
        d0 = "".join(f"{depth_at(a, f):6.2f} " for f in MATCH_FRACTIONS)
        d9 = "".join(f"{depth_at(b, f):6.2f} " for f in MATCH_FRACTIONS)
        print(f"  {label:<36} {sub:4d}  {d0:<40}{d9}")

    ship = means[(DT_ARMS[0][0], 114)]
    p513, p684 = means[(DT_ARMS[1][0], 513)], means[(DT_ARMS[2][0], 684)]
    dx250, dx188 = means[(DT_ARMS[3][0], 513)], means[(DT_ARMS[4][0], 684)]
    d6mm = means[(DT_ARMS[5][0], 114)]

    def pct(new, old):
        return 100.0 * (new - old) / old

    print("\n  THE DECOMPOSITION (each row holds every variable but the one named)")
    print(f"    dt alone,  114 -> 513 substeps at dx=0.3750 : {ship:6.4f} -> {p513:6.4f}  ({pct(p513, ship):+6.2f} %)")
    print(f"    dt alone,  114 -> 684 substeps at dx=0.3750 : {ship:6.4f} -> {p684:6.4f}  ({pct(p684, ship):+6.2f} %)")
    print(f"    dx alone,  0.3750 -> 0.2500 at 513 substeps : {p513:6.4f} -> {dx250:6.4f}  ({pct(dx250, p513):+6.2f} %)")
    print(f"    dx alone,  0.3750 -> 0.1875 at 684 substeps : {p684:6.4f} -> {dx188:6.4f}  ({pct(dx188, p684):+6.2f} %)")
    print(f"    diameter,  3 mm -> 6 mm at 114 substeps     : {ship:6.4f} -> {d6mm:6.4f}  ({pct(d6mm, ship):+6.2f} %)")

    print("\n  §3.13's PREDICTION, against today's caches")
    gap = pct(d6mm, dx188)
    dt_term = pct(p684, ship)
    print(f"    the two 16-cell routes still disagree: dx188 {dx188:.4f} vs fat jet {d6mm:.4f}  ({gap:+.2f} %)")
    print(f"    the dt-only term at the gap's own 684 substeps                        ({dt_term:+.2f} %)")

    # Undo the dx arm's dt refinement and re-read the gap. This is the quantitative
    # half of the prediction, and it rests on an assumption the design CANNOT test:
    # that a dt term measured at 8 cells transfers to the 16-cell arm. That is
    # exactly the dx x dt interaction §3.13 named as out of reach, so the residual
    # below is an estimate with a known open term, not a decomposition that closes.
    corrected = dx188 / (1.0 + dt_term / 100.0)
    residual = pct(d6mm, corrected)
    share = 100.0 * (gap - residual) / gap
    print(f"    dx188 with that term undone: {dx188:.4f} -> {corrected:.4f}, residual gap  ({residual:+.2f} %)")
    print(f"    so the timestep accounts for {share:.0f} % of the disagreement, and {100 - share:.0f} % is NOT dt.")
    print("    (that transfer assumes no dx x dt interaction — the one term this design")
    print("     cannot reach, so read the split as an estimate, not a closed account.)")
    if abs(dt_term) < DT_RESOLUTION_PCT:
        verdict = ("FALSIFIED — a dt-only refinement to the gap's own substep count does\n"
                   "      not move the ratio outside the ±%.1f %% resolution floor, so the\n"
                   "      disagreement is not a timestep effect." % DT_RESOLUTION_PCT)
    elif dt_term < 0:
        verdict = ("SUPPORTED IN SIGN — a finer dt alone moves the ratio DOWN, the\n"
                   "      direction §3.13 predicted the dx route is dragged. Whether it\n"
                   "      accounts for the gap is the magnitude line above.")
    else:
        verdict = ("FALSIFIED IN SIGN — a finer dt alone moves the ratio UP, the opposite\n"
                   "      of what §3.13 predicted. The dx route reading low is not this.")
    print(f"    verdict: {verdict}")
    print(f"\n  Resolution floor {DT_RESOLUTION_PCT} %: the repo's 0.11 % aggregate scatter, and the")
    print("  -0.113 % this mode's 225 -> 75 decimation costs the shipped arm. Nothing")
    print("  smaller than that is an effect.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache_dirs", type=Path, nargs="*")
    ap.add_argument("--family", action="store_true", help="the shipped standoff_s* family")
    ap.add_argument("--convergence", action="store_true", help="the standoff_conv_* study")
    ap.add_argument("--dt-decomposition", action="store_true",
                    help="milestone 17: separate dx from the dt CFL drags along with it")
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
        # The half-space premise is checked HERE too, not only in --family. The 6 mm
        # jet carries 2x the mass of the 3 mm one, so it is the row most able to
        # perforate — and it is also the load-bearing independent confirmation. A
        # ceiling-capped S=0 depth would INFLATE the ratio, i.e. flatter a row that
        # exists to be believed.
        measured = {}
        print("  half-space check (a perforated baseline would INFLATE its row's ratio)")
        for name, cells, c0, c9 in cfg:
            a, b = measure(args.caches / c0), measure(args.caches / c9)
            measured[name] = (a, b)
            for r in (a, b):
                _report(r)
        print()
        print("  configuration                  cells  " + "".join(f" f={f:.2f} " for f in MATCH_FRACTIONS) + "   mean")
        for name, cells, c0, c9 in cfg:
            a, b = measured[name]
            rs = compare(a, b)
            print(f"  {name:<30} {cells:4d}  " + "".join(f"{r:7.4f} " for r in rs)
                  + f"  {np.nanmean(rs):6.4f}")
        print("\n  a-priori prediction (nothing fitted)                                 1.5357")
        # COMPUTED, not typed. This line used to carry a hardcoded "~2.3x" measured
        # under Murnaghan; two rebakes (M13's EOS, M14's CFL margin) moved every row
        # under it and the prose did not follow. A figure that restates a measurement
        # in prose goes stale silently — so derive it from the row just printed.
        shipped = float(np.nanmean(compare(*measured[cfg[0][0]])))
        print(f"\n  The shipped row is the WORST. It under-reads the effect "
              f"~{(1.5357 - 1.0) / (shipped - 1.0):.1f}x on the excess "
              f"({1.5357 - 1.0:.3f} predicted vs {shipped - 1.0:.3f} measured).")
        print("  Report this as a monotone trend toward the prediction — NOT as a")
        print("  Richardson extrapolation, whose observed order here is ill-conditioned.")
        return 0

    if args.dt_decomposition:
        return _dt_decomposition(args.caches)

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
