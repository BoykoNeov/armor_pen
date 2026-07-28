#!/usr/bin/env python3
"""Measure the shaped-charge STANDOFF effect from a cache (milestone 10).

Language-neutral helper (CLAUDE.md §3): reads caches per docs/CACHE_FORMAT.md and
imports nothing from the solver or the visualizer.

    python tools/measure_standoff.py caches/standoff_s00 caches/standoff_s90
    python tools/measure_standoff.py --family          # the shipped 4-deck family
    python tools/measure_standoff.py --convergence     # the 6-deck convergence study
    python tools/measure_standoff.py --dt-decomposition  # milestone 17: dx vs dt
    python tools/measure_standoff.py --diameter-decomposition  # m18: diameter vs cells
    python tools/measure_standoff.py --route-difference  # m19: it vs cells across

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
from typing import NamedTuple

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


def route_difference(base: list[float], other: list[float]) -> list[float]:
    """Per-matched-fraction % difference between two arms' S90/S0 ratios.

    PER FRACTION, never a mean, and that is the point of it being a function.
    Milestone 16 found a decomposed cell (x=160) that swung 2.3 pp — more than the
    effect being read there — while its neighbours moved ≤0.15 pp, and milestone 18's
    scale row varies 3x across this window. A mean over a quantity that is not
    constant across the window hides its own unreliability: it reports one number
    with no hint that the number depends on where you stood.
    """
    return [100.0 * (b / a - 1.0) for a, b in zip(base, other)]


def _report(r: dict) -> None:
    flag = "  *** PERFORATED — depth is CEILING-LIMITED, the measurement is void" if r["perforated"] else ""
    print(f"  {r['cache']:<32} target {r['thickness']:5.1f} mm   deepest {r['deepest']:6.1f} mm"
          f"   margin {r['thickness'] - r['deepest']:5.1f} mm{flag}")


class Arm(NamedTuple):
    """One (S=0, S=90) cache pair, with a STABLE KEY to index it by.

    The key exists because milestone 18 documented a live hazard and milestone 19
    removed it: `_dt_decomposition` used to read `DT_ARMS[0] … DT_ARMS[5]`
    positionally, so inserting or sorting a row would silently re-key a PUBLISHED
    decomposition and stay green. Every lookup below now goes through `.key`.

    `substeps` is a LABEL, not a measurement, and it is not even the same convention
    in both tables — see the note over `DIAM_ARMS`. `dt_ms` is the quantity the
    pairing claims are actually about (two arms "share a substep" iff they share
    `dt`), and it is likewise a label: no cache carries `dt`, `grid_resolution` or a
    substep count (CACHE_FORMAT §2 records `frame_dt` only), and this tool imports
    nothing from the solver (CLAUDE.md §3). All of it is pinned against the real
    sizing path in `solver/tests/`. Do not read these fields as data.

    `cells across the jet` is NOT a label anywhere here: it is measured off the
    seeded lattice by `cells_across_jet`, which is what makes rows comparable at all.
    """

    key: str
    label: str
    s00: str
    s90: str
    substeps: int
    stride: int
    dt_ms: float


# --- milestone 17: separating dx from the dt that rides along with it ----------
#
# The claim these rows encode — that each `dt` partner runs a substep BIT-IDENTICAL
# to its `dx` arm's, so the pair differs in `dx` alone — is pinned where it can be
# checked against the real sizing path, in `solver/tests/test_standoff_dt.py`.
#
# The stride puts every arm on the same 6.0e-7 s frame cadence; see `measure`.
DT_ARMS = [
    Arm("shipped", "shipped        3 mm jet, dx=0.3750",
        "standoff_s00", "standoff_s90", 114, 3, 1.754386e-6),
    Arm("dt513", "dt partner     3 mm jet, dx=0.3750",
        "standoff_conv_dt513_s00", "standoff_conv_dt513_s90", 513, 1, 1.169591e-6),
    Arm("dt684", "dt partner     3 mm jet, dx=0.3750",
        "standoff_conv_dt684_s00", "standoff_conv_dt684_s90", 684, 1, 8.771930e-7),
    Arm("dx250", "dx arm         3 mm jet, dx=0.2500",
        "standoff_conv_dx250_s00", "standoff_conv_dx250_s90", 513, 1, 1.169591e-6),
    Arm("dx188", "dx arm         3 mm jet, dx=0.1875",
        "standoff_conv_dx188_s00", "standoff_conv_dx188_s90", 684, 1, 8.771930e-7),
    Arm("d6mm", "fat jet        6 mm jet, dx=0.3750",
        "standoff_conv_d6mm_s00", "standoff_conv_d6mm_s90", 114, 1, 1.754386e-6),
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
    for arm in DT_ARMS:
        a, b = measure(caches / arm.s00, arm.stride), measure(caches / arm.s90, arm.stride)
        arms[arm.key] = (a, b)
        for r in (a, b):
            _report(r)
    print()

    hdr = "".join(f" f={f:.2f} " for f in MATCH_FRACTIONS)
    print(f"  {'arm':<36} {'cells':>5} {'sub':>4}  {hdr}   mean")
    means = {}
    for arm in DT_ARMS:
        a, b = arms[arm.key]
        rs = compare(a, b)
        means[arm.key] = float(np.nanmean(rs))
        print(f"  {arm.label:<36} {a['cells']:5.1f} {arm.substeps:4d}  "
              + "".join(f"{r:7.4f} " for r in rs) + f"  {means[arm.key]:6.4f}")
    print("\n  a-priori prediction (nothing fitted)                                        1.5357")

    # The ratio can hide a dt effect that moves BOTH depths — a different finding
    # from "no dt effect", and §3.13's falsifier is written on the ratio alone.
    print("\n  DEPTHS at matched consumed fraction (mm) — the ratio can hide a common-mode")
    print("  move, so both arms are reported rather than only their quotient.")
    print(f"  {'arm':<36} {'sub':>4}  {'S=0: ' + hdr:<40}{'S=90: ' + hdr}")
    for arm in DT_ARMS:
        a, b = arms[arm.key]
        d0 = "".join(f"{depth_at(a, f):6.2f} " for f in MATCH_FRACTIONS)
        d9 = "".join(f"{depth_at(b, f):6.2f} " for f in MATCH_FRACTIONS)
        print(f"  {arm.label:<36} {arm.substeps:4d}  {d0:<40}{d9}")

    ship = means["shipped"]
    p513, p684 = means["dt513"], means["dt684"]
    dx250, dx188 = means["dx250"], means["dx188"]
    d6mm = means["d6mm"]

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


# --- milestone 18: separating the DIAMETER from the cells it buys ---------------
#
# A SEPARATE LIST FROM `DT_ARMS`, DELIBERATELY — and the reason is the SUBSTEP
# COLUMN, which is not `dt` and does not even use the same convention in the two
# tables. All three arms below run dt = 1.754386e-6 ms, bit-identically (verified
# through `mpm.plan_substeps`, not from a cache: CACHE_FORMAT §2 records `frame_dt`
# and nothing else). The shipped arm reads 114 only because it dumps 225 frames where
# the diagnostic arms dump 75 — a 3x finer frame cadence over the same window, which
# the `stride` column undoes. Substeps/frame x frame_dt is the invariant; neither
# factor alone is. `DT_ARMS` labels the same fat-jet cache 114 under the shipped
# cadence; merging the tables would force one convention onto a published table.
#
# Milestone 18 warned here that `_dt_decomposition` indexed its arms POSITIONALLY, so
# inserting a row would silently re-key the published §3.14 decomposition — green,
# and wrong. **Milestone 19 removed that hazard rather than deferring it again**:
# every arm carries a stable `.key` and every lookup goes through it, so these lists
# may now be sorted or extended freely.
DIAM_ARMS = [
    Arm("shipped", "shipped        3 mm jet, dx=0.3750",
        "standoff_s00", "standoff_s90", 114, 3, 1.754386e-6),
    Arm("fat", "fat jet        6 mm jet, dx=0.3750",
        "standoff_conv_d6mm_s00", "standoff_conv_d6mm_s90", 342, 1, 1.754386e-6),
    Arm("coarse_fat", "coarse fat     6 mm jet, dx=0.7500",
        "standoff_conv_d6mm_dx750_s00", "standoff_conv_d6mm_dx750_s90", 342, 1, 1.754386e-6),
]


def _mean_ratio(caches: Path, c0: str, c9: str, stride: int) -> float:
    a, b = measure(caches / c0, stride), measure(caches / c9, stride)
    return float(np.nanmean(compare(a, b)))


def _arm_ratio(caches: Path, arms: list[Arm], key: str) -> float:
    """Mean S90/S0 for the arm with this key. The ONE lookup path, by design."""
    by_key = {a.key: a for a in arms}
    arm = by_key[key]
    return _mean_ratio(caches, arm.s00, arm.s90, arm.stride)


def _dt_residual(caches: Path) -> tuple[float, float, float]:
    """Re-derive §3.14's (gap, dt term, residual) from the caches, by NAME.

    Not a literal. §3.8's table went two rebakes stale because a measurement was
    restated in prose, and `--convergence` already learned to compute its under-read
    factor rather than print it. The residual is the number milestones 18 and 19 test
    against, so it is the last one that should be typed in.
    """
    ship = _arm_ratio(caches, DT_ARMS, "shipped")
    p684 = _arm_ratio(caches, DT_ARMS, "dt684")
    dx188 = _arm_ratio(caches, DT_ARMS, "dx188")
    d6mm = _arm_ratio(caches, DT_ARMS, "d6mm")
    gap = 100.0 * (d6mm - dx188) / dx188
    dt_term = 100.0 * (p684 - ship) / ship
    corrected = dx188 / (1.0 + dt_term / 100.0)
    return gap, dt_term, 100.0 * (d6mm - corrected) / corrected


def _diameter_decomposition(caches: Path) -> int:
    print("DIAMETER-DECOMPOSITION: is §3.14's residual a finite-DIAMETER effect?")
    print("  §3.14 found the two 16-cell routes still disagree after the timestep is")
    print("  undone, and named §3.8's own hedge — 'a real finite-diameter effect,")
    print("  MOSTLY excluded' — as the leading candidate by elimination. No deck in")
    print("  that family can test it: the fat-jet route changes the CELL COUNT and the")
    print("  DIAMETER together. The third arm below unpicks them — a 6 mm jet at")
    print("  dx=0.75, so it carries the fat arm's diameter at the shipped arm's 8")
    print("  cells, with the deck `dt` pinned to both.\n")

    arms, means, per_f = {}, {}, {}
    print("  half-space check (a perforated baseline would INFLATE its row's ratio)")
    for arm in DIAM_ARMS:
        a, b = measure(caches / arm.s00, arm.stride), measure(caches / arm.s90, arm.stride)
        arms[arm.key] = (a, b)
        for r in (a, b):
            _report(r)
    print()

    hdr = "".join(f" f={f:.2f} " for f in MATCH_FRACTIONS)
    print(f"  {'arm':<36} {'cells':>5} {'sub':>4}  {hdr}   mean")
    for arm in DIAM_ARMS:
        a, b = arms[arm.key]
        rs = compare(a, b)
        per_f[arm.key] = rs
        means[arm.key] = float(np.nanmean(rs))
        print(f"  {arm.label:<36} {a['cells']:5.1f} {arm.substeps:4d}  "
              + "".join(f"{r:7.4f} " for r in rs) + f"  {means[arm.key]:6.4f}")
    print("\n  a-priori prediction (nothing fitted)                                        1.5357")
    print("  (the shipped row reads 1.2657 in --convergence and §3.8's table: that mode")
    print("   does not decimate 225 -> 75 frames. The -0.113 % is the documented cadence")
    print("   cost, NOT a stale table — both are current, and the effects below are 20x it.)")

    # §3.14's lesson, applied rather than restated: a ratio is ~3x less dt-sensitive
    # than the depths under it, so a quotient-only instrument can report "no effect"
    # for a solver that moved everything it measures. Same exposure here for dx.
    print("\n  DEPTHS at matched consumed fraction (mm) — the ratio is a partial")
    print("  common-mode cancellation, so both arms are reported (§3.14).")
    print(f"  {'arm':<36} {'sub':>4}  {'S=0: ' + hdr:<40}{'S=90: ' + hdr}")
    for arm in DIAM_ARMS:
        a, b = arms[arm.key]
        d0 = "".join(f"{depth_at(a, f):6.2f} " for f in MATCH_FRACTIONS)
        d9 = "".join(f"{depth_at(b, f):6.2f} " for f in MATCH_FRACTIONS)
        print(f"  {arm.label:<36} {arm.substeps:4d}  {d0:<40}{d9}")

    ship = means["shipped"]
    fat = means["fat"]
    coarse = means["coarse_fat"]

    def pct(new, old):
        return 100.0 * (new - old) / old

    dx_at_6mm = pct(fat, coarse)
    diam_at_8cells = pct(coarse, ship)

    # LEAD WITH THIS. §3.14 said report both arms and never only the quotient; here
    # that is not a caveat but the mechanism — the two arms respond by 3x different
    # amounts, so the quotient is nearly an S=90 measurement wearing a ratio's clothes.
    print("\n  WHERE THE EFFECT LIVES — the S=0 arm is nearly insensitive to BOTH knobs")
    f_hi = MATCH_FRACTIONS[-1]
    s0 = {k: depth_at(arms[k][0], f_hi) for k in means}
    s9 = {k: depth_at(arms[k][1], f_hi) for k in means}
    label = {a.key: a.label for a in DIAM_ARMS}
    print(f"    depth at f={f_hi:.2f}, vs the shipped arm:")
    for k in ("fat", "coarse_fat"):
        print(f"      {label[k]:<34} S=0 {s0['shipped']:6.2f} -> {s0[k]:6.2f} "
              f"({pct(s0[k], s0['shipped']):+6.2f} %)"
              f"   S=90 {s9['shipped']:6.2f} -> {s9[k]:6.2f} ({pct(s9[k], s9['shipped']):+6.2f} %)")
    print("    The jet flies the standoff and THINS before the S=90 impact, so that is")
    print("    where diameter and dx are actually spent. The quotient inherits it.")

    print("\n  THE THREE READS (all three arms run the SAME dt — none is dt-confounded,")
    print("  which is what §3.14's two-route comparison could not say of itself)")
    print(f"    dx alone at 6 mm,   0.7500 -> 0.3750  (8 -> 16 cells) : "
          f"{coarse:6.4f} -> {fat:6.4f}  ({dx_at_6mm:+6.2f} %)")
    print(f"    diameter alone at dx=0.3750, 3 -> 6 mm (8 -> 16 cells): "
          f"{ship:6.4f} -> {fat:6.4f}  ({pct(fat, ship):+6.2f} %)")
    print(f"    SCALE, both doubled at FIXED 8 cells                  : "
          f"{ship:6.4f} -> {coarse:6.4f}  ({diam_at_8cells:+6.2f} %)")

    # cells = diameter/dx identically, so "diameter at fixed cells" is not a thing
    # that can be measured: fixing the ratio and moving one factor moves the other.
    # What CAN be measured is whether the ratio is sufficient — that is the scale row.
    scale_f = route_difference(per_f["shipped"], per_f["coarse_fat"])
    print("\n  THE SCALE ROW IS THE TEST, and §3.8's claim is what it tests.")
    print("    `cells across the jet` is not an independent variable — it is the RATIO")
    print("    diameter/dx — so calling it THE controlling parameter is a claim that the")
    print("    response depends on that ratio ALONE. Both 8-cell arms are the same")
    print("    discretization scaled by 2x against physics that does not scale with it")
    print("    (the standoff, the process zone). If the claim held, the scale row would")
    print("    read 0.00 %. It does not:")
    print("      per matched fraction " + "".join(f" f={f:.2f}" for f in MATCH_FRACTIONS))
    print("                           " + "".join(f"{v:+7.2f}" for v in scale_f))
    print(f"    It varies {min(scale_f):+.2f} % to {max(scale_f):+.2f} % across the window and is NOT one number.")
    print(f"    Quote it at a stated fraction — {scale_f[-1]:+.2f} % at f={f_hi:.2f}, still "
          f"{abs(scale_f[-1]) / DT_RESOLUTION_PCT:.0f}x the floor.")

    gap, dt_term, residual = _dt_residual(caches)
    print("\n  AGAINST §3.14's RESIDUAL (re-derived from the caches, not quoted)")
    print(f"    two-route gap {gap:+.2f} %, dt-only term {dt_term:+.2f} %, "
          f"residual NOT dt: {residual:+.2f} %")
    print(f"    the SAME route comparison one scale coarser reads {scale_f[-1]:+.2f} % to "
          f"{scale_f[0]:+.2f} %,")
    print("    and needs no dt correction to say so.")
    if abs(diam_at_8cells) < DT_RESOLUTION_PCT:
        verdict = ("the route difference VANISHES one scale coarser, inside the\n"
                   "      ±%.1f %% floor. A route difference that appears only at fine\n"
                   "      resolution is not a physical offset." % DT_RESOLUTION_PCT)
    elif diam_at_8cells * residual > 0:
        verdict = ("the route difference keeps its SIGN one scale coarser, which is\n"
                   "      what a resolution-independent physical offset would do. Consistent\n"
                   "      with §3.14's candidate; two points do not establish it.")
    else:
        verdict = ("the route difference CHANGES SIGN one scale coarser, and changes\n"
                   "      size. A physical finite-diameter effect would be a roughly\n"
                   "      resolution-INDEPENDENT offset; this behaves like discretization\n"
                   "      error. That characterizes the residual rather than merely ruling\n"
                   "      a candidate out — but it does NOT refute the fine-pair reading:\n"
                   "      both arms here are coarser than both arms there, so a coarse pair\n"
                   "      cannot overturn a measurement made at finer resolution.")
    print(f"    reading: {verdict}")

    print("\n  WHAT THIS DOES NOT SETTLE, said rather than left to be found:")
    print("    * TWO POINTS ARE NOT A TREND. This mode has exactly two route-difference")
    print("      readings, at 8 and at 16 cells. Do not read a crossing between them,")
    print("      do not say where it would vanish, do not extrapolate. (Milestone 19")
    print("      added a third at 12 cells — see --route-difference, which supersedes")
    print("      this bullet's COUNT and not one word of its posture.)")
    print("    * §3.14's dx-only rows (+17.46 %, +20.27 %) were measured at 3 mm and at")
    print("      684 substeps; the dx-only row above is at 6 mm and at the shipped clock,")
    print("      so comparing SLOPES still carries the dx x dt interaction — the same")
    print("      open term §3.14 declared, not a new one, and not closed here.")
    print(f"\n  Resolution floor {DT_RESOLUTION_PCT} %: the repo's 0.11 % aggregate scatter, and the")
    print("  -0.113 % this mode's 225 -> 75 decimation costs the shipped arm.")
    return 0


# --- milestone 19: the route difference as a function of cells across -----------
#
# Every arm this mode needs, keyed. Overlaps `DT_ARMS` / `DIAM_ARMS` on purpose
# rather than importing rows out of them: those two carry PUBLISHED tables whose
# `substeps` columns use different cadence conventions (see the note over
# `DIAM_ARMS`), and quietly reusing a row would make one mode's presentation choice
# load-bearing in another mode's numbers. This table displays `dt_ms`, which is the
# quantity every pairing claim here is actually about.
ROUTE_ARMS = [
    Arm("shipped", "3.0 mm jet, dx=0.3750",
        "standoff_s00", "standoff_s90", 114, 3, 1.754386e-6),
    Arm("coarse_fat", "6.0 mm jet, dx=0.7500",
        "standoff_conv_d6mm_dx750_s00", "standoff_conv_d6mm_dx750_s90", 342, 1, 1.754386e-6),
    Arm("dx250", "3.0 mm jet, dx=0.2500",
        "standoff_conv_dx250_s00", "standoff_conv_dx250_s90", 513, 1, 1.169591e-6),
    Arm("d4p5mm", "4.5 mm jet, dx=0.3750",
        "standoff_conv_d4p5mm_dt513_s00", "standoff_conv_d4p5mm_dt513_s90", 513, 1, 1.169591e-6),
    Arm("dx188", "3.0 mm jet, dx=0.1875",
        "standoff_conv_dx188_s00", "standoff_conv_dx188_s90", 684, 1, 8.771930e-7),
    Arm("d6mm", "6.0 mm jet, dx=0.3750",
        "standoff_conv_d6mm_s00", "standoff_conv_d6mm_s90", 342, 1, 1.754386e-6),
    # Not a route arm — the dt-only partner the 16-cell row has to be corrected with.
    Arm("dt684", "3.0 mm jet, dx=0.3750 (dt partner)",
        "standoff_conv_dt684_s00", "standoff_conv_dt684_s90", 684, 1, 8.771930e-7),
]


class RouteRow(NamedTuple):
    """One cell count, reached by both of §3.8's routes.

    `fine` is the smaller-jet / finer-`dx` member and `fat` the bigger-jet /
    coarser-`dx` one, in every row, so the sign convention is uniform: the route
    difference is `fat` against `fine`, matching how §3.14 wrote its 16-cell gap
    (fat jet vs dx188) and §3.15 its 8-cell one (coarse fat vs shipped).

    `scale` is the factor separating the two arms — NOT a free choice. `cells` is
    identically `diameter/dx`, so reaching a given count from the shipped 8 moves
    each factor by `cells/8`, and the 12-cell row is therefore a 1.5x row where the
    other two are 2x rows. That is a real limit on reading these as a trend and it
    is carried in the table rather than in a footnote.
    """

    cells: int
    fine: str
    fat: str
    scale: float


ROUTE_ROWS = [
    RouteRow(8, "shipped", "coarse_fat", 2.0),
    RouteRow(12, "dx250", "d4p5mm", 1.5),
    RouteRow(16, "dx188", "d6mm", 2.0),
]


def _route_difference(caches: Path) -> int:
    print("ROUTE DIFFERENCE: by how much does `cells across the jet` fail as THE")
    print("parameter, and does that depend on resolution?")
    print("  §3.8 reaches a cell count two ways — refine `dx`, or fatten the jet at")
    print("  the shipped `dx` — and calls the ratio the controlling parameter. §3.15")
    print("  observed that `cells` IS the ratio diameter/dx, so that is a")
    print("  SCALE-INVARIANCE claim and a route difference is the amount it fails by.")
    print("  Two readings existed (8 and 16 cells) and milestone 18 forbade reading a")
    print("  trend from them. This mode adds the third, at 12 cells.\n")

    by_key = {a.key: a for a in ROUTE_ARMS}
    arms, per_f, means = {}, {}, {}
    print("  half-space check (a perforated baseline would INFLATE its row's ratio)")
    for arm in ROUTE_ARMS:
        a, b = measure(caches / arm.s00, arm.stride), measure(caches / arm.s90, arm.stride)
        arms[arm.key] = (a, b)
        per_f[arm.key] = compare(a, b)
        means[arm.key] = float(np.nanmean(per_f[arm.key]))
        for r in (a, b):
            _report(r)
    print()

    hdr = "".join(f" f={f:.2f} " for f in MATCH_FRACTIONS)
    print(f"  {'arm':<36} {'cells':>5} {'dt (ms)':>12}  {hdr}   mean")
    for arm in ROUTE_ARMS:
        a, _ = arms[arm.key]
        print(f"  {arm.label:<36} {a['cells']:5.1f} {arm.dt_ms:12.6e}  "
              + "".join(f"{r:7.4f} " for r in per_f[arm.key]) + f"  {means[arm.key]:6.4f}")
    print("\n  a-priori prediction (nothing fitted)                                             1.5357")
    print("  `cells` is MEASURED off the seeded lattice; `dt` is a label pinned in")
    print("  solver/tests/ against `plan_substeps`, since no cache records one.")

    # dt-matching is COMPUTED from the table, never asserted. The 16-cell row flags
    # itself as unmatched because its two arms carry different `dt_ms`, which is the
    # whole reason §3.14 had to correct it and this milestone's row does not.
    print("\n  THE ROUTE DIFFERENCE AT EACH CELL COUNT (the fat-jet route against the")
    print("  fine-dx route, per matched consumed fraction — never a mean, because §3.15")
    print("  measured this quantity varying 3x across the window)")
    print(f"  {'cells':>5} {'scale':>6} {'dt':>12}  {hdr}")
    rows = {}
    for row in ROUTE_ROWS:
        fine, fat = by_key[row.fine], by_key[row.fat]
        matched = fine.dt_ms == fat.dt_ms
        d = route_difference(per_f[row.fine], per_f[row.fat])
        rows[row.cells] = (d, matched, row.scale)
        tag = "matched" if matched else "NOT matched"
        print(f"  {row.cells:5d} {row.scale:5.1f}x {tag:>12}  "
              + "".join(f"{v:+7.2f} " for v in d))

    # The 16-cell row is the only dt-confounded one, and correcting it PER FRACTION
    # is a departure from §3.14, which corrected a mean. Both are printed: the
    # per-fraction form is the one comparable to the rows above, and the mean form is
    # the published §3.14 figure, re-derived from the caches rather than quoted.
    dt_f = route_difference(per_f["shipped"], per_f["dt684"])
    corrected = [f / (1.0 + t / 100.0) for f, t in zip(per_f["dx188"], dt_f)]
    d16_corr = route_difference(corrected, per_f["d6mm"])
    print(f"  {16:5d} {2.0:5.1f}x {'dt-CORRECTED':>12}  "
          + "".join(f"{v:+7.2f} " for v in d16_corr))
    print("    the correction is the dt-only term, per fraction: "
          + "".join(f"{v:+7.2f} " for v in dt_f))

    # §3.14's residual is a MEAN, and this mode is the first thing able to show what
    # that mean is a mean OF. Print both, adjacent, because §3.15 compared its own
    # per-fraction f=0.30 figure against this mean without flagging the mismatch.
    gap, dt_term, residual = _dt_residual(caches)
    lo16, hi16 = min(d16_corr), max(d16_corr)
    print(f"\n  §3.14's published split, re-derived from these caches (on MEANS, which is")
    print(f"  how it was published): gap {gap:+.2f} %, dt term {dt_term:+.2f} %, "
          f"residual {residual:+.2f} %")
    print(f"  THAT RESIDUAL IS A MEAN OVER A WINDOW IT RUNS {lo16:+.2f} % TO {hi16:+.2f} % ACROSS,")
    print("  changing sign inside it. The mean is not wrong, it is a mean; but §3.15")
    print("  compared its own PER-FRACTION f=0.30 figure to it, which is two different")
    print("  statistics. The same-statistic comparison is the table below.")
    print("  The split also rests on transferring a dt term measured at 8 cells onto")
    print("  the 16-cell arm — the dx x dt interaction §3.13 named as out of reach. The")
    print("  8- and 12-cell rows need no such transfer: their two arms share a substep.")

    f_hi = MATCH_FRACTIONS[-1]
    i_hi = len(MATCH_FRACTIONS) - 1
    d8, d12 = rows[8][0][i_hi], rows[12][0][i_hi]
    d16 = d16_corr[i_hi]
    print(f"\n  AT f={f_hi:.2f}, SAME STATISTIC ON ALL THREE — the most-consumed, most-converged")
    print("  end, and the one §3.15 says to quote (f=0.15 swings most, §3.13's x=160 reason):")
    print(f"    8 cells (2.0x, dt-free)       {d8:+7.2f} %")
    print(f"   12 cells (1.5x, dt-free)       {d12:+7.2f} %   <-- milestone 19")
    print(f"   16 cells (2.0x, dt-corrected)  {d16:+7.2f} %   (raw, uncorrected: "
          f"{route_difference(per_f['dx188'], per_f['d6mm'])[i_hi]:+.2f} %)")

    print("\n  WHAT THE THIRD POINT CAN AND CANNOT SAY.")
    print("    CAN, 1 — scale invariance is falsified at a THIRD resolution, and the")
    print("    falsification cannot be blamed on the scale factor: an exact invariance")
    print(f"    reads 0.00 % at ANY factor. Every row is {min(abs(v) for v in (d8, d12, d16)) / DT_RESOLUTION_PCT:.0f}x the {DT_RESOLUTION_PCT} % floor or more.")
    # The factor confound makes a DIRECTIONAL prediction, so it can be tested rather
    # than merely declared. A violation whose size is set by the scale separation must
    # be SMALLER at 1.5x than at 2.0x.
    # The QUANTITATIVE form, not the inequality. "Lands outside the bracket the other
    # two span" is true here by 0.38 pp, which at ~2x the floor is a weak instrument.
    # Proportionality makes a real prediction and misses by ~7 floors.
    r12 = next(row.scale for row in ROUTE_ROWS if row.cells == 12)
    r8 = next(row.scale for row in ROUTE_ROWS if row.cells == 8)
    predicted = d8 * (r12 - 1.0) / (r8 - 1.0)
    print(f"    CAN, 2 — if the violation scaled WITH the separation, the {r12}x row would")
    print(f"    read about {predicted:+.2f} % against the {r8}x row's {d8:+.2f} %. Measured "
          f"{d12:+.2f} %,")
    print(f"    off that prediction by {abs(d12 - predicted):.2f} pp = "
          f"{abs(d12 - predicted) / DT_RESOLUTION_PCT:.0f}x the floor, and in the wrong direction")
    print("    (larger, not smaller). The scale factor is not what sets the magnitude.")
    across = [abs(a - b) for a, b in zip(rows[8][0], rows[12][0])]
    print(f"    The two dt-free rows differ by {min(across):.2f}-{max(across):.2f} pp across the window — "
          f"and the")
    print(f"    largest of those ({max(across):.2f} pp) is at f={MATCH_FRACTIONS[across.index(max(across))]:.2f}, the end §3.15 says is least")
    print("    trustworthy. Do not read them as coincident; read them as not ordered by")
    print("    the scale factor.")
    print("    CANNOT — say which of TWO explanations makes 16 cells the outlier:")
    print("    a genuine resolution dependence appearing between 12 and 16 cells, or a")
    print("    dt correction that does not transfer. Note the 16-cell row is an outlier")
    print("    BEFORE any correction, so the second would need a dt term at 16 cells far")
    print("    larger than the one measured at 8 — which is precisely the dx x dt")
    print("    interaction, named again and still out of reach.")
    print("    AND STILL NOT: a crossing, a zero-point, an extrapolation, or an order.")
    print("    Three points on a two-factor grid are not a trend either, and this repo")
    print("    has been bitten by exactly that reading three times.")

    print(f"\n  Resolution floor {DT_RESOLUTION_PCT} %: the repo's 0.11 % aggregate scatter, and the")
    print("  -0.113 % this mode's 225 -> 75 decimation costs the shipped arm.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache_dirs", type=Path, nargs="*")
    ap.add_argument("--family", action="store_true", help="the shipped standoff_s* family")
    ap.add_argument("--convergence", action="store_true", help="the standoff_conv_* study")
    ap.add_argument("--dt-decomposition", action="store_true",
                    help="milestone 17: separate dx from the dt CFL drags along with it")
    ap.add_argument("--diameter-decomposition", action="store_true",
                    help="milestone 18: separate the jet's diameter from the cells it buys")
    ap.add_argument("--route-difference", action="store_true",
                    help="milestone 19: the two routes' disagreement at 8 / 12 / 16 cells")
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
        # Keyed, like every other table here (see `Arm`): the under-read line below
        # reads the SHIPPED row out of this list and used to do it as `cfg[0][0]`.
        cfg = [
            ("shipped", "3 mm jet, dx=0.375 (SHIPPED)", 8, "standoff_s00", "standoff_s90", "CFL"),
            ("dx250", "3 mm jet, dx=0.250", 12, "standoff_conv_dx250_s00", "standoff_conv_dx250_s90", "CFL"),
            ("dx188", "3 mm jet, dx=0.1875", 16, "standoff_conv_dx188_s00", "standoff_conv_dx188_s90", "CFL"),
            ("d4p5mm", "4.5 mm jet, dx=0.375", 12,
             "standoff_conv_d4p5mm_dt513_s00", "standoff_conv_d4p5mm_dt513_s90", "deck"),
            ("d6mm", "6 mm jet, dx=0.375", 16, "standoff_conv_d6mm_s00", "standoff_conv_d6mm_s90", "CFL"),
        ]
        print("CONVERGENCE: is the standoff shortfall numerical?")
        print("  The derivation is DIAMETER-INDEPENDENT, so every row predicts 1.536.")
        print("  Two independent routes to each cell count: refine dx, or fatten the jet.\n")
        # The half-space premise is checked HERE too, not only in --family. The 6 mm
        # jet carries 2x the mass of the 3 mm one, so it is the row most able to
        # perforate — and it is also the load-bearing independent confirmation. A
        # ceiling-capped S=0 depth would INFLATE the ratio, i.e. flatter a row that
        # exists to be believed.
        measured = {}
        print("  half-space check (a perforated baseline would INFLATE its row's ratio)")
        for key, name, cells, c0, c9, bound in cfg:
            a, b = measure(args.caches / c0), measure(args.caches / c9)
            measured[key] = (a, b)
            for r in (a, b):
                _report(r)
        print()
        print("  configuration                  cells  dt by  "
              + "".join(f" f={f:.2f} " for f in MATCH_FRACTIONS) + "   mean")
        for key, name, cells, c0, c9, bound in cfg:
            a, b = measured[key]
            rs = compare(a, b)
            print(f"  {name:<30} {cells:4d}  {bound:>5}  " + "".join(f"{r:7.4f} " for r in rs)
                  + f"  {np.nanmean(rs):6.4f}")
        print("\n  a-priori prediction (nothing fitted)                                        1.5357")
        # The `dt by` column exists because this table's rows do NOT share a substep,
        # and never did — dt is CFL-bound to dx, so the dx rows each run their own
        # (114 / 513 / 684) while the fat-jet row runs the shipped one. §3.14 measured
        # what that costs. The 4.5 mm row is the one whose dt is set by its DECK, to
        # pair it with dx250 for `--route-difference`; it is flagged rather than left
        # to be discovered, since a reader could otherwise take this table's rows as
        # differing in cells alone. They differ in cells and in clock.
        print("  `dt by`: which of min(deck_dt, cfl_dt) bound the substep. NOT a shared")
        print("  clock — see --dt-decomposition for what that costs, and note the 4.5 mm")
        print("  row is pinned to dx250's substep on purpose (--route-difference).")
        # COMPUTED, not typed. This line used to carry a hardcoded "~2.3x" measured
        # under Murnaghan; two rebakes (M13's EOS, M14's CFL margin) moved every row
        # under it and the prose did not follow. A figure that restates a measurement
        # in prose goes stale silently — so derive it from the row just printed.
        shipped = float(np.nanmean(compare(*measured["shipped"])))
        print(f"\n  The shipped row is the WORST. It under-reads the effect "
              f"~{(1.5357 - 1.0) / (shipped - 1.0):.1f}x on the excess "
              f"({1.5357 - 1.0:.3f} predicted vs {shipped - 1.0:.3f} measured).")
        print("  Each ROUTE is monotone toward the prediction — 1.2657 -> 1.4573 ->")
        print("  1.4968 by dx, and 1.2657 -> 1.3674 -> 1.5587 by diameter. But the two")
        print("  routes DISAGREE at both 12 and 16 cells, so this table is a trend")
        print("  within each route and NOT a single sequence in `cells`. That")
        print("  disagreement is the subject of --route-difference; do not read across")
        print("  the routes here. And not a Richardson extrapolation either — the")
        print("  observed order in this family is ill-conditioned.")
        return 0

    if args.dt_decomposition:
        return _dt_decomposition(args.caches)

    if args.diameter_decomposition:
        return _diameter_decomposition(args.caches)

    if args.route_difference:
        return _route_difference(args.caches)

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
