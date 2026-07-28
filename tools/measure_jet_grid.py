#!/usr/bin/env python3
"""Measure how a shaped-charge jet's penetration depends on GRID RESOLUTION.

Language-neutral helper (CLAUDE.md §3): reads caches per docs/CACHE_FORMAT.md and
imports nothing from the solver or the visualizer.

    python tools/measure_jet_grid.py --family          # the heat_conv_* ladder
    python tools/measure_jet_grid.py caches/a caches/b # any two arms, a = baseline
    python tools/measure_jet_grid.py --family --plot out.png

WHY THIS TOOL EXISTS, WHEN TWO PENETRATION TOOLS ALREADY DO. Neither can answer
the question:

  * `measure_penetration.py` deliberately does NOT report depth (it reports the
    instantaneous erosion rate u) and its own docstring warns that a PERFORATING
    deck fits a beautiful straight line to a physically unreachable number.
  * `measure_standoff.py` reports depth, but matches on CONSUMED FRACTION because
    its arms impact at different times, and it treats perforation as a condition
    that VOIDS the measurement rather than one to measure through.

The deck this study is about — `heat_vs_composite` — perforates by construction.

THE METRIC IS THE WHOLE EXPERIMENT, AND THE OBVIOUS ONE IS BLIND HERE. The stack's
tip clears the back face at ~26 us of a 30 us window, so DEPTH AT THE END OF THE
WINDOW IS CEILINGED: every grid reads ~the stack thickness and the arms look
identical no matter how differently they penetrated. That is an instrument that
cannot see the failure it is pointed at. PHYSICS §3.8 chose a 150 mm half-space
precisely to dodge this, which is exactly why the shipped composite deck was never
itself refined. So:

  * PRIMARY: the penetration front as a CURVE, x(t), not a scalar.
  * SCALARS: the ARRIVAL TIME at each interface in the stack. Every one of them is
    uncapped until the last. They are read off the same curve.
  * The saturated depth-at-window-end is reported too, LOUDLY, so the ceiling is
    visible rather than inviting (the same posture `measure_standoff.py` takes
    toward depth at a fixed lab time).

LAB TIME IS A LEGITIMATE AXIS HERE — and that is not a contradiction of §3.8's
"match on consumed fraction, NEVER lab time". That rule is a STANDOFF confound: a
longer standoff impacts later, so it penetrates for less of the window and reads
shallower, i.e. the metric reports the OPPOSITE SIGN of the effect. These arms
differ only in `grid_resolution` — same seeding, same standoff, same virtual
origin, so first contact is at the same instant and the clock is shared. Consumed
fraction is reported at every interface anyway, so the two axes can be checked
against each other rather than one being taken on trust.

HOW IT AVOIDS BAKING IN ASSUMPTIONS:

  * The jet is whatever is MOVING at frame 0 — not a material id, not an x-band.
  * The interfaces are found by CLUSTERING the frame-0 static particles by
    material and contiguity, before anything moves. They are NOT read from the
    manifest's `armor` block: that block is PROVENANCE, not data (CACHE_FORMAT
    §2.1), and measuring from it would let a mislabelled deck agree with itself.
  * The front is a high PERCENTILE of jet x, not the max, so one particle spat off
    the crater lip cannot define it. A percentile is a fixed fraction of jet
    MATERIAL, which is what makes it comparable across grids that hold wildly
    different particle counts; `--sensitivity` re-reads every arrival at three
    percentiles so a reader can see the choice is not carrying the result.
  * Nothing here knows the grid spacing from the manifest (it does not record one).
    `dx` is MEASURED from the frame-0 particle lattice, so the cells-across-the-jet
    label is a reading of the cache rather than a number typed in beside it. The
    exact reading is the ROW COUNT across the jet; `dx` follows from it. The
    solver's `_fill_rect` ROUNDS each object's lattice to fit it exactly, so a
    3 mm jet at dx=0.390625 seeds 15 rows (pitch 0.200), not 15.36 — the seeded
    resolution is 7.5 cells, ~2 % off the 7.68 that `domain/grid_resolution`
    implies. This tool reports what was SEEDED, which is what penetrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The penetration front, as a percentile of jet x. Deliberately includes damaged
# particles: jet material at the crater bottom is actively being consumed, so
# restricting to live particles would systematically lag the true front. Same
# choice, and the same reason, as `measure_standoff.py`.
FRONT_PERCENTILE = 99.5

# Re-read every arrival at these, to show the answer does not live in the choice.
SENSITIVITY_PERCENTILES = (99.0, 99.5, 99.9)

# A particle is "moving" at frame 0 above this speed (mm/ms). Armor is seeded at
# rest, so anything above a rounding error is projectile.
MOVING_MM_PER_MS = 1.0

# A void between armor layers is a gap of at least this many particle spacings.
# Bonded layers touch (one spacing apart); the deck's standoff gaps are tens of mm.
VOID_SPACINGS = 4.0

# Interface candidates closer together than this many spacings are the two sides of
# one bonded contact (a layer's back face and the next layer's front face) and are
# merged into a single interface.
MERGE_SPACINGS = 2.0

# Ignore clusters smaller than this — a handful of strays must not invent a layer.
MIN_LAYER_PARTICLES = 32

# Particles per cell along one axis. `_seed` lays particles on an n_side x n_side
# lattice per cell with n_side = round(sqrt(particles_per_cell)), and every deck in
# this repo uses particles_per_cell = 4. Only the cells-across LABEL depends on it;
# no measurement does.
N_SIDE = 2

# The residual tip: among jet particles past the back face, the leading this-fraction
# by x. Below MIN_RESIDUAL of them there is no tip to speak of and the reading is
# withheld.
RESIDUAL_LEAD_FRACTION = 0.05
MIN_RESIDUAL = 20

# Read the residual this long after THAT ARM'S OWN breakout, not at the final frame.
# The arms break out at different times, so at a shared final frame they have had
# different amounts of free flight — a lab-time confound of exactly the shape §3.8
# warns about, imported into the one metric that lives entirely after perforation.
# 4 us fits inside every arm's post-breakout window with room to spare; the
# final-frame reading is reported beside it so the correction is visible.
RESIDUAL_ELAPSED_US = 4.0


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


def _spacing(xs: np.ndarray) -> float:
    """Particle lattice spacing, read off a sorted coordinate array.

    Particles sit in columns, so most sorted neighbour gaps are 0 and the real
    ones are the column pitch. The median of the POSITIVE gaps is that pitch.
    """
    d = np.diff(np.sort(xs))
    d = d[d > 0]
    if d.size == 0:
        raise SystemExit("degenerate particle lattice: no distinct coordinates")
    return float(np.median(d))


def interfaces(f0: np.ndarray, col: dict, static: np.ndarray) -> tuple[list[float], float]:
    """Every material interface in the stack, from frame 0 — before anything moves.

    Returns (interface x positions ascending, particle spacing). The stack is
    segmented by BOTH a void (the standoff gap) and a material change (a bonded
    ceramic/steel contact, which has no gap at all), because the deck this exists
    for has one of each and a gap-only rule would silently miss the bonded one.
    """
    xs = f0[static, col["pos_x"]]
    mid = f0[static, col["material_id"]]
    order = np.argsort(xs, kind="stable")
    xs, mid = xs[order], mid[order]
    sp = _spacing(xs)

    brk = (mid[1:] != mid[:-1]) | (np.diff(xs) > VOID_SPACINGS * sp)
    segs = np.split(np.arange(xs.size), np.flatnonzero(brk) + 1)

    faces: list[float] = []
    for s in segs:
        if s.size < MIN_LAYER_PARTICLES:
            continue
        faces.extend((float(xs[s].min()), float(xs[s].max())))

    # Merge the two sides of a bonded contact into one interface.
    merged: list[float] = []
    for x in sorted(faces):
        if merged and x - merged[-1] <= MERGE_SPACINGS * sp:
            merged[-1] = 0.5 * (merged[-1] + x)
        else:
            merged.append(x)
    return merged, sp


def measure(cache_dir: Path, percentile: float = FRONT_PERCENTILE) -> dict:
    manifest, frames = load(cache_dir)
    col = {a: i for i, a in enumerate(manifest["attributes"])}
    for need in ("pos_x", "pos_y", "vel_mag", "damage", "material_id"):
        if need not in col:
            raise SystemExit(f"cache lacks required attribute {need!r}")

    f0 = np.asarray(frames[0])
    jet = f0[:, col["vel_mag"]] > MOVING_MM_PER_MS
    if not jet.any() or jet.all():
        raise SystemExit("need both a moving jet and a static target at frame 0")

    faces, sp = interfaces(f0, col, ~jet)
    if len(faces) < 2:
        raise SystemExit(f"{cache_dir.name}: found {len(faces)} interfaces, need >= 2")
    face_x, back_x = faces[0], faces[-1]

    # The jet's width at frame 0, and from it the grid it was baked on. `dx` is not
    # in the manifest, so it is measured: the seeding lattice has N_SIDE particles
    # per cell per axis, so the lattice pitch is dx / N_SIDE.
    #
    # ROWS is the exact reading and `dx` is derived from it, not the other way
    # round, because `_fill_rect` rounds each object's lattice to fit the object —
    # so the seeded pitch is the object's own size / an integer, and can sit a
    # couple of percent off `domain / grid_resolution`. What penetrated is the
    # seeded jet, so that is what gets reported.
    jet_y = f0[jet, col["pos_y"]]
    jet_sp = _spacing(jet_y)
    rows = int(round(float(jet_y.max() - jet_y.min()) / jet_sp)) + 1
    dx = jet_sp * N_SIDE
    width = float(jet_y.max() - jet_y.min()) + jet_sp  # +1 pitch: centres to edges

    n_jet = int(jet.sum())
    fc = manifest["frame_count"]
    frame_dt_us = manifest["frame_dt"] * 1.0e6
    t_us = np.arange(fc) * frame_dt_us
    front = np.empty(fc)
    consumed = np.empty(fc)
    for f in range(fc):
        fr = np.asarray(frames[f])
        front[f] = np.percentile(fr[jet, col["pos_x"]], percentile)
        consumed[f] = float((fr[jet, col["damage"]] >= 0.5).sum()) / n_jet

    # Residual tip past the back face — once at the shared final frame, once at a
    # matched elapsed time after this arm's own breakout (see RESIDUAL_ELAPSED_US).
    def _residual(f_idx: int):
        fr = np.asarray(frames[f_idx])[jet]
        past = fr[fr[:, col["pos_x"]] > back_x]
        if past.shape[0] < MIN_RESIDUAL:
            return float("nan"), 0, float("nan")
        k = max(MIN_RESIDUAL, int(round(RESIDUAL_LEAD_FRACTION * past.shape[0])))
        lead = past[np.argsort(past[:, col["pos_x"]])[-k:]]
        # `/ n_jet` is a MATERIAL fraction, not a particle-count artifact: `_seed`
        # gives every particle the same volume (`p_vol = spacing*spacing`) and so
        # every jet particle within an arm the same mass. The fraction is therefore
        # directly comparable across arms holding 9k and 40k jet particles.
        return (float(np.median(lead[:, col["vel_mag"]])), int(past.shape[0]),
                float(past.shape[0]) / n_jet)

    residual_v, n_residual, frac_through = _residual(fc - 1)
    t_break = _cross(front, t_us, back_x)
    # The matched target must be INSIDE the window. `argmin` alone would silently
    # return the last frame for an arm that broke out too late — reporting a
    # final-frame reading under the "matched" heading, which is the exact confound
    # this metric exists to remove, wearing its label.
    if np.isfinite(t_break) and t_break + RESIDUAL_ELAPSED_US <= t_us[-1]:
        i_m = int(np.argmin(np.abs(t_us - (t_break + RESIDUAL_ELAPSED_US))))
        residual_v_matched, n_residual_matched, frac_through_matched = _residual(i_m)
        t_matched = float(t_us[i_m])
    else:
        residual_v_matched = frac_through_matched = t_matched = float("nan")
        n_residual_matched = 0

    return {
        "cache": cache_dir.name,
        "n_particles": manifest["particle_count"],
        "n_jet": n_jet,
        "dx": dx,
        "jet_width": width,
        "rows_across": rows,
        "cells_across": rows / N_SIDE,
        "faces": faces,
        "face_x": face_x,
        "back_x": back_x,
        "t_us": t_us,
        "front": front,
        "consumed": consumed,
        "depth_end": float(front[-1] - face_x),
        "thickness": back_x - face_x,
        "perforated": bool(front.max() >= back_x),
        "residual_v": residual_v,
        "n_residual": n_residual,
        "frac_through": frac_through,
        "residual_v_matched": residual_v_matched,
        "n_residual_matched": n_residual_matched,
        "frac_through_matched": frac_through_matched,
        "t_break": t_break,
        "t_matched": t_matched,
        # The RECORDING length, not a physical one. Kept so `--dt-ladder` can refuse
        # to compare the one quantity that depends on it (`depth_end`) across arms
        # that do not share it: milestone 19's `heat_conv_dt342` runs a longer window
        # than the arm it partners, deliberately (its header says why), and a table
        # that quietly differenced their depth_end would report the window as physics.
        "window_us": float(t_us[-1]),
    }


def _cross(front: np.ndarray, t: np.ndarray, x: float) -> float:
    """When `front` first reaches x, interpolated between frames (us, from t=0).

    NaN if it never does. That distinction is the point: an arm that fails to reach
    an interface inside its window has NOT arrived late, it has not arrived, and a
    tool that returned the window end for both would erase the difference.
    """
    hit = np.flatnonzero(front >= x)
    if hit.size == 0 or hit[0] == 0:
        return float("nan")
    i = int(hit[0])
    x0, x1 = front[i - 1], front[i]
    if x1 == x0:
        return float(t[i])
    return float(t[i - 1] + (x - x0) / (x1 - x0) * (t[i] - t[i - 1]))


def arrival_us(r: dict, x: float) -> float:
    """When this arm's front first reaches x (us from t=0); NaN if it never does."""
    return _cross(r["front"], r["t_us"], x)


def consumed_at(r: dict, t_us: float) -> float:
    """Jet consumed fraction at a lab time — the second axis, for cross-checking."""
    if not np.isfinite(t_us):
        return float("nan")
    return float(np.interp(t_us, r["t_us"], r["consumed"]))


def _report_arm(r: dict) -> None:
    # NOT "ceilinged": this number routinely EXCEEDS the stack thickness, because
    # after perforation the leading edge is a free residual, not a crater bottom.
    # The label has to say which quantity it became, or it invites the reading the
    # tool exists to prevent.
    ceiling = "  <-- PERFORATED: this is free flight, not depth" if r["perforated"] else ""
    print(
        f"  {r['cache']:<22} {r['n_particles']:>8d} particles  dx={r['dx']:.4f} mm"
        f"  {r['rows_across']:3d} rows = {r['cells_across']:5.1f} cells"
        f" across a {r['jet_width']:.2f} mm jet"
    )
    print(
        f"  {'':<22} interfaces at x = "
        + ", ".join(f"{x:.2f}" for x in r["faces"])
        + f"   depth_end {r['depth_end']:6.1f} of {r['thickness']:.1f} mm{ceiling}"
    )


def _table(arms: list[dict]) -> None:
    # Every arm is timed to the SAME x positions — the coarsest arm's faces — not
    # to its own. Each arm measures its interfaces half a lattice pitch inside the
    # true face, and that pitch shrinks with the grid, so per-arm faces would put a
    # ~0.05 mm systematic INTO the very difference being measured. Using one shared
    # set makes the offset common-mode. `main` refuses arms whose geometry disagrees
    # by more than 0.5 mm, so the shared set is never the wrong scenario's.
    faces = arms[0]["faces"]
    base = arms[0]

    print("\nARRIVAL TIME at each interface (us from t=0), and vs the coarsest arm")
    head = "  cells   " + "".join(f"  x={x:<7.1f}" for x in faces[1:])
    print(head)
    for r in arms:
        row = f"  {r['cells_across']:5.1f}   "
        for x in faces[1:]:
            a = arrival_us(r, x)
            row += f"  {a:8.3f}  " if np.isfinite(a) else "   not rchd "
        print(row)
    print("  " + "-" * (len(head) - 2))
    for r in arms[1:]:
        row = f"  {r['cells_across']:5.1f}   "
        for x in faces[1:]:
            a, b = arrival_us(base, x), arrival_us(r, x)
            row += f"  {100.0 * (b - a) / a:+7.2f}% " if np.isfinite(a) and np.isfinite(b) else "      --    "
        print(row + "   vs coarsest")

    print("\nCONSUMED FRACTION at those same arrivals (the second axis)")
    print(head)
    for r in arms:
        row = f"  {r['cells_across']:5.1f}   "
        for x in faces[1:]:
            c = consumed_at(r, arrival_us(r, x))
            row += f"  {c:8.4f}  " if np.isfinite(c) else "      --    "
        print(row)

    print(f"\nRESIDUAL TIP past the back face, {RESIDUAL_ELAPSED_US:.0f} us after"
          " THAT ARM'S OWN breakout")
    print("  cells    v_resid     jet through    (at the shared final frame:"
          " v_resid, through)")
    for r in arms:
        v, vf = r["residual_v_matched"], r["residual_v"]
        vs = f"{v:7.1f} m/s" if np.isfinite(v) else "    (none)"
        vfs = f"{vf:7.1f} m/s" if np.isfinite(vf) else "    (none)"
        print(f"  {r['cells_across']:5.1f}  {vs}   {r['frac_through_matched']:7.4f}"
              f" of jet      {vfs}  {r['frac_through']:7.4f}")
    print("  The matched column is the honest one: the arms break out at different")
    print("  times, so at a shared final frame they have had different amounts of")
    print("  free flight. Both are printed so the size of that correction is visible.")

    # THE WINDOW GUARD LIVES HERE, not in one mode. `depth_end` is the only quantity
    # this tool reports that depends on how long recording continued, and milestone 19
    # baked an arm (`heat_conv_dt342`) that deliberately records 4 us longer than the
    # rest. `--dt-ladder` knows that; `--family` and the pairwise path did not, and the
    # pairwise path is exactly how §3.13 assembled its tables — so it is exactly how
    # someone will reach for that cache. A spread printed across unequal windows is the
    # recording length wearing a grid effect's label.
    if len({round(r["window_us"], 6) for r in arms}) > 1:
        print("\nTHE TRAP — 'depth' at the end of the window: WITHHELD.")
        print("  These arms do not share a recording window ("
              + ", ".join(f"{r['cells_across']:.0f}c: {r['window_us']:.1f} us" for r in arms) + "),")
        print("  so `depth_end` differs by the recording length alone and a spread across")
        print("  them would report that as physics. The arrival times above are read off")
        print("  the front CURVE and the residual off each arm's OWN breakout, so neither")
        print("  is affected. Read those.")
        return

    print("\nTHE TRAP — 'depth' at the end of the window (the OBVIOUS metric):")
    print("  " + "   ".join(f"{r['cells_across']:.0f}c: {r['depth_end']:.1f} mm" for r in arms))
    spread = [r["depth_end"] for r in arms]
    rng = (max(spread) - min(spread)) / min(spread) * 100.0
    thick = arms[0]["thickness"]
    print(f"  spread {rng:.2f} % across the ladder — against a stack only {thick:.1f} mm thick.")
    print("  Every arm PERFORATED, so past the back face the leading edge is no longer")
    print("  a crater bottom at all: it is a free residual flying downrange at ~its own")
    print("  velocity. This number is therefore not a ceiling that hides a difference,")
    print("  it is a DIFFERENT QUANTITY that invents one — the same failure")
    print("  measure_penetration.py warns of, where a perforated deck fits a beautiful")
    print("  line to an unreachable number. Read the arrival times above instead.")


def _sensitivity(dirs: list[Path]) -> None:
    print("\nSENSITIVITY — every arrival re-read at three front percentiles")
    for p in SENSITIVITY_PERCENTILES:
        arms = [measure(d, percentile=p) for d in dirs]
        faces = arms[0]["faces"]
        print(f"  percentile {p}")
        for r in arms:
            row = f"    {r['cells_across']:5.1f} cells  "
            for x in faces[1:]:
                a = arrival_us(r, x)
                row += f"{a:9.3f}" if np.isfinite(a) else "   not rchd"
            print(row)


def _plot(arms: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in arms:
        ax.plot(r["t_us"], r["front"], label=f"{r['cells_across']:.0f} cells (dx={r['dx']:.4f})")
    for x in arms[0]["faces"]:
        ax.axhline(x, color="0.7", lw=0.7, ls="--")
    ax.set_xlabel("time (us)")
    ax.set_ylabel(f"penetration front x (mm), p{FRONT_PERCENTILE}")
    ax.set_title("Jet penetration front vs grid resolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\n  wrote {out}")


# `heat_conv_dx094` is DELIBERATELY ABSENT, and the reason is the tool's own posture.
# That arm BREACHED its CFL audit by 1.60x (§3.17) — and a cache does not record the
# audit (CACHE_FORMAT §2 stores `frame_dt`, not a substep, a `c_max` or a warning), so
# this mode has no way to see it and no way to flag it. `--family` is the CONFOUNDED
# joint ladder, quoted for shape rather than for numbers, and printing a breached arm
# in it unflagged is exactly [[instruments-that-cannot-see-the-failure]]. The 32-cell
# arm is reported by `--dt-ladder`, where §3.17's prose carries the breach with it.
FAMILY = ("heat_vs_composite", "heat_conv_dx250", "heat_conv_dx188", "heat_conv_dx125")


# --- milestone 19: the matched-dt dx ladder ------------------------------------
#
# §3.13 built the dt PARTNERS but assembled its decomposition table by invoking this
# tool pairwise and copying numbers into prose. That is exactly how §3.8's table went
# two rebakes stale, so the ladder is a mode now — and NAME-KEYED from the first
# line, because milestone 18 found a published decomposition one `sort()` away from
# being silently re-keyed by positional indexing.
#
# Each rung is (cells LABEL, dx arm, its dt partner, substeps each). `cells` here is
# a label only: the tool MEASURES cells off the seeded lattice and the table below
# prints the measured value, so a wrong label shows up rather than propagating.
#
# The substep columns are labels too — no cache records `dt`, `grid_resolution` or a
# substep count (CACHE_FORMAT §2 records `frame_dt` only) and this tool imports
# nothing from the solver (CLAUDE.md §3). What they encode is pinned against the real
# sizing path in `solver/tests/`.
LADDER_BASELINE = "heat_vs_composite"
LADDER = [
    (12.0, "heat_conv_dx250", "heat_conv_dt_mid", 171, 171),
    (16.0, "heat_conv_dx188", "heat_conv_dt_fine", 228, 230),
    (24.0, "heat_conv_dx125", "heat_conv_dt342", 342, 342),
    (32.0, "heat_conv_dx094", "heat_conv_dt456", 456, 456),
]

# The interfaces worth quoting, per §3.13's own sensitivity sweep: at x=160 the front
# is a few hundred nanoseconds old and the percentile DEFINITION dominates (that cell
# swung 2.3 pp against a 0.9 pp effect), and x=190 is marginal. Read x=215 and the
# breakout at x=235. Nominal — each is resolved to the BASELINE arm's own measured
# face, see `_dt_ladder`.
LADDER_INTERFACES = (215.0, 235.0)


def _pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def _slope_table(vals: list[float], steps: list[float]) -> dict:
    """A ladder quantity's increments, its SLOPES in ``dx``, and consecutive-slope ratios.

    THE STATISTIC, IN ONE PLACE, SO A TEST CAN PIN IT WITHOUT A CACHE. §3.16 published
    an increment RATIO and could, because its two rungs were equally spaced in ``dx``.
    §3.17's rung halves the step, and over unequal steps a raw increment ratio has a
    different null at every rung (``h_new/h_old``) — so the same number would mean two
    things in one table. That is §3.16's own "apply a rule to BOTH SIDES" defect.

    If the discretisation error is first order, ``value(h) = value* + C*h`` and a rung's
    effect against the shipped grid is ``e(h) = C*(h - h_shipped)`` — LINEAR in ``h``.
    So ``increment/step`` recovers ``C`` at any spacing, and consecutive slopes have a
    null of 1.00 everywhere. Over EQUAL steps the step cancels and the slope ratio is
    identically the increment ratio, which is what makes this a change of convention
    rather than of answer: §3.16's 0.35 / 0.10 / 0.65 are reproduced exactly.

    ``ratios`` carries ``None`` where the denominator is not a reading — see the
    near-zero-denominator note in ``_dt_ladder``.
    """
    incs = [vals[k] - vals[k - 1] for k in range(1, len(vals))]
    slopes = [inc / s for inc, s in zip(incs, steps)]
    ratios = []
    for k in range(1, len(slopes)):
        if abs(incs[k - 1]) < 0.10 * abs(vals[k - 1]):
            ratios.append(None)
        else:
            ratios.append(abs(slopes[k] / slopes[k - 1]))
    return {"incs": incs, "slopes": slopes, "ratios": ratios}


SETTLING_THRESHOLD = 0.5


def _settling_verdict(vals: list[float], slopes: list[float], ratios: list,
                      order: list[float]) -> tuple[str, str]:
    """Is this ladder quantity settling, and which way is it moving? Two questions.

    Kept apart because they fail independently and one of them survives when the other
    does not: a ratio can be WITHHELD as noise-amplified while the sequence's direction
    is still perfectly readable off the increments' signs.

    A WITHHELD RATIO IS NOT A SMALL RATIO. The first draft let ``None`` fall through a
    ``>= 0.5`` comparison as NaN — which is False — and printed SETTLING for a quantity
    whose ratio had been suppressed as unreadable two lines earlier. That is the
    strongest form of [[instruments-that-cannot-see-the-failure]]: a verdict computed
    from evidence the same function had already declared inadmissible.

    §3.16 saw one reversal and correctly refused to call it a turnover ("one reversal
    is one point"). Whether it REPEATS is a property of the sequence, not of any ratio,
    so it is reported for every quantity either way.
    """
    seq = ", ".join(f"{s:+.1f}" for s in slopes)
    r = ratios[-1] if ratios else None
    if r is None:
        verdict = (f"RATIO WITHHELD — slopes (pp/mm) {seq}, but the denominator "
                   "increment\n        is under 10 % of the value it moved, so the "
                   "latest ratio is noise\n        amplified rather than a reading. "
                   "NO settling verdict is available;\n        what is readable is "
                   "the direction line below.")
    elif len(slopes) > 1 and slopes[-1] * slopes[-2] < 0:
        verdict = (f"SLOPE REVERSED at the newest rung (pp/mm: {seq}).\n        "
                   "A reversal is a turnover only once it repeats; with one of them\n"
                   "        *stopped growing* is claimed and *turned over* is only "
                   "named\n        — the posture §3.14 took on its own saturating dt "
                   "term.")
    elif r >= SETTLING_THRESHOLD:
        verdict = (f"STILL NOT SETTLING — slopes (pp/mm) {seq}, latest ratio "
                   f"{r:.2f}\n        against a null of 1.00. The decay is slower than "
                   "the rungs below\n        it could show. Do not quote this quantity "
                   "at any resolution here.")
    else:
        verdict = f"SETTLING — slopes (pp/mm) {seq}, latest ratio {r:.2f}."

    incs = [vals[k] - vals[k - 1] for k in range(1, len(vals))]
    shown = ", ".join(f"{v:+.2f}" for v in incs)
    flips = [k for k in range(1, len(incs)) if incs[k] * incs[k - 1] < 0]
    if not flips:
        direction = f"MONOTONE across all {len(incs)} increments ({shown} pp)."
    else:
        after = len(incs) - flips[-1]
        direction = (f"reversed at rung {int(order[flips[-1] + 1])}c ({shown} pp), and "
                     f"has held that sign for\n              {after} increment(s) since "
                     "— " + ("TURNED OVER, the repeat §3.16 said it needed."
                             if after >= 2 else "one reversal is still one point."))
    return verdict, direction


def _rung_delta(dx_arm: dict, dt_arm: dict, base: dict, faces) -> list[float]:
    """One rung's dx-only effect, as a percentage OF THE SHIPPED ARM.

    THE DENOMINATOR IS THE SHIPPED ARM, NOT THE PARTNER, and that is not a detail.
    §3.13 published every row of its decomposition — the dt-only rows, the joint
    ladder rows and the matched-dt dx rows — against the shipped arm, which is what
    makes them ADDITIVE: dt-only plus dx-only compares against the joint row because
    all three divide by the same thing. Normalising a rung by its own partner instead
    yields a different number for the same measurement (+34.1 % where §3.13 published
    +31.6 %), and two figures for one cache is a mismatch until someone says why.
    So: same convention, and the 12- and 16-cell rungs reproduce §3.13 exactly, which
    is this mode's own regression check.
    """
    out = [100.0 * (arrival_us(dx_arm, x) - arrival_us(dt_arm, x)) / arrival_us(base, x)
           for x in faces]
    out.append(100.0 * (dx_arm["residual_v_matched"] - dt_arm["residual_v_matched"])
               / base["residual_v_matched"])
    out.append(100.0 * (dx_arm["frac_through_matched"] - dt_arm["frac_through_matched"])
               / base["frac_through_matched"])
    return out


def _dt_ladder(caches: Path) -> int:
    print("MATCHED-dt LADDER: is the residual state converging in `dx`?")
    print("  §3.13 closed with a specific unfinished statement — '16 cells is NOT")
    print("  enough for the residual state' — because residual velocity and")
    print("  mass-through were still GROWING in absolute terms at 16 cells, with two")
    print("  rungs to read increments from. §3.16 added a third at 24 cells and split")
    print("  the claim in two; §3.17 adds a fourth at 32, where the `dx` step HALVES")
    print("  and the statistic has to stop being an increment ratio to survive it.")
    print("  Every rung differences a `dx` arm against a dt PARTNER at the SHIPPED")
    print("  grid running the same substep count, so each row is a MEASURED dx-only")
    print("  effect rather than the difference of two opposing errors (§3.13's rule).\n")

    # Name what is missing rather than dying on the first `open`. A rung is TWO bakes
    # and the expensive one can be hours; a reader who baked half the ladder should be
    # told which half. Refuses either way — a ladder silently one rung short would
    # publish a settling verdict computed from fewer points than its own prose claims.
    wanted = [LADDER_BASELINE] + [n for _, dx, dt, *_ in LADDER for n in (dx, dt)]
    missing = [n for n in dict.fromkeys(wanted) if not (caches / n / "manifest.json").exists()]
    if missing:
        print("  !! missing cache(s): " + ", ".join(missing))
        print("     Every rung of this ladder is a dx arm AND its dt partner; the mode")
        print("     will not report a shorter ladder than the one it describes.")
        return 2

    arms = {LADDER_BASELINE: measure(caches / LADDER_BASELINE)}
    for cells, dx_cache, dt_cache, dx_sub, dt_sub in LADDER:
        for name in (dx_cache, dt_cache):
            if name not in arms:
                arms[name] = measure(caches / name)

    print("ARMS (geometry read from frame 0, before anything moves)")
    for name, r in arms.items():
        _report_arm(r)

    # Same geometry guard `--family` applies: a shared face set is only shared if the
    # arms are the same scenario.
    ref = arms[LADDER_BASELINE]["faces"]
    for r in arms.values():
        if len(r["faces"]) != len(ref) or max(abs(a - b) for a, b in zip(ref, r["faces"])) > 0.5:
            print("\n  !! arms disagree on the stack geometry — they are not the same scenario")
            return 2

    # THE WINDOW GUARD. Arrival times and the MATCHED residual are read off the
    # penetration-front curve and off each arm's own breakout, so neither depends on
    # how long recording continued. `depth_end` and the final-frame residual DO.
    windows = {round(r["window_us"], 6) for r in arms.values()}
    print(f"\nRECORDING WINDOWS: " + ", ".join(
        f"{r['cache']} {r['window_us']:.1f} us" for r in arms.values()))
    if len(windows) > 1:
        print("  !! THE ARMS DO NOT SHARE A WINDOW, so `depth_end` and the final-frame")
        print("     residual are NOT comparable across them and are WITHHELD below.")
        print("     Everything reported is read off the front CURVE (arrival at an x)")
        print("     or at a matched elapsed time after each arm's OWN breakout, and")
        print("     neither depends on the recording length. `heat_conv_dt342` runs a")
        print("     longer window on purpose — its deck header says why, and the")
        print("     alternative (extending its dx partner to match) would have pushed")
        print("     that arm's faster residual into the far wall, PHYSICS §1.1.2.")
    else:
        print("  (all equal — `depth_end` would be comparable, though §3.13 says not to")
        print("   quote it: past the back face it is free flight, not depth.)")

    # Every arm is timed to the BASELINE arm's own measured faces, never to its own.
    # Each arm reads an interface half a lattice pitch inside the true face and that
    # pitch shrinks with the grid, so per-arm faces would put a ~0.05 mm systematic
    # INTO the difference being measured. `_table` makes the same choice for the same
    # reason; using the shipped arm's set here also reproduces §3.13's rows exactly.
    faces = []
    for x in LADDER_INTERFACES:
        near = [f for f in ref if abs(x - f) < 0.5]
        if not near:
            print(f"\n  !! x={x} is not an interface of this stack; the arms are not the deck")
            return 2
        faces.append(near[0])

    hdr = "".join(f"  x={x:<7.1f}" for x in faces)
    print(f"\nTHE dx-ONLY EFFECT AT EACH RUNG (dx arm minus its dt partner, both at the")
    print(f"same substep count — so this is `dx` alone, not `dx` and the clock — as a")
    print(f"percentage of the SHIPPED arm, which is §3.13's convention and keeps the")
    print("decomposition additive)")
    print(f"  {'cells':>6} {'substeps':>18}  {hdr}   {'v_resid':>10} {'through':>10}")
    eff = {}
    base = arms[LADDER_BASELINE]
    for cells, dx_cache, dt_cache, dx_sub, dt_sub in LADDER:
        row = _rung_delta(arms[dx_cache], arms[dt_cache], base, faces)
        eff[cells] = row
        mism = "" if dx_sub == dt_sub else f"  ({_pct(dt_sub, dx_sub):+.1f} % mismatch)"
        print(f"  {arms[dx_cache]['cells_across']:6.1f} {f'{dx_sub} vs {dt_sub}':>18}  "
              + "".join(f"  {v:+7.2f} % " for v in row[:len(faces)])
              + f"   {row[-2]:+9.1f} % {row[-1]:+9.1f} %{mism}")
    print("  cells are MEASURED off the seeded lattice, not read from the deck label.")
    print("  The 12- and 16-cell rows REPRODUCE §3.13's published table exactly; that")
    print("  is this mode's regression check, not a coincidence to be admired.")
    print("  The 24-cell rung is the family's FIRST EXACT pair (342 vs 342); §3.13's")
    print("  16-cell rung carries a 0.9 % substep mismatch, the closest that deck grid")
    print("  allowed. An exact rung is worth more than its position suggests.")

    # THE POINT OF THE THIRD RUNG. With two rungs an increment can be computed but not
    # read: there is one of them, and one number cannot say whether a sequence is
    # settling. Three rungs give two increments and a ratio between them.
    #
    # AND THE POINT OF THE FOURTH — plus the trap it walks into. 32 cells is dx =
    # 0.09375, which is HALF the step the 16- and 24-cell rungs were spaced by. §3.16
    # read its ratio against a null of 1.00 and said so explicitly ("over equal steps a
    # first-order error gives 1.00"); the same first-order error over a half step gives
    # 0.50. Publishing a new ratio against the old null would be one statistic under
    # two conventions — the exact defect §3.16 corrected in §3.15's headline, so it is
    # applied here rather than restated.
    #
    # THE FIX IS THE SLOPE, not a different ladder. If the underlying discretisation
    # error is first order, value(h) = value* + C*h, so a rung's effect (measured
    # against the SHIPPED grid at matched dt) is e(h) = C*(h - h_shipped): LINEAR in h
    # with slope C. Then increment/step = C is constant at ANY spacing, and the ratio
    # of consecutive slopes has a null of 1.00 at every rung, equal steps or not.
    # Where the steps ARE equal the slope ratio is identically the increment ratio, so
    # §3.16's 0.65 is REPRODUCED by the new column rather than superseded by it — which
    # is what makes this a change of convention and not a change of answer.
    names = [f"x={x:.0f}" for x in faces] + ["v_resid", "through"]
    order = [c for c, *_ in LADDER]
    dxs = [arms[dx_cache]["dx"] for _, dx_cache, *_ in LADDER]
    steps = [dxs[k - 1] - dxs[k] for k in range(1, len(dxs))]
    equal = max(steps) - min(steps) < 1e-9
    print("\nIS IT SETTLING? — the increment between rungs, DIVIDED BY THE dx STEP THAT")
    print("PRODUCED IT. The rungs are spaced "
          + ", ".join(f"{s:.5f}" for s in steps) + " mm in dx"
          + (", equal throughout." if equal else " — NOT EQUAL."))
    if not equal:
        print("A raw increment ratio therefore has a DIFFERENT NULL AT EVERY RUNG (1.00")
        print("where the steps are equal, h_new/h_old where they are not), and quoting")
        print("one against another compares two conventions. The published statistic is")
        print("the SLOPE, increment/step in pp per mm of dx: a first-order error gives a")
        print("CONSTANT slope at any spacing, and consecutive slopes have a null of 1.00.")
        print("Over equal steps the slope ratio IS the increment ratio, so §3.16's")
        print("figures are reproduced by this column, not replaced by it.")
    print(f"  {'quantity':>10}  " + "  ".join(f"{int(c):>7}c" for c in order)
          + "  | " + "  ".join(f"{'s' + str(int(b)) + 'c':>8}" for b in order[1:])
          + "  | " + "  ".join(f"{'r' + str(int(b)) + 'c':>6}" for b in order[2:]))
    suppressed = False
    slopes_by_q, ratios_by_q, incs_by_q = {}, {}, {}
    for i, q in enumerate(names):
        vals = [eff[c][i] for c in order]
        # A RATIO WITH A NEAR-ZERO DENOMINATOR IS NOT A SMALL RATIO, IT IS NOISE
        # AMPLIFIED. Two consecutive rungs that happen to land on top of each other
        # make the next ratio arbitrarily large, and printing it beside the real ones
        # invites reading a divergence off an accident. Withheld, loudly, rather than
        # printed with a caveat nobody reads — the same posture §3.13 took at x=160.
        # The test is on the INCREMENT against the value it moved, so it does not
        # change meaning when the increment is rescaled into a slope.
        tab = _slope_table(vals, steps)
        slopes, ratios = tab["slopes"], tab["ratios"]
        slopes_by_q[q], ratios_by_q[q] = slopes, ratios
        incs_by_q[q] = tab["incs"]
        cells = []
        for k, r in enumerate(ratios, start=1):
            if r is None:
                cells.append("   (—)")
                suppressed = True
            else:
                cells.append(f"{r:6.2f}" + ("*" if slopes[k] * slopes[k - 1] < 0 else " "))
        print(f"  {q:>10}  " + "  ".join(f"{v:+7.2f} " for v in vals)
              + " | " + "  ".join(f"{v:+8.1f}" for v in slopes)
              + " | " + " ".join(cells))
    if suppressed:
        print("  (—) that rung's increment is under 10 % of the value it moved, so the")
        print("      ratio is a near-zero denominator rather than a reading. Withheld.")
    print("  (*) marks a slope that reversed sign against the one before it.")
    print("  Slopes are pp per mm of dx; ratios are consecutive slopes, null 1.00.")
    if not equal:
        print("\n  RECONCILIATION with the raw increment ratio §3.16 published, so the two")
        print("  can be told apart on sight. Each rung's raw-ratio null is h_new/h_old:")
        for k in range(1, len(steps)):
            print(f"    r{int(order[k + 1])}c: steps {steps[k - 1]:.5f} -> {steps[k]:.5f} mm, "
                  f"raw-ratio null {steps[k] / steps[k - 1]:.2f}, slope-ratio null 1.00"
                  + ("   (identical here)" if abs(steps[k] - steps[k - 1]) < 1e-9 else ""))
        # AND WHETHER IT ACTUALLY MATTERED, computed rather than asserted. The
        # verdict block calls a ratio >= 0.5 "not settling", so a rung where the raw
        # and slope ratios land on opposite sides of 0.5 is one where the OLD
        # statistic would have published the OPPOSITE conclusion from the same caches.
        print("  Raw vs slope at the unequal rung, and whether the choice changes the")
        print("  verdict (the block below calls >= 0.50 'not settling'):")
        k = len(steps) - 1
        for q in names:
            incs, ratios = incs_by_q[q], ratios_by_q[q]
            if ratios[k - 1] is None:
                print(f"    {q:>10}  slope ratio withheld — no verdict either way")
                continue
            raw = abs(incs[k] / incs[k - 1])
            flip = (raw >= 0.5) != (ratios[k - 1] >= 0.5)
            print(f"    {q:>10}  raw {raw:5.2f}  slope {ratios[k - 1]:5.2f}   "
                  + ("<-- OPPOSITE VERDICTS from the same caches" if flip else "same verdict"))

    print("\n  A NOTE ON WHAT §3.13's '0.20 / 0.43 / 0.55' WERE, because they are NOT")
    print("  the column above. With two rungs there is one increment, so the only ratio")
    print("  available is increment-over-VALUE; with three there is a genuine")
    print("  increment-over-INCREMENT. They are different quantities and must not be")
    print("  read as a sequence. For the record this run reproduces §3.13's:")
    for i, q in enumerate(names):
        vals = [eff[c][i] for c in order]
        if vals[0]:
            print(f"      {q:>8}  increment/value at 16 cells "
                  f"{abs((vals[1] - vals[0]) / vals[0]):.2f}")
    print("  Neither column is a convergence ORDER. §3.8 measured the observed order in")
    print("  this family as ill-conditioned (it swings ~0.7 to ~5 with the matching")
    print("  point). Read the ratios; quote no exponent.")

    # The verdict is COMPUTED, not typed. §3.8's table went two rebakes stale because
    # a measurement was restated in prose; a milestone whose whole point is a third
    # data point must not hardcode what that point said.
    print("\n  THE VERDICT ON §3.13's CLOSING CLAIM — '16 cells is NOT enough for the")
    print("  RESIDUAL STATE'. It named two quantities together. They do not agree.")
    print("  Read on the SLOPE, which is the only column that means the same thing at")
    print("  every rung of an unequally-spaced ladder.")
    for i, q in enumerate(names):
        if q not in ("v_resid", "through"):
            continue
        vals = [eff[c][i] for c in order]
        verdict, direction = _settling_verdict(
            vals, slopes_by_q[q], ratios_by_q[q], order)
        print(f"    {q:>8}: {verdict}")
        print(f"    {'':>8}  direction: {direction}")
    print("    So 'the residual state' was one phrase over two quantities that behave")
    print("    differently, and two rungs could not tell them apart. That was the third")
    print("    rung's deliverable, and the fourth tests whether it survives a rung that")
    print("    is NOT equally spaced — NOT a convergence claim for either of them.")

    print("\n  WHAT THIS STILL CANNOT REACH — unchanged from §3.13, and not narrowed:")
    print("    * the `dx` effect at the SHIPPED arm's 110 substeps. `dt = min(deck_dt,")
    print("      cfl_dt)` only ever LOWERS dt, so fine `dx` at coarse `dt` is")
    print("      unreachable by construction. Every row above is the dx effect at its")
    print("      OWN rung's substep count, never at 110.")
    print("    * the `dx` x `dt` interaction. Each row is clean at its own dt; the")
    print("      TREND between rows mixes the grid ladder with any interaction term.")
    print("      A third rung gives a second increment, not a second variable.")
    return 0


# --- milestone 20: did the CFL breach contaminate the 32-cell rung? ------------
#
# `heat_conv_dx094` is the first deck here to breach its CFL audit (1.60x, §3.17). A
# breach is not a divergence — it is a fraction of the CFL=0.3 safety factor, so 1.60x
# is a Courant number of 0.48 against a limit near 1 — and the bake is finite with
# every guard at zero. "Not obviously wrong" is exactly the state that needs a
# measurement rather than a paragraph.
#
# THE PAIR IS dt-ONLY AT FIXED dx, so it needs no partner: it IS the pair. What it can
# show is whether the breached arm is UNUSUALLY dt-sensitive, which requires a
# reference — ordinary `dt` error is real and large in this family. The reference rows
# are the dt-only pairs at the SHIPPED dx over comparable refinement ratios.
#
# The floors are stated here rather than chosen after the numbers: §3.13 measured
# run-to-run scatter on the front-curve percentile readings at <= 0.0024 %, and the
# repo's aggregate figure floor is 0.11 % (residual velocity, mass-through).
BREACH_ARM, BREACH_CONTROL = "heat_conv_dx094", "heat_conv_dx094_dt770"
BREACH_REFERENCE = [
    ("heat_vs_composite", "heat_conv_dt_mid", 110, 171),
    ("heat_conv_dt_fine", "heat_conv_dt342", 230, 342),
    ("heat_conv_dt342", "heat_conv_dt456", 342, 456),
]
FLOOR_CURVE, FLOOR_AGGREGATE = 0.0024, 0.11


def _dt_pair(coarse: dict, fine: dict, faces) -> list[float]:
    """A dt-only pair's effect: the finer arm against the coarser, in percent."""
    out = [_pct(arrival_us(fine, x), arrival_us(coarse, x)) for x in faces]
    out.append(_pct(fine["residual_v_matched"], coarse["residual_v_matched"]))
    out.append(_pct(fine["frac_through_matched"], coarse["frac_through_matched"]))
    return out


def _breach_control(caches: Path) -> int:
    print("DID THE CFL BREACH CONTAMINATE THE 32-CELL RUNG? (PHYSICS §3.17)")
    print("  `heat_conv_dx094` breached its audit 1.60x. This mode differences it")
    print("  against a same-`dx`, finer-`dt` control and puts that beside the SAME")
    print("  family's known `dt` sensitivity at the shipped grid, because ordinary")
    print("  `dt` error here is several percent and a bare difference proves nothing.")
    print("  A dt-only pair measures a DIFFERENCE, never a truth: this cannot show the")
    print("  arm is correct, only whether it responds like its own family.\n")

    wanted = [BREACH_ARM, BREACH_CONTROL] + [n for a, b, *_ in BREACH_REFERENCE
                                             for n in (a, b)]
    missing = [n for n in dict.fromkeys(wanted)
               if not (caches / n / "manifest.json").exists()]
    if missing:
        print("  !! missing cache(s): " + ", ".join(missing))
        return 2

    arms = {n: measure(caches / n) for n in dict.fromkeys(wanted)}
    for r in arms.values():
        _report_arm(r)

    ref = arms[BREACH_ARM]["faces"]
    faces = []
    for x in LADDER_INTERFACES:
        near = [f for f in ref if abs(x - f) < 0.5]
        if not near:
            print(f"\n  !! x={x} is not an interface of this stack")
            return 2
        faces.append(near[0])

    hdr = "".join(f"  x={x:<7.1f}" for x in faces)
    print(f"\nTHE dt-ONLY EFFECT (finer arm against coarser, same `dx` in every row)")
    print(f"  {'pair':>34} {'ratio':>7}  {hdr}   {'v_resid':>10} {'through':>10}")
    rows = []
    for coarse, fine, cs, fs in BREACH_REFERENCE:
        row = _dt_pair(arms[coarse], arms[fine], faces)
        rows.append((f"{cs}->{fs} @ dx={arms[coarse]['dx']:.4f}", fs / cs, row))
    control = _dt_pair(arms[BREACH_ARM], arms[BREACH_CONTROL], faces)
    for label, ratio, row in rows:
        print(f"  {label:>34} {ratio:>6.2f}x  "
              + "".join(f"  {v:+7.2f} % " for v in row[:len(faces)])
              + f"   {row[-2]:+9.2f} % {row[-1]:+9.2f} %")
    control_label = f"456->770 @ dx={arms[BREACH_ARM]['dx']:.4f}"
    print(f"  {control_label:>34} {770 / 456:>6.2f}x  "
          + "".join(f"  {v:+7.2f} % " for v in control[:len(faces)])
          + f"   {control[-2]:+9.2f} % {control[-1]:+9.2f} %   <-- THE BREACHED ARM")

    # THE VERDICT, COMPUTED. "Unusually sensitive" needs a number: the control is
    # compared against the LARGEST reference row per quantity, scaled to a common
    # refinement ratio, because the reference ratios are not identical to the
    # control's. Scaling assumes the dt error is roughly linear in the refinement,
    # which §3.14 measured SATURATING — so this scaling is CONSERVATIVE (it inflates
    # the reference's expectation, making an indictment harder, not easier).
    names = [f"x={x:.0f}" for x in faces] + ["v_resid", "through"]
    print("\nIS THE BREACHED ARM UNUSUALLY dt-SENSITIVE? — the control against the")
    print("largest reference row for the same quantity, each scaled to the control's")
    print(f"{770 / 456:.2f}x refinement. Scaling linearly OVERSTATES the reference")
    print("(§3.14 measured the dt term saturating), so it is the conservative test.")
    print(f"  {'quantity':>10} {'control':>10} {'worst ref':>11} {'ref scaled':>11} "
          f"{'control/ref':>12}   floor")
    flagged = []
    for i, q in enumerate(names):
        floor = FLOOR_AGGREGATE if q in ("v_resid", "through") else FLOOR_CURVE
        worst, worst_ratio = 0.0, 1.0
        for _, ratio, row in rows:
            if abs(row[i]) > abs(worst):
                worst, worst_ratio = row[i], ratio
        scaled = worst * (770 / 456 - 1.0) / (worst_ratio - 1.0)
        rel = abs(control[i]) / abs(scaled) if scaled else float("inf")
        note = "" if abs(control[i]) > floor else "  (under the floor — not a reading)"
        if abs(control[i]) > floor and rel > 2.0:
            flagged.append(q)
            note = "  <-- MORE THAN 2x ITS FAMILY"
        print(f"  {q:>10} {control[i]:+10.2f} {worst:+11.2f} {scaled:+11.2f} "
              f"{rel:>12.2f}   {floor:.4f} %{note}")
    print()
    if flagged:
        print("  VERDICT: " + ", ".join(flagged) + " respond to `dt` more than twice as")
        print("  strongly as this family does at the shipped grid. The 32-cell rung's")
        print("  reading of those quantities is NOT safe to quote (§3.17).")
    else:
        print("  VERDICT: every quantity moves within twice its family's own `dt`")
        print("  sensitivity, so the 1.60x breach is behaving like ordinary `dt` error")
        print("  rather than like an instability. That is evidence the rung is")
        print("  quotable; it is NOT evidence that the arm is converged, which is a")
        print("  different question and the ladder's own answer to it is 'no'.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache_dirs", type=Path, nargs="*")
    ap.add_argument("--family", action="store_true", help="the heat_conv_* ladder")
    ap.add_argument("--dt-ladder", action="store_true",
                    help="the matched-dt dx ladder, 12 / 16 / 24 / 32 cells")
    ap.add_argument("--breach-control", action="store_true",
                    help="milestone 20: did the 32-cell arm's 1.60x CFL breach move "
                         "its readings? A dt-only pair at fixed dx, against the same "
                         "family's known dt sensitivity")
    ap.add_argument("--caches", type=Path, default=Path("caches"))
    ap.add_argument("--sensitivity", action="store_true", help="re-read at three percentiles")
    ap.add_argument("--plot", type=Path, help="write the front-vs-time curves to a PNG")
    args = ap.parse_args(argv)

    if args.dt_ladder:
        return _dt_ladder(args.caches)

    if args.breach_control:
        return _breach_control(args.caches)

    if args.family:
        dirs = [args.caches / d for d in FAMILY]
    elif len(args.cache_dirs) >= 2:
        dirs = list(args.cache_dirs)
    else:
        ap.error("give two or more cache dirs (coarsest first), or --family")

    missing = [d for d in dirs if not (d / "manifest.json").exists()]
    if missing:
        raise SystemExit("missing cache(s): " + ", ".join(str(d) for d in missing))

    arms = [measure(d) for d in dirs]
    arms.sort(key=lambda r: r["cells_across"])

    print("ARMS (geometry read from frame 0, before anything moves)")
    for r in arms:
        _report_arm(r)

    ref = arms[0]["faces"]
    for r in arms[1:]:
        if len(r["faces"]) != len(ref) or max(abs(a - b) for a, b in zip(ref, r["faces"])) > 0.5:
            print("\n  !! arms disagree on the stack geometry — they are not the same scenario")
            return 2

    _table(arms)
    if args.sensitivity:
        _sensitivity(dirs)
    if args.plot:
        _plot(arms, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
