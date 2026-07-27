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


FAMILY = ("heat_vs_composite", "heat_conv_dx250", "heat_conv_dx188")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache_dirs", type=Path, nargs="*")
    ap.add_argument("--family", action="store_true", help="the heat_conv_* ladder")
    ap.add_argument("--caches", type=Path, default=Path("caches"))
    ap.add_argument("--sensitivity", action="store_true", help="re-read at three percentiles")
    ap.add_argument("--plot", type=Path, help="write the front-vs-time curves to a PNG")
    args = ap.parse_args(argv)

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
