#!/usr/bin/env python3
"""Measure the steady penetration velocity u from a cache.

Language-neutral helper (CLAUDE.md §3): reads the cache per docs/CACHE_FORMAT.md
and imports nothing from the solver or the visualizer.

    python tools/measure_penetration.py caches/sweep_tungsten_v3500
    python tools/measure_penetration.py caches/sweep_tungsten_v3500 --plot out.png

WHAT IT MEASURES. During steady eroding penetration the penetrator's leading edge
sits at the crater bottom and advances at the penetration velocity u, while its
tail still flies at the impact velocity v. So u is the slope of the live
penetrator's leading-edge position over time, and u/v is the dimensionless number
the hydrodynamic (Tate-Alekseevskii) model actually predicts:

    u/v  ->  1 / (1 + sqrt(rho_t/rho_p))     as strength becomes negligible

This deliberately does NOT report penetration depth. Depth is cumulative and rides
on the whole erosion history, which has a partly-numerical component (crushed
particles latch `damage` and leave the live set). u is an instantaneous rate.

HOW IT AVOIDS BAKING IN ASSUMPTIONS:

  * The penetrator is identified as whatever is MOVING at frame 0 -- not by a
    hardcoded material id, and not by an x-band. Armor is seeded at rest.
  * `v` is measured from frame 0 rather than passed in, so the tool cannot be
    told the wrong answer.
  * The fit window is derived from the erosion curve itself (the span over which
    the penetrator is consumed from 80% to 20% of its initial live count), not
    from hardcoded frame numbers or x positions. Different velocities erode on
    wildly different clocks; a fixed window would silently measure the transient
    on one deck and the tail on another.
  * The leading edge is a high percentile, not the max, so one stray particle
    cannot define the front.
  * R^2 is reported. A steady phase is a straight line; if the fit is not straight
    the premise of the measurement failed and the number should not be used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The leading edge of the penetrator, as a percentile of live penetrator x. Not
# the max: a single particle spat forward off the crater lip would define the
# front and add noise the slope cannot survive.
TIP_PERCENTILE = 99.5

# Fit over the span where the penetrator is consumed from this fraction of its
# initial live count down to (1 - this). Brackets the steady phase: skips the
# nose-consumption transient at the head and the rod-exhausted stall at the tail.
EROSION_WINDOW = (0.80, 0.20)


def load_cache(cache_dir: Path):
    """Memory-map frames.bin per CACHE_FORMAT §3 and return (manifest, frames)."""
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("dtype") != "float32":
        raise SystemExit(f"unsupported dtype {manifest.get('dtype')!r}; expected float32")
    pc = manifest["particle_count"]
    fc = manifest["frame_count"]
    stride = len(manifest["attributes"])
    frames = np.memmap(
        cache_dir / "frames.bin", dtype="<f4", mode="r", shape=(fc, pc, stride)
    )
    return manifest, frames


def measure(cache_dir: Path, verbose: bool = True):
    manifest, frames = load_cache(cache_dir)
    attrs = manifest["attributes"]
    # Read the layout from the manifest; never hardcode column offsets (root §4).
    col = {a: i for i, a in enumerate(attrs)}
    for need in ("pos_x", "vel_mag", "damage", "material_id"):
        if need not in col:
            raise SystemExit(f"cache lacks required attribute {need!r} (has: {attrs})")

    f0 = np.asarray(frames[0])
    mat = f0[:, col["material_id"]]

    # The penetrator is what is moving at t=0. Armor is seeded at rest, so this
    # needs no material table and no deck.
    moving = f0[:, col["vel_mag"]] > 1.0  # mm/ms
    if not moving.any():
        raise SystemExit("nothing is moving at frame 0 — cannot identify the penetrator")
    proj_ids = np.unique(mat[moving])
    if len(proj_ids) != 1:
        raise SystemExit(f"expected exactly one moving material at frame 0, got {proj_ids}")
    proj_id = float(proj_ids[0])
    proj_name = manifest.get("materials", {}).get(str(int(proj_id)), f"id={proj_id:.0f}")

    is_proj = mat == proj_id
    v = float(np.median(f0[is_proj, col["vel_mag"]]))

    # --- sweep the frames -------------------------------------------------
    fc = manifest["frame_count"]
    frame_dt_us = manifest["frame_dt"] * 1e6
    t = np.arange(fc) * frame_dt_us
    x_tip = np.full(fc, np.nan)
    n_live = np.zeros(fc, dtype=int)
    for f in range(fc):
        fr = np.asarray(frames[f])
        live = is_proj & (fr[:, col["damage"]] < 0.5)
        n_live[f] = int(live.sum())
        if n_live[f] > 0:
            x_tip[f] = np.percentile(fr[live, col["pos_x"]], TIP_PERCENTILE)

    n0 = n_live[0]
    if n0 == 0:
        raise SystemExit("no live penetrator particles at frame 0")
    frac = n_live / n0

    # --- window: derived from the erosion curve, not hardcoded ------------
    hi, lo = EROSION_WINDOW
    in_window = (frac <= hi) & (frac >= lo) & np.isfinite(x_tip)
    idx = np.flatnonzero(in_window)
    if len(idx) < 5:
        return {
            "cache": cache_dir.name, "material": proj_name, "v": v,
            "u": float("nan"), "u_over_v": float("nan"), "r2": float("nan"),
            "n_fit": len(idx), "frac_end": float(frac[-1]),
            "note": (f"penetrator eroded only to {frac[-1]:.0%} of its initial live "
                     f"count — never reached the {hi:.0%}-{lo:.0%} steady window "
                     f"({len(idx)} usable frames). Deck needs more total_time, or "
                     f"the impact is below the erosion threshold."),
        }

    # --- the half-space premise, checked ---------------------------------
    # u is only the penetration velocity while the penetrator is still INSIDE the
    # target. If it perforates, the leading edge stops being a crater bottom
    # advancing at u and becomes a free residual flying at ~v. That still fits a
    # near-perfect straight line, so R^2 will happily bless it -- this check is
    # the only thing standing between a perforating deck and a confident wrong
    # number. (Measured on apfsds_vs_rha, a 40 mm plate: u/v reads 0.729 with
    # R^2=0.999, above the 0.600 hydrodynamic ceiling that strength can only
    # push it below. Physically impossible, beautifully fitted.)
    # Back face is taken from frame 0, before any armor moves or spalls.
    if (~is_proj).any():
        back_face = float(f0[~is_proj, col["pos_x"]].max())
        reached = np.nanmax(x_tip[idx])
        if reached >= back_face:
            return {
                "cache": cache_dir.name, "material": proj_name, "v": v,
                "u": float("nan"), "u_over_v": float("nan"), "r2": float("nan"),
                "n_fit": len(idx), "frac_end": float(frac[-1]),
                "note": (f"penetrator reached x={reached:.1f} mm but the target's back "
                         f"face is at x={back_face:.1f} mm — it PERFORATED, so the "
                         f"leading edge is a free residual, not a penetration front. "
                         f"u is undefined here. This measurement needs a semi-infinite "
                         f"target (see the sweep_* decks)."),
            }

    tw, xw = t[idx], x_tip[idx]
    slope, intercept = np.polyfit(tw, xw, 1)
    pred = slope * tw + intercept
    ss_res = float(((xw - pred) ** 2).sum())
    ss_tot = float(((xw - xw.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    u = slope * 1000.0  # mm/us -> mm/ms == m/s

    out = {
        "cache": cache_dir.name, "material": proj_name, "v": v, "u": u,
        "u_over_v": u / v, "r2": r2, "n_fit": len(idx),
        "frac_end": float(frac[-1]),
        "window_us": (float(tw[0]), float(tw[-1])), "note": "",
    }
    if verbose:
        print(f"cache      : {out['cache']}")
        print(f"penetrator : {proj_name} (identified as the only material moving at t=0)")
        print(f"v          : {v:.1f} m/s   (measured from frame 0, not supplied)")
        print(f"u          : {u:.1f} m/s   (fit over {len(idx)} frames, "
              f"t={tw[0]:.1f}-{tw[-1]:.1f} us, erosion {hi:.0%}->{lo:.0%})")
        print(f"u/v        : {u/v:.4f}")
        print(f"R^2        : {r2:.5f}   ({'straight — steady phase' if r2 > 0.99 else 'NOT STRAIGHT — the steady-phase premise failed, do not use'})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--json", action="store_true", help="emit the result as JSON")
    args = ap.parse_args(argv)
    if not (args.cache_dir / "manifest.json").is_file():
        raise SystemExit(f"{args.cache_dir} is not a cache dir (no manifest.json)")
    out = measure(args.cache_dir, verbose=not args.json)
    if args.json:
        print(json.dumps(out))
    if out["note"]:
        print(f"\nWARNING: {out['note']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
