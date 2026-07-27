#!/usr/bin/env python3
"""Measure what actually reaches a domain wall, from baked caches.

Solver-free by construction (tools/ depends on neither half — root §3): it reads
the cache contract and nothing else.

    python tools/measure_wall_contact.py caches/apfsds_vs_rha
    python tools/measure_wall_contact.py --all
    python tools/measure_wall_contact.py caches/apfsds_vs_era_oblique --json

WHY THIS EXISTS, AND WHAT QUESTION IT IS ALLOWED TO ANSWER.

The repo has carried two incompatible statements of the boundary defect:

  * README: "domain/BC work so oblique-deck debris **never reaches a wall**."
  * PHYSICS §1.1: "Late in a bake, spall spray does reach the top/bottom walls
    and slide along them... the artifact is confined to the far-field debris."

They cannot both be the bar. The README's is also stricter than the seeding
design permits — every deck lays its armor slabs **wall to wall** on purpose, so
that the slip wall's mirror makes them a plate that continues beyond the frame
(§1.1). Material at a wall is not the defect; a wall that isn't there was
(§1.1.1). The bar this tool measures against is therefore the weaker, correct
one:

    **No wall-reflected momentum contaminates a quoted figure, inside that
    figure's measurement window.**

WHY A NAIVE BAND COUNT WOULD BE USELESS. "Count particles within 3 cells of a
wall" fires on every deck at frame 0, because that is where the armor is seeded.
An instrument that reports the same alarm on a healthy deck and a sick one has
not measured anything — the repo's most-repeated defect, in a new costume. So a
particle is only ever counted here if it **started well clear of that wall and
travelled to it**: `START_KEEPOUT` mm at frame 0, `ARRIVE` mm at some later
frame. Seeded-at-the-wall armor is excluded by construction, not by tuning.

THE TWO THRESHOLDS, AND WHY THEY ARE IN MILLIMETRES.

`_grid_op`'s slip band is **3 cells** wide, but the cache does not record `dx`
(it is not in the v3 manifest), and this tool may not read a deck. So both
thresholds are fixed lengths chosen to bracket every deck in the repo rather
than derived per-deck:

  * `ARRIVE = 1.2 mm` — at least 3 cells on every current deck (dx runs
    0.1875..0.3906, so 3dx runs 0.56..1.17 mm). On the finer decks it is
    *generous*, i.e. it counts approaches that the wall band would not yet have
    touched. Erring that way is the right sign for a screen.
  * `START_KEEPOUT = 3.0 mm` — about 4x the widest seeding inset in the repo
    (`_seed` insets slabs 2 cells, so 0.375..0.78 mm). Wide enough that no
    seeded-at-the-wall particle is ever eligible.

Both are swept (`ARRIVE_SWEEP`, and `--keepout`) so that no single magic number
is load-bearing: if the verdict changes across the sweep, the verdict is the
threshold, not the physics. **If a future deck runs dx > 0.4 mm, `ARRIVE` stops
covering its band and must be revisited** — which is the standing argument for
putting `dx` in the manifest.

WHAT IT CANNOT SEE — state these before quoting it.

  * **Transmitted impulse.** Excluding wall-touched particles measures *direct
    participation* only. A particle that bounced off a wall and then shoved its
    neighbour through the grid leaves its fingerprint on a particle this tool
    calls clean. Every contamination figure here is therefore a **lower bound**.
  * **Sub-frame excursions.** It sees only the frames the solver dumped. A
    particle that dips into the band and back out between two dumps is
    invisible, and frame-cadence aliasing has already bitten this repo once
    (§3.9). Only a bake-time audit sees every substep.
  * **Anything about whether the reflection mattered.** "Arrived at a wall" is
    not "corrupted a figure". That is what the contamination block is for, and
    even it only bounds the direct part.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# A particle must START at least this far from a wall to be eligible to "arrive"
# at it. Excludes wall-to-wall seeded armor by construction (see the docstring).
START_KEEPOUT = 3.0  # mm

# Reaching within this of a wall counts as arrival. >= 3 cells on every deck.
ARRIVE = 1.2  # mm

# Swept so the verdict's threshold-sensitivity is visible rather than assumed.
ARRIVE_SWEEP = (0.6, 1.2, 2.4, 5.0)  # mm

# Same rule measure_penetration.py uses: the projectile is whatever moves at
# t=0, since armor is seeded at rest. No material table, no deck, no x-band.
MOVING = 1.0  # mm/ms

WALLS = ("x_lo", "x_hi", "y_lo", "y_hi")


class Cache:
    """Cache reader per docs/CACHE_FORMAT.md §3."""

    def __init__(self, d: Path):
        self.d = d
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        self.m = m
        if m.get("dtype") != "float32":
            raise SystemExit(f"{d}: unsupported dtype {m.get('dtype')!r}")
        self.attrs = list(m["attributes"])
        self.n_p = int(m["particle_count"])
        self.n_f = int(m["frame_count"])
        self.dt_us = float(m["frame_dt"]) * 1e6
        self.dom = m["domain"]
        self.names = {int(k): v for k, v in m.get("materials", {}).items()}
        self.stride = len(self.attrs)
        # Read the layout from the manifest; never hardcode column offsets (§4).
        self.col = {a: i for i, a in enumerate(self.attrs)}
        for need in ("pos_x", "pos_y", "vel_mag", "damage", "material_id"):
            if need not in self.col:
                raise SystemExit(f"{d}: cache lacks {need!r} (has {self.attrs})")
        self.frames = np.memmap(d / "frames.bin", dtype="<f4", mode="r",
                                shape=(self.n_f, self.n_p, self.stride))

    def wall_dist(self, fr: np.ndarray) -> dict:
        """Distance from every particle to each of the four walls, in mm."""
        x, y = fr[:, self.col["pos_x"]], fr[:, self.col["pos_y"]]
        return {
            "x_lo": x - self.dom["xmin"], "x_hi": self.dom["xmax"] - x,
            "y_lo": y - self.dom["ymin"], "y_hi": self.dom["ymax"] - y,
        }


def measure(cache_dir: Path, keepout: float = START_KEEPOUT,
            stride: int = 1) -> dict:
    c = Cache(cache_dir)
    cv, cd, cm = c.col["vel_mag"], c.col["damage"], c.col["material_id"]

    f0 = np.asarray(c.frames[0])
    mats = np.rint(f0[:, cm]).astype(int)
    # Populations labelled at FRAME 0 and followed by index — the contract fixes
    # particle count and persists particles, which makes an index a durable
    # material label. Labelling by final position would silently redefine them.
    proj = f0[:, cv] > MOVING
    armor = ~proj

    d0 = c.wall_dist(f0)
    eligible = {w: d0[w] > keepout for w in WALLS}
    # Eligibility for a SWEPT radius has to clear that radius too. With
    # keepout=3.0 and D=5.0, a particle seeded 3.5 mm out is "within 5 mm" at
    # frame 0 having travelled nowhere — the frame-0 false positive the keepout
    # exists to prevent, walking back in through the sweep. So a swept count
    # requires a start clear of BOTH. (The counts are therefore not monotone in
    # D: a wider radius also demands a more distant start.)
    el_D = {w: {D: d0[w] > max(keepout, D) for D in ARRIVE_SWEEP} for w in WALLS}

    # Per wall: first frame each eligible particle came within ARRIVE, and the
    # closest any eligible particle ever got (the margin on a clean deck, which
    # is far more informative than a zero count).
    first = {w: np.full(c.n_p, -1, dtype=np.int32) for w in WALLS}
    ever = {w: {D: np.zeros(c.n_p, dtype=bool) for D in ARRIVE_SWEEP}
            for w in WALLS}
    closest = {w: float(d0[w][eligible[w]].min()) if eligible[w].any()
               else float("inf") for w in WALLS}
    closest_t = {w: 0.0 for w in WALLS}
    v_at_arrival = {w: 0.0 for w in WALLS}

    frame_ix = range(0, c.n_f, stride)
    for f in frame_ix:
        fr = np.asarray(c.frames[f])
        d = c.wall_dist(fr)
        for w in WALLS:
            el = eligible[w]
            if not el.any():
                continue
            dw = d[w]
            for D in ARRIVE_SWEEP:
                ever[w][D] |= el_D[w][D] & (dw < D)
            fresh = el & (dw < ARRIVE) & (first[w] < 0)
            if fresh.any():
                first[w][fresh] = f
                v_at_arrival[w] = max(v_at_arrival[w],
                                      float(fr[fresh, cv].max()))
            near = float(dw[el].min())
            if near < closest[w]:
                closest[w], closest_t[w] = near, f * c.dt_us

    touched = np.zeros(c.n_p, dtype=bool)
    for w in WALLS:
        touched |= ever[w][ARRIVE]

    walls_out = {}
    for w in WALLS:
        arr = first[w] >= 0
        n = int(arr.sum())
        by_mat = {}
        if n:
            for mid in np.unique(mats[arr]):
                sel = arr & (mats == mid)
                by_mat[c.names.get(int(mid), f"id={mid}")] = {
                    "n": int(sel.sum()),
                    "first_us": float(first[w][sel].min() * c.dt_us),
                    # Was it already spalled debris when it got there, or intact
                    # material? Debris at a wall is the artifact §1.1 tolerates;
                    # intact structure at a wall is a different claim.
                    "frac_damaged_at_end": float(
                        np.asarray(c.frames[-1])[sel, cd].mean()),
                }
        walls_out[w] = {
            "n_eligible": int(eligible[w].sum()),
            "n_arrived": n,
            "frac_arrived": n / max(int(eligible[w].sum()), 1),
            "first_us": float(first[w][arr].min() * c.dt_us) if n else None,
            "closest_mm": closest[w],
            "closest_us": closest_t[w],
            "max_v_at_arrival": v_at_arrival[w] if n else 0.0,
            "by_material": by_mat,
            "sweep": {D: int(ever[w][D].sum()) for D in ARRIVE_SWEEP},
        }

    out = {
        "cache": cache_dir.name,
        "domain": c.dom,
        "n_particles": c.n_p,
        "n_frames": c.n_f,
        "total_us": c.n_f * c.dt_us,
        "stride": stride,
        "keepout_mm": keepout,
        "arrive_mm": ARRIVE,
        "walls": walls_out,
        "n_touched": int(touched.sum()),
        "contamination": _contamination(c, proj, armor, touched),
        "symmetry": _symmetry(c, mats),
    }
    return out


def _contamination(c: Cache, proj, armor, touched) -> dict:
    """Recompute the repo's headline aggregates with wall-touched particles
    excluded. The delta is the contamination — a LOWER BOUND, because it counts
    only direct participation and not impulse transmitted through the grid to a
    particle this tool calls clean.

    Both aggregates are read at the FINAL frame, which is the window
    `measure_reactive_ab.py` quotes. A figure measured earlier is contaminated
    only by arrivals earlier than it, so read the per-wall `first_us` against
    that figure's own window rather than reading this block for it.
    """
    cv, cd = c.col["vel_mag"], c.col["damage"]
    fF = np.asarray(c.frames[-1])
    live_proj = proj & (fF[:, cd] < 0.5)

    def agg(mask):
        if not mask.any():
            return None
        return float(fF[mask, cv].mean())

    out = {
        "n_proj": int(proj.sum()),
        "proj_touched": int((proj & touched).sum()),
        "n_armor": int(armor.sum()),
        "armor_touched": int((armor & touched).sum()),
    }
    # Residual velocity of the coherent penetrator (measure_reactive_ab's metric).
    v_all, v_clean = agg(live_proj), agg(live_proj & ~touched)
    out["rod_resid_v"] = v_all
    out["rod_resid_v_clean"] = v_clean
    out["rod_resid_v_delta_pct"] = (
        (v_clean - v_all) / v_all * 100.0 if v_all and v_clean else None)
    # Armor spall fraction (damage is latched 0/1, so the mean is the fraction).
    if armor.any():
        s_all = float(fF[armor, cd].mean())
        clean = armor & ~touched
        s_clean = float(fF[clean, cd].mean()) if clean.any() else None
        out["armor_spall"] = s_all
        out["armor_spall_clean"] = s_clean
        out["armor_spall_delta_pct"] = (
            (s_clean - s_all) / s_all * 100.0 if s_all and s_clean is not None
            else None)
    return out


def _symmetry(c: Cache, mats) -> dict:
    """The cheap check from §1.1.1, and the one that caught the dead high walls.

    A normal-incidence deck is symmetric about mid-height BY CONSTRUCTION —
    `_seed` spans the full height and the impact axis is mid-height — so the
    MATERIAL must be too. Dead walls read `rha` pos_y 0.88..119.61 against a
    120 mm domain: jammed onto the top clamp, standing off the bottom mirror.
    Live walls read 0.88 vs 0.88, exact.

    Skipped on oblique decks: the rod is tilted and the impact is deliberately
    off-centre there (§3.2), so the material is not symmetric to begin with and
    an asymmetry would mean nothing.
    """
    angle = float((c.m.get("projectile") or {}).get("angle_deg") or 0.0)
    if abs(angle) > 1e-9:
        return {"applicable": False,
                "why": f"angle_deg={angle} — not symmetric by construction"}
    fF = np.asarray(c.frames[-1])
    y = fF[:, c.col["pos_y"]]
    out = {"applicable": True, "by_material": {}}
    worst = 0.0
    for mid in np.unique(mats):
        sel = mats == mid
        lo = float(y[sel].min() - c.dom["ymin"])
        hi = float(c.dom["ymax"] - y[sel].max())
        out["by_material"][c.names.get(int(mid), f"id={mid}")] = {
            "gap_lo_mm": lo, "gap_hi_mm": hi, "asym_mm": abs(lo - hi)}
        worst = max(worst, abs(lo - hi))
    out["worst_asym_mm"] = worst
    return out


def report(r: dict) -> None:
    d = r["domain"]
    print(f"=== {r['cache']} ===")
    print(f"  domain {d['xmin']}..{d['xmax']} x {d['ymin']}..{d['ymax']} mm, "
          f"{r['n_particles']} particles, {r['n_frames']} frames "
          f"({r['total_us']:.1f} us)"
          + (f", frame stride {r['stride']}" if r["stride"] != 1 else ""))
    print(f"  eligible = started > {r['keepout_mm']:.1f} mm from that wall; "
          f"arrived = came within {r['arrive_mm']:.1f} mm")
    print(f"  {'wall':<6}{'eligible':>10}{'arrived':>9}{'first us':>10}"
          f"{'closest mm':>12}{'at us':>9}{'max v':>9}   sweep "
          f"{'/'.join(f'{D}' for D in ARRIVE_SWEEP)} mm")
    for w in WALLS:
        x = r["walls"][w]
        first = f"{x['first_us']:.1f}" if x["first_us"] is not None else "-"
        sweep = "/".join(str(x["sweep"][D]) for D in ARRIVE_SWEEP)
        print(f"  {w:<6}{x['n_eligible']:>10}{x['n_arrived']:>9}{first:>10}"
              f"{x['closest_mm']:>12.2f}{x['closest_us']:>9.1f}"
              f"{x['max_v_at_arrival']:>9.0f}   {sweep}")
        for name, m in sorted(x["by_material"].items(),
                              key=lambda kv: -kv[1]["n"]):
            print(f"           {name}: {m['n']} from {m['first_us']:.1f} us, "
                  f"{m['frac_damaged_at_end']:.0%} spalled by the end")

    ct = r["contamination"]
    print(f"  contamination (final frame; DIRECT participation only, a LOWER "
          f"BOUND — impulse through the grid is not counted)")
    print(f"    projectile {ct['proj_touched']}/{ct['n_proj']} touched, "
          f"armor {ct['armor_touched']}/{ct['n_armor']} touched")
    if ct.get("rod_resid_v") is not None:
        dv = ct["rod_resid_v_delta_pct"]
        print(f"    rod residual v : {ct['rod_resid_v']:.1f} -> "
              f"{ct['rod_resid_v_clean']:.1f} m/s excluding them"
              + (f"  ({dv:+.2f}%)" if dv is not None else ""))
    if ct.get("armor_spall") is not None:
        ds = ct["armor_spall_delta_pct"]
        print(f"    armor spall    : {ct['armor_spall']:.4f} -> "
              f"{ct['armor_spall_clean']:.4f} excluding them"
              + (f"  ({ds:+.2f}%)" if ds is not None else ""))

    sym = r["symmetry"]
    if not sym["applicable"]:
        print(f"  symmetry check : skipped — {sym['why']}")
    else:
        print(f"  symmetry check : worst mid-height asymmetry "
              f"{sym['worst_asym_mm']:.3f} mm")
        for name, s in sym["by_material"].items():
            print(f"           {name}: gap lo {s['gap_lo_mm']:.2f} vs hi "
                  f"{s['gap_hi_mm']:.2f} mm  (asym {s['asym_mm']:.3f})")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache_dir", type=Path, nargs="?")
    ap.add_argument("--all", action="store_true",
                    help="sweep every cache dir under caches/")
    ap.add_argument("--keepout", type=float, default=START_KEEPOUT,
                    help="mm a particle must start from a wall to be eligible")
    ap.add_argument("--stride", type=int, default=1,
                    help="read every Nth frame. >1 can MISS an arrival that "
                         "the material passes through between dumps — it "
                         "biases toward a clean verdict, never a dirty one.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.all:
        dirs = sorted(p.parent for p in Path("caches").glob("*/manifest.json"))
    elif args.cache_dir:
        dirs = [args.cache_dir]
    else:
        ap.error("give a cache dir or --all")

    results = []
    for d in dirs:
        if not (d / "manifest.json").is_file():
            raise SystemExit(f"{d} is not a cache dir (no manifest.json)")
        r = measure(d, keepout=args.keepout, stride=args.stride)
        results.append(r)
        if not args.json:
            report(r)
            sys.stdout.flush()
    if args.json:
        print(json.dumps(results if args.all else results[0], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
