"""Measure the reactive-armor A/B (ERA vs its inert twin) from baked caches.

Solver-free by construction (tools/ depends on neither half — root §3): it reads
the cache contract and nothing else.

WHY THIS EXISTS AS A COMMITTED TOOL. Milestones 5 and 6 measured this A/B with an
**ad-hoc probe that was never committed**, so their headline figures — 16 % lower
main-plate spall at 55°, −8.5 % residual velocity — cannot be reproduced, only
quoted. That is exactly the defect PHYSICS §3.3 already records for the M5 plate-
separation numbers ("M5's probe was ad-hoc and never committed. Don't quote them").
A number you cannot re-derive is a number you cannot re-measure when the physics
moves — and it moved twice: Mie-Grüneisen (§3.10) and the boundary-condition fix
(§1.1.1), which hits these decks hardest because the detonation drives filler
straight at the ceiling.

WHAT IT COMPARES. `apfsds_vs_era` vs `apfsds_vs_era_inert` (and the `_oblique`
pair). The decks are identical in geometry, areal mass, and timing; the filler's
`reactive` flag is the only variable. So the delta is the reactive contribution,
isolated from "there is simply more material in the path".

METRIC DEFINITIONS ARE THE WHOLE ARGUMENT — §3.3's rule: *state the metric or the
SIGN flips*. Every population here is labelled at **frame 0** and then followed by
index, because the contract fixes particle count and persists particles (§5), which
makes an index a durable material label (the milestone-7 trick). Labelling by
position at the FINAL frame instead would silently redefine "the main plate" to
mean "whatever ended up there", which includes the flyer debris driven into it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PAIRS = [
    ("0°", "apfsds_vs_era", "apfsds_vs_era_inert"),
    ("55°", "apfsds_vs_era_oblique", "apfsds_vs_era_oblique_inert"),
]


class Cache:
    def __init__(self, d: Path):
        self.d = d
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        self.attrs = list(m["attributes"])
        self.n_p = int(m["particle_count"])
        self.n_f = int(m["frame_count"])
        self.dt = float(m["frame_dt"])
        self.names = {int(k): v for k, v in m.get("materials", {}).items()}
        self.ids = {v: k for k, v in self.names.items()}
        self.stride = len(self.attrs)

    def col(self, name: str) -> int:
        return self.attrs.index(name)

    def frame(self, f: int) -> np.ndarray:
        if f < 0:
            f += self.n_f
        return np.asarray(np.memmap(self.d / "frames.bin", dtype="<f4", mode="r",
                                    offset=f * self.n_p * self.stride * 4,
                                    shape=(self.n_p, self.stride)))


def measure(cache_dir: Path) -> dict:
    c = Cache(cache_dir)
    ix, iv, idm, imt = (c.col(k) for k in ("pos_x", "vel_mag", "damage",
                                           "material_id"))
    A0, AF = c.frame(0), c.frame(-1)
    mats = np.rint(A0[:, imt]).astype(int)

    rod = mats == c.ids["tungsten_rod"]
    rha = mats == c.ids["rha"]
    # The MAIN plate is the one behind the standoff gap — the protected plate, and
    # the only one the A/B is about. Labelled by where it SEEDS: the stack's plates
    # are driven downrange by the detonation, so a final-frame x-band would mix them.
    # The gap is the widest empty span in the seeded rha, so find it rather than
    # hardcoding 199 — a deck edit must not silently re-point this at the flyer.
    xs = np.sort(A0[rha, ix])
    gaps = np.diff(xs)
    g = int(np.argmax(gaps))
    split = 0.5 * (xs[g] + xs[g + 1])
    main = rha & (A0[:, ix] > split)

    live_rod = rod & (AF[:, idm] < 0.5)
    out = {
        "gap_at_x": float(split),
        "gap_mm": float(gaps[g]),
        "n_main": int(main.sum()),
        "n_rod": int(rod.sum()),
        # Mean damage over the main plate's OWN particles = the fraction spalled
        # (damage is a latched 0/1 here), which is M6's "main-plate spall".
        "main_spall": float(AF[main, idm].mean()),
        "rod_damage": float(AF[rod, idm].mean()),
        # Residual velocity: mean speed of the rod material that is still COHERENT
        # at the final frame. Spalled fragments are excluded because they are debris
        # carrying momentum, not the penetrator.
        "rod_resid_v": float(AF[live_rod, iv].mean()) if live_rod.any() else 0.0,
        "n_live_rod": int(live_rod.sum()),
        # Free-flight POSITION, reported but not to be leaned on — see the note in
        # `report`. Both rods perforate, so this is where the residual got to, not
        # how hard it was resisted.
        "rod_tip_x": float(AF[rod, ix].max()),
    }
    return out


def report(pairs) -> int:
    print("Reactive-armor A/B — reactive filler vs its INERT twin, one field apart.")
    print("Measured from baked caches; populations labelled at FRAME 0 and followed "
          "by index.\n")
    for label, a_name, b_name in pairs:
        a_dir, b_dir = Path("caches") / a_name, Path("caches") / b_name
        if not (a_dir / "manifest.json").exists() or not (b_dir / "manifest.json").exists():
            print(f"=== {label}: SKIPPED (missing cache) ===\n")
            continue
        a, b = measure(a_dir), measure(b_dir)
        print(f"=== {label}:  {a_name}  vs  {b_name} ===")
        if a["n_main"] != b["n_main"] or a["n_rod"] != b["n_rod"]:
            print(f"  *** ARMS DO NOT MATCH: main {a['n_main']} vs {b['n_main']}, "
                  f"rod {a['n_rod']} vs {b['n_rod']}. The decks must be identical "
                  f"in geometry — this comparison is VOID. ***")
        print(f"  main plate identified behind a {a['gap_mm']:.1f} mm gap at "
              f"x={a['gap_at_x']:.1f} ({a['n_main']} particles)")
        print(f"  {'metric':<26}{'reactive':>12}{'inert':>12}{'delta':>12}")
        for key, name, fmt in (("main_spall", "main-plate spall", "{:.4f}"),
                               ("rod_resid_v", "rod residual v (m/s)", "{:.1f}"),
                               ("rod_damage", "rod damage", "{:.4f}"),
                               ("rod_tip_x", "rod tip x (mm)", "{:.2f}")):
            va, vb = a[key], b[key]
            d = (va - vb) / vb * 100.0 if vb else float("nan")
            print(f"  {name:<26}{fmt.format(va):>12}{fmt.format(vb):>12}"
                  f"{d:>11.1f}%")
        print()
    print("READING THIS HONESTLY — every one of these is a mistake already made here:")
    print("  * rod tip x is a free-flight POSITION, not resistance. Both rods")
    print("    perforate, so a SLOWER rod at equal position is diverging (§3.2).")
    print("    Read residual v instead.")
    print("  * run-to-run scatter is <=0.11% on AGGREGATES; extrema wobble ~1%.")
    print("    Model sensitivity is far larger: this same A/B has read 40/21/16%")
    print("    across geometry and nose changes. Sign robust, magnitude NOT portable.")
    print("  * 'not cut' != 'not affected'. The rod is not cut at either angle;")
    print("    that failed a-priori expectation is the honest null, and the")
    print("    backing-plate effect is the real finding.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pair", choices=["0", "55", "both"], default="both")
    args = p.parse_args()
    sel = [x for x in PAIRS if args.pair == "both" or x[0].startswith(args.pair)]
    return report(sel)


if __name__ == "__main__":
    sys.exit(main())
