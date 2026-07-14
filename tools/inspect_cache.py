#!/usr/bin/env python3
"""Quick visual sanity check of a cache — matplotlib scatter, no Godot needed.

Language-neutral helper (CLAUDE.md §3): reads the cache per docs/CACHE_FORMAT.md
and scatters a few frames so you can eyeball a bake without opening the viewer.

    python tools/inspect_cache.py caches/apfsds_vs_rha
    python tools/inspect_cache.py caches/apfsds_vs_rha --color stress --frames 0,45,89

Reads the attribute layout from the manifest — never hardcodes column offsets.
Requires numpy + matplotlib (`pip install -e "solver[inspect]"` or install them
directly). Uses no solver/visualizer imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_frame(cache_dir: Path, manifest: dict, frame: int):
    import numpy as np

    pc = manifest["particle_count"]
    stride = len(manifest["attributes"])
    offset = frame * pc * stride * 4  # bytes, per CACHE_FORMAT §3
    with (cache_dir / "frames.bin").open("rb") as fh:
        fh.seek(offset)
        buf = fh.read(pc * stride * 4)
    return np.frombuffer(buf, dtype="<f4").reshape(pc, stride)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scatter-plot frames of a cache.")
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--color", default="vel_mag", help="attribute to color by")
    parser.add_argument("--frames", default=None,
                        help="comma-separated frame indices (default: first/middle/last)")
    args = parser.parse_args(argv)

    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError:
        print("inspect_cache needs numpy + matplotlib "
              "(pip install numpy matplotlib)", file=sys.stderr)
        return 2

    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    attrs = manifest["attributes"]
    if args.color not in attrs:
        print(f"--color {args.color!r} not in attributes {attrs}", file=sys.stderr)
        return 2
    ix, iy, ic = attrs.index("pos_x"), attrs.index("pos_y"), attrs.index(args.color)

    if args.frames:
        frames = [int(f) for f in args.frames.split(",")]
    else:
        last = manifest["frame_count"] - 1
        frames = sorted({0, last // 2, last})

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 5), squeeze=False)
    dom = manifest["domain"]
    for ax, f in zip(axes[0], frames):
        data = _load_frame(args.cache_dir, manifest, f)
        sc = ax.scatter(data[:, ix], data[:, iy], c=data[:, ic], s=6, cmap="inferno")
        ax.set_title(f"frame {f}")
        ax.set_xlim(dom["xmin"], dom["xmax"])
        ax.set_ylim(dom["ymin"], dom["ymax"])
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label=args.color, shrink=0.7)

    fig.suptitle(f"{manifest['scenario']} — colored by {args.color}")
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
