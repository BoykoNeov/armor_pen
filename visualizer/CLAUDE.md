# visualizer/ — CLAUDE.md

Godot 4 playback viewer. **Reads the root `CLAUDE.md` first** — this file only
adds viewer-local notes.

## Hard rules (repeated because they're load-bearing)

- **Knows ONLY the cache format.** Never import or reference the solver,
  Python, or Taichi (root §2). The only input is a cache directory.
- **Read the attribute layout from the manifest — never hardcode column
  offsets** (root §4, CACHE_FORMAT §2). The solver can append attributes and
  the viewer must keep working.
- **Develop against the fixture.** `fixtures/tiny_golden_cache/` lets you build
  and run the viewer with no solver present. `particle_view.gd` defaults to it.

## Layout

| File | Role |
|---|---|
| `project.godot` | Godot 4 project; main scene = `main.tscn` |
| `main.tscn` | Minimal scene: one `MultiMeshInstance2D` + `particle_view.gd` |
| `scripts/cache_loader.gd` | Parses manifest + streams `frames.bin` (the contract reader) |
| `scripts/particle_view.gd` | `MultiMeshInstance2D` playback + per-instance color |
| `fixtures/tiny_golden_cache/` | Canonical 3-frame cache (regenerate via `tools/make_golden_cache.py`) |

## Why MultiMeshInstance2D

One draw call for the whole point cloud, per-instance color, trivial `float32`
reads — the viewer is "just a player." Unlikely to change (root §5).
