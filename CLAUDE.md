# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo. Read this first.

---

## 1. What this project is

A **2D offline simulation of terminal ballistics** — a shell (KE penetrator / shaped-charge jet) hitting tank armor (RHA, composite, spaced/NERA, reactive) — rendered as a deformation sim with spall and fragmentation.

**The fidelity bar is *plausible*, not *validated*.** We want results that look and behave like real penetration events and respond sensibly to real-ish parameters. We are **not** building an engineering/predictive tool, and we do not need to match experimental data. When a choice trades physical rigor for stability or visual quality, take the stable/pretty option and note it.

**Not real-time.** The solver runs offline ("bakes") a scenario to a cache on disk. A separate viewer plays the cache back. A scenario taking 30s vs 90s to bake is irrelevant; iteration speed and clarity matter more than raw solver throughput.

---

## 2. The one architectural rule

> **The solver and the visualizer are two independent programs that share exactly one thing: the on-disk cache format. Nothing else.**

Everything else in this document serves that rule. The point is that **the solver engine is disposable** — we may rewrite it (Taichi → Warp → raw CUDA/C++ → Rust) when we want performance or capability we can't get otherwise. That rewrite must touch **zero** visualizer code.

Concretely:

- The solver **never** imports, references, or knows about Godot.
- The visualizer **never** imports, references, or knows about Python/Taichi/the solver's internals. It knows only how to read the cache.
- Neither side depends on the other's source tree. They could live in separate repos; they live in one here only for convenience.
- The **contract is a written spec** (`docs/CACHE_FORMAT.md`), not just code. If the format changes, the spec changes first, then both sides.
- The cache format is **language-neutral**: JSON manifest + raw little-endian `float32` binary. A future C++/Rust solver must be able to emit it with `fwrite`; Godot must be able to read it with `FileAccess`. No format that assumes a specific language's serialization (no pickle, no HDF5, no Alembic).

If you're ever unsure whether something belongs in the solver or the visualizer: **does it require knowing the physics? → solver. Does it require knowing how to draw? → visualizer.** The cache is the membrane between them.

---

## 3. Repo layout

```
/
  CLAUDE.md                      # this file
  README.md
  docs/
    CACHE_FORMAT.md              # THE CONTRACT — source of truth, edit this first
    PHYSICS.md                   # MLS-MPM notes, material models, unit system, public refs
  solver/                        # standalone; knows nothing about Godot
    pyproject.toml
    src/ballistics_solver/
      config.py                  # scenario schema (dataclasses/pydantic), loads YAML
      mpm.py                     # MLS-MPM transfer kernels (Taichi)
      materials.py               # elasticity + von Mises plasticity + damage
      cache_writer.py            # writes manifest.json + frames.bin per the spec
      run.py                     # CLI: scenario.yaml -> cache dir
    scenarios/                   # input decks (YAML)
      apfsds_vs_rha.yaml
      heat_vs_composite.yaml
    tests/
  visualizer/                    # Godot 4 project; knows ONLY the cache format
    project.godot
    scripts/
      cache_loader.gd            # parses manifest + streams frames.bin
      particle_view.gd           # MultiMeshInstance2D + colormap shader
    fixtures/
      tiny_golden_cache/         # a canned 3-frame cache so the viewer can be
                                 # built & tested with NO solver present
  tools/                         # language-neutral helpers, depend on neither side
    validate_cache.py            # checks a cache dir against CACHE_FORMAT.md
    inspect_cache.py             # matplotlib scatter preview — sanity-check bakes
                                 # without opening Godot
  caches/                        # gitignored — bake outputs live here (large)
```

This layout **physically enforces** the decoupling. Keep it that way. Do not add a shared library that both `solver/` and `visualizer/` import.

---

## 4. The cache format (summary — full spec in `docs/CACHE_FORMAT.md`)

A cache is a **directory**:

```
caches/apfsds_vs_rha/
  manifest.json
  frames.bin
```

**`manifest.json`** (human-readable, small):

```json
{
  "schema_version": 1,
  "scenario": "apfsds_vs_rha",
  "particle_count": 240000,
  "frame_count": 90,
  "attributes": ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id"],
  "dtype": "float32",
  "frame_dt": 2.0e-6,
  "domain": {"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 100},
  "units": "mm-ms-g (see docs/PHYSICS.md)",
  "materials": {"0": "tungsten_rod", "1": "rha", "2": "ceramic", "3": "era_filler"}
}
```

**`frames.bin`**: `particle_count × frame_count` records, each record = one particle's attributes in the order given by `attributes`, packed as little-endian `float32`, row-major. Frame *f* starts at byte offset `f * particle_count * len(attributes) * 4`.

Design commitments that keep this loosely coupled:

- **Particle count is fixed** for a bake (MPM particles persist; "spall" = particles flagged via `damage` and detached, not created/destroyed). Simplifies everything downstream.
- **The visualizer reads the attribute layout from the manifest** — it must NOT hardcode column offsets. The solver can add a new attribute (e.g. `temperature`) by appending to `attributes` and the viewer keeps working, coloring by whatever it's told to.
- **Substeps ≠ frames.** The solver runs thousands of tiny physics substeps but dumps only every Nth as a render frame. `frame_dt` documents the spacing. **Target a uniform `frame_dt` (currently `2.0e-7` s), not a frame count**: smoothness is frames-per-simulated-microsecond, so a fixed frame budget makes long decks jerkier than short ones — the oblique decks used to be the least smooth precisely because they were the longest. Set `total_time` per deck to cover the whole event, then derive `frame_count = total_time / frame_dt`. Decks currently run 200–700 frames; frame count drives cache size and viewer cost, **not** solver cost (substeps are derived from `frame_dt` and the CFL bound).
- Alternative per-frame files (`frame_00042.bin`) are allowed by the spec for crash-resilient/streamable bakes; v1 uses a single blob for simplicity. Either way the manifest is authoritative.

**Rule: any change to what's in a cache is a change to `docs/CACHE_FORMAT.md` and a bump of `schema_version`.** `tools/validate_cache.py` must pass on every cache the solver emits.

---

## 5. Tech stack & why (all swappable except the format)

| Layer | Choice | Rationale | Swap trigger |
|---|---|---|---|
| Solver | **Python + Taichi** (MLS-MPM) | Canonical 88-line MLS-MPM to grow from; fastest path to a working bake | If Taichi blocks us on Blackwell or we need more perf → **NVIDIA Warp** (NVIDIA-maintained, day-one Blackwell), then raw CUDA/C++ |
| Viewer | **Godot 4**, `MultiMeshInstance2D` | One draw call for the whole point cloud; per-instance color; trivial `float32` reading | Unlikely to change; it's just a player |
| Contract | JSON + raw `float32` | Language-neutral; survives any solver rewrite | Never, casually |

**Hardware:** target machine has an RTX 5090 (Blackwell, compute capability **sm_120**). This is new enough that GPU toolchains lagged it across the ecosystem through late 2025.

**⚠️ First task on this machine: verify the solver actually runs on the GPU.** Taichi JITs via LLVM→PTX and PTX is forward-compatible, so it *should* run on sm_120 via driver PTX-JIT with a recent driver + CUDA 12.8+ — but confirm empirically (run a trivial `ti.init(arch=ti.cuda)` kernel and check it's not silently on CPU) before building on it. Taichi's active maintenance has slowed (latest ~1.7.x), so if Blackwell support is broken, **do not sink time fighting it — switch the solver to Warp.** The architecture is designed to make that a contained change; that's the whole point.

---

## 6. Physics backbone (details in `docs/PHYSICS.md`)

- **Method: MLS-MPM** (Moving Least Squares Material Point Method). Chosen over SPH because the background grid handles stress divergence and self-contact automatically (rod and armor auto-collide by sharing the grid), and it avoids SPH's tensile instability — which is exactly what would wreck the spall/fracture we care about. SPH may return later specifically for HEAT-jet fluid-like erosion.
- **Material model:** elasticity (fixed-corotated or Neo-Hookean) + **von Mises plastic return-mapping** for metals + a **damage threshold** that detaches particles into free fragments (the spall spray). Ceramic/composite = higher stiffness, lower damage threshold, brittle-ish; ERA = an impulse layer that degrades the penetrator.
- **The cost driver is the CFL timestep, not particle count.** Steel's sound speed ~5 km/s + explicit MPM ⇒ `dt` on the order of 1e-8–1e-7 s (in SI). Penetration is a ~microseconds event, so it's a short physical window = thousands of cheap substeps. This is *why* we're offline and on the GPU.

---

## 7. Units — read before touching numbers

**Work in a single consistent, non-dimensionalized unit system. Never mix raw SI into the kernels.** Raw SI makes stiffness numbers huge and `dt` tiny, inviting float error and confusion.

Recommended system (well-established for impact/ballistics; document in `PHYSICS.md`): **mm – ms – g**, which gives velocity in m/s, stress in MPa, force in N. In this system steel ρ≈7.85e-3, E≈2e5 MPa, a 1.5 km/s impact ≈ 1500. Keep **all** physical constants in `materials.py` / a constants module, defined once, in these units. The `units` field in the manifest records the choice.

---

## 8. Commands (intended interface — build toward these)

```bash
# Solver setup
cd solver && pip install -e .

# Bake a scenario -> cache dir
python -m ballistics_solver.run scenarios/apfsds_vs_rha.yaml --out ../caches/apfsds_vs_rha

# Validate a cache against the format spec (must pass before the viewer sees it)
python tools/validate_cache.py caches/apfsds_vs_rha

# Quick visual sanity check WITHOUT Godot (matplotlib scatter of a few frames)
python tools/inspect_cache.py caches/apfsds_vs_rha

# Visualize: open visualizer/ in Godot 4, point it at the cache dir, play.
# The viewer can also be developed against visualizer/fixtures/tiny_golden_cache/
# with no solver present.
```

---

## 9. Working conventions

- **Scenarios are data, not code.** New setups (projectile type/velocity/angle, armor stack) go in a YAML deck under `solver/scenarios/`, parsed by `config.py`. Don't hardcode scenario specifics in kernels.
- **Prefer growing the reference MLS-MPM** over rewriting from scratch — add plasticity, then damage, then the multi-material armor stack, validating visually at each step via `inspect_cache.py`.
- **Keep a golden cache.** `visualizer/fixtures/tiny_golden_cache/` lets the viewer be built and tested independently of the solver. Regenerate it only via a documented command, and update it when `schema_version` bumps.
- When editing the cache format: **spec first** (`docs/CACHE_FORMAT.md` + `schema_version`), then `cache_writer.py`, then `cache_loader.gd`, then `validate_cache.py`. In that order.

---

## 10. Scope guard (non-goals — keep the project on the right side of the line)

This is an **educational / game-grade** simulator built entirely on **public-domain, textbook-level physics** (e.g. Tate–Alekseevskii, hydrodynamic penetration, standard MPM; public references cited in `PHYSICS.md`). Keep it there:

- **Plausibility, not prediction.** We do not validate against, or aim to reproduce, real experimental performance.
- **No engineering-grade munition or armor-defeat work.** Do not seek out, encode, or "tune toward" real liner geometries, explosive formulations, specific real armor-package specifications, or anything meant to optimize lethality or defeat a specific real system.
- **Material parameters are representative and illustrative**, sourced from public literature and documented as such. Order-of-magnitude, not spec-sheet.
- No classified or export-controlled data enters this repo. If a request would only be answerable with non-public military data, it's out of scope by construction — public physics is the ceiling here anyway.

---

## 11. Gotchas cheat-sheet

- **Silent CPU fallback:** always assert the solver is actually on `ti.cuda` (or Warp CUDA). A "slow but working" bake may mean it quietly ran on CPU.
- **CFL / stability:** if the sim blows up, `dt` is almost always the first suspect — check it against sound speed × grid spacing before touching the material model.
- **Unit scaling:** a sim that explodes or does nothing is often a units mistake, not a physics bug. Everything in one system (§7).
- **Attribute layout:** viewer reads it from the manifest; never hardcode offsets (§4).
- **Frames vs substeps:** dump every Nth substep. Dumping every substep = enormous caches and a slideshow with thousands of near-identical frames.
- **Don't couple the halves.** No shared import between `solver/` and `visualizer/`. The cache is the only bridge.
