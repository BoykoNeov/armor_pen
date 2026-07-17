# Cache Format — THE CONTRACT (v2)

This document is the **single source of truth** for the on-disk cache format.
It is the one and only thing the solver and the visualizer share. If this
document and any code disagree, **this document wins** and the code is a bug.

> **Version history.**
> - **v2** (milestone 13) — appended the `internal_energy` column. Mie-Grüneisen
>   evolves a specific internal energy per particle (`docs/PHYSICS.md` §3.10);
>   v2 exposes it. No layout, ordering, or endianness change: v2 is v1 with one
>   more name in `attributes`. A v1 cache is **not** a valid v2 cache (the column
>   is absent), which is why the version moved — see the note in §2.
> - **v1** — initial format.

> **Change protocol (CLAUDE.md §9).** Any change to what a cache contains is a
> change to *this file first*, plus a bump of `schema_version`, then — in this
> order — `solver/.../cache_writer.py`, `visualizer/scripts/cache_loader.gd`,
> `tools/validate_cache.py`, and finally the golden fixture. Never change the
> format in code without changing this file.

The format is **language-neutral by design**: a JSON manifest plus a raw
little-endian `float32` blob. A future C++/Rust solver must be able to emit it
with `fwrite`; Godot must be able to read it with `FileAccess`. No format that
assumes a specific language's serialization (no pickle, no HDF5, no Alembic).

---

## 1. A cache is a directory

```
caches/<scenario_name>/
  manifest.json      # human-readable, small; authoritative for layout
  frames.bin         # raw float32 particle data (v1: single blob)
```

The directory name is a convenience label only; `manifest.json` is
authoritative for everything.

---

## 2. `manifest.json`

A single JSON object. Example (`schema_version: 2`):

```json
{
  "schema_version": 2,
  "scenario": "apfsds_vs_rha",
  "particle_count": 240000,
  "frame_count": 90,
  "attributes": ["pos_x", "pos_y", "vel_mag", "stress", "damage", "material_id", "internal_energy"],
  "dtype": "float32",
  "frame_dt": 2.0e-6,
  "domain": {"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 100},
  "units": "mm-ms-g (see docs/PHYSICS.md)",
  "materials": {"0": "tungsten_rod", "1": "rha", "2": "ceramic", "3": "era_filler"}
}
```

### Field reference

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema_version` | int | yes | Format version. v2 = this document. Readers **must** reject a version they do not understand. |
| `scenario` | string | yes | Human label for the bake. Matches the input deck's name by convention, not by requirement. |
| `particle_count` | int > 0 | yes | Number of particles, **fixed for the whole bake** (see §5). |
| `frame_count` | int > 0 | yes | Number of render frames stored. |
| `attributes` | array of string | yes | Ordered list of per-particle `float32` values in each record. Order defines the binary layout. Must be non-empty and contain no duplicates. |
| `dtype` | string | yes | `"float32"` in v1. Reserved for future widening; readers must reject anything else. |
| `frame_dt` | number > 0 | yes | Simulated time between consecutive render frames, in the manifest's unit system (**not** the solver substep dt). |
| `domain` | object | yes | World-space bounds: `xmin`, `xmax`, `ymin`, `ymax` (numbers, `max > min`). Used by the viewer to frame the camera. |
| `units` | string | yes | Free text naming the unit system, e.g. `"mm-ms-g"`. See `docs/PHYSICS.md`. |
| `materials` | object | yes | Map from stringified integer material id → human name. Every `material_id` value appearing in the data should have a key here. |

### Required & recommended attributes

`attributes` is **open** — the solver may append new columns and older viewers
keep working, coloring by whatever they are told. Two constraints:

- `pos_x` and `pos_y` **must** be present (the viewer needs positions to draw).
- The viewer **must** read the attribute layout from the manifest and locate
  columns by name. It **must not** hardcode column offsets.

**Openness is a property of *readers*, not a licence to skip the version bump.**
A name-driven reader survives an appended column — that is what "open" buys, and
it is why v2 needs no `cache_loader.gd` rewrite. It does **not** make the two
versions interchangeable: a v1 cache lacks `internal_energy` entirely, so a
consumer that requires it would fault on data the manifest never promised. The
bump is what lets a reader tell the two apart *before* reading. Both rules hold
at once (root CLAUDE.md §4: any change to what is in a cache bumps the version).

Conventional attribute semantics (all `float32`):

| Attribute | Meaning / units |
|---|---|
| `pos_x`, `pos_y` | Position in world space (domain units, e.g. mm). |
| `vel_mag` | Velocity magnitude (m/s in the mm-ms-g system). |
| `stress` | A scalar stress measure, e.g. von Mises equivalent (MPa). |
| `damage` | Scalar damage in `[0, 1]`; a detached (spalled) particle is one flagged via this attribute, not a created/destroyed particle. |
| `material_id` | Integer material id **stored as a float32**; readers round to nearest int and look it up in `materials`. |
| `internal_energy` | **Specific** internal energy — per unit **mass**, `J/kg` (numerically `(m/s)²` in mm-ms-g). `0` = the reference state, so a deck at rest reads 0. **Not** energy per unit volume: the per-reference-volume density is `ρ₀·e`, which is what the solver's energy balance is written in. See the reader's note below. |

**Reading `internal_energy` honestly.** Three things a viewer must not assume:

- **It is not temperature, and must not be labelled as one.** Temperature needs a
  per-material heat capacity `c_v`, which this project does not carry; deriving one
  is a future change, not a rename. A viewer may color by this column, but the
  legend says *internal energy*.
- **It is volumetric + shock-heating work only.** Plastic dissipation is **not**
  fed to `e` (PHYSICS §3.10), so strongly-shearing regions — the crater walls, not
  the jet stagnation point — are missing a real heat source and this column
  **under-reads** there. It is a deliberate, stated limit, not a bug to fix in the
  viewer.
- **It is not comparable across materials.** Different `ρ₀` and different baselines
  mean a shared color scale reads as "copper is cold" when it is not. Normalize
  per material when coloring; that is a viewer concern, not a format one.
- **Its zero carries a small positive bias, and it is never negative.** The solver
  clamps `e ≥ 0` — that is a theorem of the model, but float32 cancellation in the
  energy solve violates it by ~1e-4 J/kg, and left alone that seeds a runaway
  (negative `e` ⇒ negative thermal pressure ⇒ spurious tension ⇒ more negative
  `e`). The clamp is one-sided, so it injects a bounded trickle of energy rather
  than removing any. Against a physical ~1e5 J/kg the bias is noise. A viewer must
  not infer from `min(e) == 0` that a region is exactly at the reference state.

---

## 3. `frames.bin`

`particle_count × frame_count` records. Each record is one particle's
attributes for one frame, packed as little-endian `float32` in the exact order
listed in `manifest.attributes`.

Layout is **frame-major, then particle-major** (row-major over
`[frame][particle][attribute]`):

```
record stride  R = len(attributes) * 4 bytes
frame  stride  F = particle_count * R  bytes

byte offset of (frame f, particle p, attribute a)
    = f * F + p * R + a * 4

total file size = frame_count * particle_count * len(attributes) * 4 bytes
```

Frame `f` therefore begins at byte offset `f * particle_count *
len(attributes) * 4` — a viewer can `seek` directly to any frame.

**Endianness is little-endian, always** (matches x86/ARM and Godot's default
`float32` reads). A big-endian emitter must byte-swap.

---

## 4. Alternative per-frame layout (allowed, not used in v1)

For crash-resilient or streamable bakes, a cache **may** instead store one file
per frame:

```
caches/<scenario_name>/
  manifest.json
  frame_00000.bin
  frame_00001.bin
  ...
```

Each `frame_NNNNN.bin` holds exactly `particle_count` records in the same
record layout as §3. When this layout is used the manifest must set
`"frame_layout": "per_frame"` (default when absent: `"single_blob"`).
**v1 tooling and the golden fixture use the single-blob layout.**

---

## 5. Design commitments (why the format is shaped this way)

These keep the two halves loosely coupled — do not violate them casually:

- **Fixed particle count.** MPM particles persist for the whole bake. "Spall"
  is particles *flagged* (via `damage`) and detached, **never** created or
  destroyed. A fixed `particle_count` makes every offset in §3 a constant and
  makes the viewer's buffers static.
- **Manifest-driven layout.** The viewer discovers columns by name from the
  manifest. Appending an attribute is a backward-compatible change for readers
  that color by name.
- **Substeps ≠ frames.** The solver runs thousands of tiny physics substeps and
  dumps only every Nth as a render frame. `frame_dt` documents the spacing; it
  is *not* the substep dt. Decks target a **uniform `frame_dt`** (currently
  `2.0e-7` s) rather than a fixed frame count, so playback smoothness is the
  same regardless of how long an event runs; `frame_count` therefore varies with
  `total_time` (200–700 today). This is a solver-side convention, not a format
  rule — readers take `frame_count` from the manifest and must not assume a
  range.
- **No language-specific serialization.** JSON + raw `float32` only.

---

## 6. Validation

`tools/validate_cache.py` checks a cache directory against this document and
**must pass on every cache the solver emits**. At minimum it verifies:

1. `manifest.json` parses and contains every required field with the right type.
2. `schema_version` is a version the tool understands.
3. `attributes` is non-empty, duplicate-free, and contains `pos_x` and `pos_y`.
4. `dtype == "float32"`.
5. `domain` has `xmax > xmin` and `ymax > ymin`.
6. `frames.bin` (single-blob) has **exactly**
   `frame_count * particle_count * len(attributes) * 4` bytes — no more, no less.
7. Every distinct `material_id` in the data has a `materials` entry (best-effort;
   samples the first frame only).
8. **Every value is finite** — no `NaN`, no `±Inf`, in any column, in any frame.

### Why finiteness is checked, and why it is not optional

Rule 8 was added in v2 after a diverged bake — 97 % of particles carrying `NaN`
velocity from frame 276 onward — validated **`OK`** against rules 1–7. Every
structural rule passed, because they all did their job: the manifest was
well-formed, the blob was exactly the right size, and rule 7 sampled frame 0,
which was still clean.

That is the shape of the hazard. A blown-up cache is **structurally perfect** —
divergence changes the values, never the layout — so a size-and-schema validator
cannot see it, and the failure reaches the viewer as particles that silently
vanish (`NaN` positions fail every comparison) rather than as an error. Rules 1–7
answer "is this a cache?". Rule 8 is the only one that asks "is it *data*?".

`frame_count`-many frames are scanned, not sampled: a divergence has a first
frame, and the tool reports it, because *when* it broke is the whole diagnostic.
Sampling would trade the one number worth having for a cost that is already
small next to the bake that produced the file.

The golden fixture at `visualizer/fixtures/tiny_golden_cache/` is a canonical,
committed cache that `validate_cache.py` passes. It lets the viewer be built and
tested with **no solver present**. Regenerate it only via the documented command
(`python tools/make_golden_cache.py`) and update it whenever `schema_version`
bumps.
