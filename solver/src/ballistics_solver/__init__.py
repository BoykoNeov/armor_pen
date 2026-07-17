"""ballistics_solver — offline MLS-MPM terminal-ballistics solver.

This package bakes a scenario (YAML deck) into an on-disk cache described by
``docs/CACHE_FORMAT.md``. It is a *standalone* program: it must never import,
reference, or know about the Godot visualizer. The only thing it shares with
the visualizer is the cache format. See CLAUDE.md §2.
"""

__version__ = "0.0.1"

# The cache format this solver emits. Must match docs/CACHE_FORMAT.md and the
# validator/loader. Bumping this is a format change — follow CLAUDE.md §9.
# v2 (milestone 13): appended the `internal_energy` column.
# v3: added the scenario block (`projectile` / `armor` / `material_descriptions`)
#     so a reader can say what it is drawing. Manifest-only — no column moved, so
#     the 30 shipped caches were migrated with `run.py --remanifest` rather than
#     rebaked (CACHE_FORMAT §2.2).
CACHE_SCHEMA_VERSION = 3
