---
description: Bake a scenario to a cache, then validate it against the format spec
argument-hint: <scenario-name> (e.g. apfsds_vs_rha)
---

Bake the scenario `$1` and verify the output cache is spec-compliant.

1. Run the solver:
   `cd solver && python -m ballistics_solver.run scenarios/$1.yaml --out ../caches/$1`
   - If the solver reports a silent-CPU-fallback / GPU assert failure, STOP and
     report it — do not "fix" by passing `--cpu`. A broken GPU path is a real
     signal (see CLAUDE.md §5, §11).
   - The bake step is currently a stub (`mpm.py`); if it raises
     `NotImplementedError`, say so plainly rather than pretending it baked.
2. Validate the cache: `python tools/validate_cache.py caches/$1`
3. If valid, offer a quick visual check: `python tools/inspect_cache.py caches/$1`
4. Report the frame count, particle count, and validation result concisely.
