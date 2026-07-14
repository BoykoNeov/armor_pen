# tiny_golden_cache

A canonical, committed **3-frame** cache so the visualizer can be built and
tested with **no solver present** (CLAUDE.md §9). `tools/validate_cache.py`
passes on it — it is the concrete pin for the `docs/CACHE_FORMAT.md` contract.

**Do not hand-edit these files.** Regenerate only via the documented command:

```bash
python tools/make_golden_cache.py
```

Regenerate whenever `schema_version` bumps. The scene (a small tungsten rod
approaching an RHA plate) is an **illustrative toy, not a physics result** —
just well-formed data: 20 particles × 3 frames × 6 attributes × 4 bytes =
1440 bytes in `frames.bin`.
