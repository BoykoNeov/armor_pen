"""MLS-MPM transfer kernels (Taichi) — the physics core.

STATUS: STUB. The kernels are not implemented yet. This module will hold the
P2G / grid-update / G2P cycle and the substep loop, growing from the canonical
88-line MLS-MPM reference (see docs/PHYSICS.md §1).

Two hard rules for whoever fills this in:

  1. This module must NEVER import or reference the visualizer (CLAUDE.md §2).
  2. Before trusting the GPU, assert it is actually on ti.cuda and not a silent
     CPU fallback (CLAUDE.md §11). ``assert_gpu()`` below is the guard to call.

The RTX 5090 is Blackwell / sm_120; Taichi *should* PTX-JIT onto it with a
recent driver + CUDA 12.8+, but this is unconfirmed. If Taichi fights back on
Blackwell, do not rabbit-hole — switch the solver to NVIDIA Warp (CLAUDE.md §5).
"""

from __future__ import annotations


def assert_gpu() -> None:
    """Fail loudly if the solver is not actually running on the CUDA backend.

    A 'slow but working' bake often means Taichi silently fell back to CPU
    (CLAUDE.md §11). Call this right after ``ti.init`` before any real bake.
    """
    import taichi as ti  # local import so config/schema stay Taichi-free

    backend = ti.lang.impl.current_cfg().arch
    if backend != ti.cuda:
        raise RuntimeError(
            f"solver is on backend {backend!r}, not ti.cuda — refusing to bake. "
            "Silent CPU fallback usually means a driver/CUDA/Blackwell issue; "
            "see CLAUDE.md §5 and §11."
        )


def bake(scenario, writer) -> None:  # noqa: ANN001 — types land with the impl
    """Run the substep loop and dump every Nth substep as a render frame.

    Not implemented. Wiring lives in run.py; the physics lands here.
    """
    raise NotImplementedError(
        "MLS-MPM bake not implemented yet — this is a scaffold. "
        "See docs/PHYSICS.md for the intended transfer cycle."
    )
