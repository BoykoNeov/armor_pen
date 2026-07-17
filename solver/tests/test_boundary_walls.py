"""The free-slip domain walls must be REACHABLE — on all four sides.

WHY THIS FILE EXISTS. `_grid_op`'s high walls never fired. Not on one deck — on
every deck, since milestone 1. The test was `i > nx - bound`, but `nx` is the
ALLOCATED width, which carries 3 pad nodes past the domain so a clamped particle's
3x3 stencil has somewhere to land. So the high wall band sat entirely in the pad,
outside the domain, where `_g2p`'s position clamp guarantees no material ever goes.

The low walls worked the whole time (indices count from 0, so `i < bound` is
genuinely inside), which is exactly why it survived: every bake ran a working mirror
on its low edges and, on its high edges, no wall at all. What stops material there
instead is the position clamp — an infinitely rigid plane that arrests DISPLACEMENT
and leaves velocity untouched, so material piles onto it still carrying its full
inbound speed and is crushed by everything behind it.

It surfaced only when milestone 13 gave the solver an energy equation: the pinned
filler's internal energy jumped 4 orders of magnitude (24 -> 7.1e5 J/kg) in exactly
the frames it reached the clamp plane, and `era_filler` reads `e` in its EOS stress
branch. `apfsds_vs_nera` stayed clean throughout for the reason that names the
mechanism: an inert filler is never detonated INTO the ceiling.

Two properties are pinned here, and the first is the one that was false:

  1. REACHABILITY (`test_every_wall_is_reachable...`) — pure arithmetic on the
     clamp and the stencil, no GPU. This is the test that would have caught it, and
     it is written against the clamp rather than against `_grid_op` so it cannot be
     satisfied by copying the kernel's own mistake.
  2. BEHAVIOUR (the kernel tests) — the wall actually zeroes an inbound normal
     velocity at a node the material can reach, on each of the four sides, and
     leaves tangential motion and outbound motion alone.

Never assert a wall band against an ARRAY SHAPE. Assert it against the domain.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ballistics_solver import mpm

BOUND = 3.0  # must match `_grid_op`'s `bound`

# (name, xmax, ymax, grid_res). The y extents are deliberately chosen so that
# (ymax-ymin)/dx is NOT an integer for most of them — that non-integer edge is
# where an int-index wall test goes wrong, and 120/300*768 = 307.2 is the real
# apfsds_vs_era deck.
DECKS = [
    ("era-like", 300.0, 120.0, 768),
    ("square", 200.0, 200.0, 512),
    ("tall", 300.0, 160.0, 768),
    ("awkward", 250.0, 97.5, 640),
]


def _geom(xmax, ymax, grid_res):
    """The exact arithmetic `bake` does. Kept in one place so a drift shows up here."""
    dx = xmax / grid_res
    inv_dx = 1.0 / dx
    nx = grid_res + 3
    ny = int(math.ceil(ymax * inv_dx)) + 3
    edge = (xmax * inv_dx, ymax * inv_dx)
    clamp_lo = (dx, dx)
    clamp_hi = (xmax - dx, ymax - dx)
    return dx, inv_dx, nx, ny, edge, clamp_lo, clamp_hi


def _stencil(pos, inv_dx):
    """The node range `_p2g`/`_g2p` touch for a particle at `pos` — base .. base+2."""
    Xp = pos * inv_dx
    base = math.floor(Xp - 0.5)
    return base, base + 2


@pytest.mark.parametrize("name,xmax,ymax,grid_res", DECKS)
def test_every_wall_is_reachable_by_material_the_clamp_allows(name, xmax, ymax,
                                                              grid_res):
    """All four wall bands must contain a node some clamped particle can reach.

    THE test. It compares the wall band to where the POSITION CLAMP lets material
    go, which is the only definition of "reachable" that means anything. Against the
    old `i > nx - bound` band both high walls fail this outright.
    """
    dx, inv_dx, nx, ny, edge, clamp_lo, clamp_hi = _geom(xmax, ymax, grid_res)

    for axis, lo, hi, e in ((0, clamp_lo[0], clamp_hi[0], edge[0]),
                            (1, clamp_lo[1], clamp_hi[1], edge[1])):
        ax = "xy"[axis]
        # LOW wall: band is index < BOUND. The clamp's low stop is at `lo`.
        base, top = _stencil(lo, inv_dx)
        assert base < BOUND, (
            f"[{name}] {ax} LOW wall unreachable: a particle at the clamp "
            f"({lo:.3f}) touches nodes {base}..{top}, band is index < {BOUND}"
        )
        # HIGH wall: band is index > e - BOUND. The clamp's high stop is at `hi`.
        base, top = _stencil(hi, inv_dx)
        assert top > e - BOUND, (
            f"[{name}] {ax} HIGH wall unreachable: a particle at the clamp "
            f"({hi:.3f}) touches nodes {base}..{top}, band is index > "
            f"{e - BOUND:.2f}. THE PAD IS NOT THE DOMAIN — this is the milestone-13 "
            f"bug: the old test used the allocated size ({nx if axis == 0 else ny}) "
            f"instead of the domain edge ({e:.2f})."
        )


@pytest.mark.parametrize("name,xmax,ymax,grid_res", DECKS)
def test_the_high_wall_band_is_inside_the_domain_not_in_the_pad(name, xmax, ymax,
                                                                grid_res):
    """The band must sit in the material, not in the stencil pad.

    Distinct from reachability: a band could clip the domain by one node and pass
    the test above while still being mostly pad. The mirror is meant to be as thick
    on the high side as the low side's `index < 3`.
    """
    dx, inv_dx, nx, ny, edge, _, clamp_hi = _geom(xmax, ymax, grid_res)
    for axis, e in ((0, edge[0]), (1, edge[1])):
        ax = "xy"[axis]
        # Nodes in the band that are also inside the domain (index <= edge).
        in_band = [k for k in range(int(e) + 1) if k > e - BOUND]
        assert len(in_band) >= int(BOUND), (
            f"[{name}] {ax} HIGH band holds only {len(in_band)} in-domain node(s) "
            f"{in_band}; the low band holds {int(BOUND)} (0..{int(BOUND) - 1}). "
            f"The wall would be thinner on one side than the other."
        )


def _run_grid_op(nx, ny, edge, node, vel):
    """Launch the real `_grid_op` on a one-node grid state and return that node's v."""
    wp = pytest.importorskip("warp")
    try:
        wp.init()
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
    except Exception:  # pragma: no cover - no GPU/driver in this environment
        pytest.skip("warp could not initialise")
    gv = wp.zeros((nx, ny), dtype=wp.vec2, device=device)
    gm = wp.zeros((nx, ny), dtype=float, device=device)
    hv, hm = gv.numpy(), gm.numpy()
    # _grid_op divides momentum by mass; mass 1 makes velocity == momentum.
    hv[node[0], node[1]] = vel
    hm[node[0], node[1]] = 1.0
    gv.assign(hv)
    gm.assign(hm)
    wp.launch(mpm._grid_op, dim=(nx, ny), device=device,
              inputs=[gv, gm, wp.vec2(float(edge[0]), float(edge[1]))])
    wp.synchronize()
    return gv.numpy()[node[0], node[1]]


@pytest.mark.parametrize("side,axis,sign", [
    ("x-low", 0, -1.0), ("x-high", 0, +1.0),
    ("y-low", 1, -1.0), ("y-high", 1, +1.0),
])
def test_the_wall_actually_stops_inbound_material_on_every_side(side, axis, sign):
    """Behaviour, on the GPU, at a node the clamp can actually reach.

    `sign` is the INBOUND direction for that wall (into the wall = out of the
    domain). The node is the one a particle sitting on the position clamp touches —
    the same construction as the reachability test, so this cannot pass on a node
    only the pad can reach.
    """
    dx, inv_dx, nx, ny, edge, clamp_lo, clamp_hi = _geom(300.0, 120.0, 768)
    stop = (clamp_lo[axis] if sign < 0 else clamp_hi[axis])
    base, top = _stencil(stop, inv_dx)
    idx = base if sign < 0 else top
    node = [1, 1]
    node[axis] = idx

    # Inbound normal velocity, plus a tangential component that must SURVIVE.
    v = [0.0, 0.0]
    v[axis] = sign * 500.0
    v[1 - axis] = 250.0
    out = _run_grid_op(nx, ny, edge, node, v)

    assert out[axis] == 0.0, (
        f"{side} wall did NOT stop inbound material at node {tuple(node)}, which is "
        f"a node a particle on the position clamp ({stop:.3f}) touches. Without it "
        f"the clamp becomes the wall: it arrests displacement but not velocity, so "
        f"material piles onto the clamp plane at full speed and is crushed there."
    )
    assert out[1 - axis] == 250.0, (
        f"{side} wall is not free-SLIP: it killed the TANGENTIAL component "
        f"({out[1 - axis]} != 250). Slip walls are mirror planes; a sticky wall "
        f"would drag the armor's own material to a halt along the boundary."
    )


@pytest.mark.parametrize("side,axis,sign", [
    ("x-low", 0, -1.0), ("x-high", 0, +1.0),
    ("y-low", 1, -1.0), ("y-high", 1, +1.0),
])
def test_the_wall_lets_material_leave(side, axis, sign):
    """Outbound material passes: the wall kills only the component heading INTO it.

    Pins the half of the condition that is easy to over-fix. A wall that zeroed both
    directions would trap the spall spray against the boundary — the same pileup,
    arrived at from the opposite mistake.
    """
    dx, inv_dx, nx, ny, edge, clamp_lo, clamp_hi = _geom(300.0, 120.0, 768)
    stop = (clamp_lo[axis] if sign < 0 else clamp_hi[axis])
    base, top = _stencil(stop, inv_dx)
    idx = base if sign < 0 else top
    node = [1, 1]
    node[axis] = idx

    v = [0.0, 0.0]
    v[axis] = -sign * 500.0  # OUTBOUND: away from this wall, back into the domain
    out = _run_grid_op(nx, ny, edge, node, v)
    assert out[axis] == -sign * 500.0, (
        f"{side} wall blocked OUTBOUND material ({out[axis]}) — it must only kill "
        f"the inbound normal component."
    )


def test_the_walls_are_symmetric_because_the_decks_are():
    """Low and high bands must be the same thickness in material.

    The bug's signature was an ASYMMETRY in a setup that is symmetric by
    construction: `_seed` lays armor across the full domain height and the decks
    put the impact axis at mid-height, so a mirror on the bottom and a rigid clamp
    on the top is a physically different problem top vs bottom. It showed up in the
    bakes as RHA standing off the bottom clamp (y=0.88, held by the working mirror)
    while jammed onto the top clamp plane (y=119.61).
    """
    dx, inv_dx, nx, ny, edge, clamp_lo, clamp_hi = _geom(300.0, 120.0, 768)
    for axis in (0, 1):
        ax = "xy"[axis]
        lo_base, _ = _stencil(clamp_lo[axis], inv_dx)
        _, hi_top = _stencil(clamp_hi[axis], inv_dx)
        lo_depth = BOUND - lo_base            # how far into the band the low stop sits
        hi_depth = hi_top - (edge[axis] - BOUND)
        assert lo_depth == pytest.approx(hi_depth, abs=1.0), (
            f"{ax}: low stop sits {lo_depth:.2f} node(s) into its band, high stop "
            f"{hi_depth:.2f}. The two walls are not the same wall."
        )
