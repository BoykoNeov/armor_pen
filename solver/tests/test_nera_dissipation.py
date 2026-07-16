"""Pin the NERA filler's dissipation path — the properties, not the values.

Milestone 12. `nera_filler` used to be `reactive=True` with `ignition_compression=0`:
a filler that never ignites. That was a mis-encoding. `reactive=True` exists to run
the ERA state machine, but mpm.py ALSO uses the flag to gate out `_return_mapping`
(plasticity) and `_update_damage` (ductile spall). The stated reason for those gates
is that a filler "must not spall before it detonates" — which cannot apply to a
filler that never detonates. So nera_filler inherited a gate written for its igniting
twin and ended up with NO dissipation path at all: it could neither yield nor break.

These tests pin the three claims that fix rests on. They are properties of the
material library and the kernel gates, so they run on CPU with no bake and no GPU.

What is deliberately NOT tested here: the bulge, the spall fraction, and worst-live-J.
Those are bake outcomes measured in PHYSICS §3.3/§3.6 — a unit test that pinned them
would just be re-encoding numbers that a geometry change is allowed to move.
"""

import pytest

from ballistics_solver import materials


def test_nera_filler_is_not_reactive() -> None:
    """The whole fix: non-reactive => mpm.py stops gating it out of dissipation.

    Both gates key off `reactive > 0.5` (`_return_mapping`, `_update_damage`). If
    this flips back to True, the filler silently loses plasticity AND spall again
    and PHYSICS §3.6's defect returns with no other symptom.
    """
    assert materials.get("nera_filler").reactive is False


def test_nera_filler_yield_and_damage_are_live_not_dead() -> None:
    """Both fields must be consumable. They were DEAD before milestone 12.

    A zero/absent yield would skip `_return_mapping` (it returns early on ys<=0),
    and a zero damage_threshold would skip the spall latch (`dthr > 0.0`), which is
    the same no-dissipation state by a different route.
    """
    m = materials.get("nera_filler")
    assert m.yield_strength > 0.0, "yield_strength=0 => _return_mapping returns early"
    assert m.damage_threshold > 0.0, "damage_threshold=0 => the spall gate never fires"


def test_nera_is_single_variable_against_era_filler_inert() -> None:
    """The A/B the deck header asked for: differ in EXACTLY one field.

    apfsds_vs_nera.yaml's header says a genuine cohesion test "would be a NON-reactive
    filler with a high damage_threshold". That is only a single-variable test if
    nera_filler and era_filler_inert agree on everything else — same mass, same
    stiffness, same yield, same reactivity — so a nera/era_inert delta isolates
    COHESION rather than "there is simply a different material in the path".

    material_id is excluded: it is a cache label (the viewer colours by it), not
    physics.
    """
    nera = materials.get("nera_filler")
    inert = materials.get("era_filler_inert")

    for field in ("density", "youngs_modulus", "poisson_ratio", "yield_strength",
                  "brittle", "reactive"):
        assert getattr(nera, field) == getattr(inert, field), (
            f"{field} differs — the nera/era_inert A/B is no longer single-variable"
        )

    assert nera.damage_threshold != inert.damage_threshold, (
        "damage_threshold must be the ONE difference — it is what cohesion means here"
    )
    assert nera.damage_threshold > inert.damage_threshold, (
        "nera is the COHESIVE arm: it must tolerate more plastic strain before tearing"
    )


def test_nera_carries_no_dead_reactive_payload() -> None:
    """A non-reactive filler must not keep detonation fields that nothing consumes.

    `_update_reactive` returns immediately on `reactive < 0.5`, so any non-zero value
    here would be a dead field — exactly the class of defect milestone 12 removed.
    """
    m = materials.get("nera_filler")
    assert m.detonation_pressure == 0.0
    assert m.burn_time == 0.0
    assert m.ignition_compression == 0.0


@pytest.mark.parametrize("name", ["era_filler"])
def test_reactive_gate_still_protects_the_igniting_filler(name: str) -> None:
    """The gate was right for the material it was written for — don't over-correct.

    era_filler MUST stay reactive: it ignites, and `_return_mapping`/`_update_damage`
    are gated out for it so the soft filler cannot spall in the same substep it should
    detonate (which would silently no-op the detonation — see the reactive note in
    mpm.py). Milestone 12 narrows that gate's population; it does not remove it.
    """
    m = materials.get(name)
    assert m.reactive is True
    assert m.ignition_compression > 0.0, "an igniting filler needs a live trigger"
