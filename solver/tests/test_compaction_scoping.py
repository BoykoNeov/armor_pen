"""Milestone 15 closed as a NEGATIVE: no volumetric/compaction criterion is added.

PHYSICS §3.12 is the writeup. This file pins the three RELATIONS that make the
negative true, so it cannot silently rot — every one is derived from
``materials.py`` and the deck files, and none restates a measured literal.

WHY A NEGATIVE NEEDS TESTS AT ALL. §3.6.2 asked for "a VOLUMETRIC criterion
(compaction/pore collapse), not a deviatoric one" and README carried it as the
next milestone. The answer is that a P-α model (Herrmann 1969; Carroll-Holt
1972) is INERT everywhere this repo could use it — but "inert" is a statement
about the current constants and decks, and constants move. If a deck lands slow
enough, or a filler tolerant enough, that compaction becomes live, these asserts
fire and the negative gets revisited rather than inherited.

THE CRUSH-UP PRESSURES BELOW ARE PUBLIC ORDER-OF-MAGNITUDE (root §10) AND WERE
WRITTEN DOWN BEFORE THE MEASUREMENT, not fitted to it. A P-α material is fully
compacted above its crush-up pressure ``p_c``: the distension α reaches 1, the
pores are gone, and the law reduces EXACTLY to the solid-matrix EOS. So
compaction can only change anything if the pressures in play are comparable to
``p_c``.

EACH ASSERT WAS VERIFIED TO FAIL FIRST, against the mutation that would make
compaction genuinely live — not against a scrambled constant
([[instruments-that-cannot-see-the-failure]]):

  * the contact-shock test: a deck slow enough to bring its shock into the
    crush-up band (verified with a temporary 150 mm/ms deck);
  * the ignition test: LOWERING ``era_filler.ignition_compression``, i.e. a
    filler that tolerates more shock before it ignites, which is exactly the
    filler whose pores WOULD finish collapsing first;
  * the impedance test: raising the filler's impedance toward its neighbours',
    which is what would make deck-wide sizing stop being the conservative choice.
"""
from pathlib import Path

import pytest

from ballistics_solver import config, materials, mpm

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
# Globbed, not listed — the same posture `test_cfl_sizing.py` uses, and for the same
# reason: a new deck must be covered the moment it lands. A SLOW deck is precisely
# what would make compaction live, so it must not be able to arrive unnoticed.
ALL_DECKS = sorted((p.stem, p) for p in SCENARIOS.glob("*.yaml"))

# Public order-of-magnitude crush-up (full-compaction) pressures, MPa.
#   polymer foam / syntactic     ~10-50    crush plateau of a light foam
#   pressed HE / powder compact  ~100-500  powder crush-up to solid density
#   porous metal                 ~0.5-1e3  porous Fe/Al, Herrmann's own materials
# The porous metal is the TOP of the plausible band, so it is the conservative
# number to test against: if even that is dwarfed, every softer candidate is too.
P_C_PRESSED_POWDER: float = 500.0
P_C_MAX_PLAUSIBLE: float = 1000.0

# How far above the crush-up band a pressure has to sit before "the pores are long
# gone" is a safe statement rather than a close call. 10x is not a tuned number: it
# is one decade, the granularity the crush-up values themselves are known to.
DECADE: float = 10.0


def _deck_contact_pressure(scenario) -> float:
    """The deck-wide worst contact pressure, UNMARGINED (MPa).

    Deliberately without ``EOS_CFL_P_MARGIN``: the claim being pinned is about the
    physical shock, and multiplying by a stability constant that has shipped at 3
    and 4 would make a physics test depend on a tunable. The margin only makes the
    gap larger, so dropping it is the conservative direction.

    YES, THIS RE-IMPLEMENTS ``bake``'s deck-wide arithmetic, which
    ``test_cfl_sizing.py`` warns against ("a test that recomputed ``Jd`` the way
    ``bake`` does would be satisfied by copying the bug"). That warning is about
    tests CHECKING THE SIZING CODE, where re-running the formula is circular. These
    tests check a MATERIAL-AND-DECK relation — is any shock in this repo near a
    crush-up pressure — and would hold whatever ``bake`` did with the number. If the
    deck-wide posture ever changes, ``test_cfl_sizing.py`` is the file that should
    notice; this one only needs a defensible upper bound on what the filler sees.
    """
    proj = materials.get(scenario.projectile.material)
    names = {scenario.projectile.material, *(a.material for a in scenario.armor)}
    v_tip = scenario.projectile.velocity
    return max(
        [mpm._impact_pressure(proj, materials.get(nm), v_tip) for nm in names]
        + [mpm._impact_pressure(proj, proj, v_tip)]
    )


@pytest.mark.parametrize("deck_name,deck_path", ALL_DECKS)
def test_contact_shock_is_a_decade_past_any_plausible_crush_up(deck_name, deck_path):
    """No deck's shock is anywhere near the pressure where compaction lives.

    This is the whole negative in one relation. Above ``p_c`` a P-α material IS its
    solid matrix, so a compaction model cannot move the design state, the CFL bound,
    or the pole guard's clearance — at ANY distension, for ANY choice of crush curve.
    """
    p = _deck_contact_pressure(config.load_scenario(deck_path))
    assert p >= DECADE * P_C_MAX_PLAUSIBLE, (
        f"{deck_name}: deck-wide contact shock is {p:.0f} MPa, within a decade of "
        f"the {P_C_MAX_PLAUSIBLE:.0f} MPa crush-up of the most resistant plausible "
        f"porous solid. Compaction may no longer be inert on this deck — PHYSICS "
        f"§3.12 closed milestone 15 on the premise that it is. Re-open it."
    )


def test_era_filler_ignites_before_its_pores_could_finish_collapsing():
    """The one material here where porosity would be PHYSICAL leaves first.

    A pressed/cast explosive is a genuinely heterogeneous solid with real porosity —
    unlike ``nera_filler``, a near-incompressible elastomer, where distension would
    have to be invented. But ``era_filler`` ignites at ``ignition_compression`` and
    hands off to the detonation state machine, and that happens BELOW the pressure
    at which a pressed powder's pores finish collapsing. So the compaction branch
    would act over a sliver and then be overwritten.

    THE 191.8-vs-500 GAP IS NOT THE SAFETY MARGIN — do not read it as one. ``p_ign``
    here is a COLD pressure (``_mg_p_cold_host`` omits the thermal term ``Gamma0*rho0*e``),
    and ignition is a shock event, so the real pressure at handoff is HIGHER than this
    and the apparent 2.6x is thinner than it looks. That is the one direction a guard
    should not err in, so the argument is deliberately NOT the pressure ordering.

    THE LOAD-BEARING FACT IS THE HANDOFF ITSELF: ``_update_reactive`` ignites on
    ``det(F) < ignition_compression``, a VOLUME criterion, so era_filler leaves the
    ordinary constitutive path at J=0.98 whatever pressure that turns out to be. The
    assert below is the cheap, derived proxy that goes red when the handoff moves far
    enough to matter (see the red-check mutation: lowering ``ignition_compression``).
    """
    era = materials.get("era_filler")
    mu, lam = mpm._lame(era.youngs_modulus, era.poisson_ratio)
    mg = mpm._mg_params(era)
    p_ign = float(mpm._mg_p_cold_host(era.ignition_compression, lam + mu, mg))

    assert p_ign < P_C_PRESSED_POWDER, (
        f"era_filler now ignites at {p_ign:.0f} MPa, at or above the "
        f"{P_C_PRESSED_POWDER:.0f} MPa crush-up of a pressed powder. It would now "
        f"spend real time in the compaction regime before igniting, which is the "
        f"case PHYSICS §3.12 ruled out. Re-open milestone 15."
    )


def test_the_filler_is_sized_by_its_neighbours_not_by_its_own_impedance():
    """Why ``era_filler``'s 0.0004 guard clearance is a POSTURE, not a property.

    ``bake`` sizes every material by the DECK's worst contact pressure, because a
    confined soft layer is crushed by its stiff neighbours rather than by its own
    tiny impedance (see the sizing block in ``mpm.bake``). That posture is correct
    and milestone 14 defended it deliberately; this test pins the material fact that
    MAKES it correct, so the tightness of the ERA clearance is never mistaken for a
    statement about ``era_filler`` itself.

    Not a licence to switch to per-material sizing: M14 rejected that as backwards,
    and nothing here is evidence against it.
    """
    scen = config.load_scenario(SCENARIOS / "apfsds_vs_era.yaml")
    proj = materials.get(scen.projectile.material)
    era = materials.get("era_filler")
    v = scen.projectile.velocity

    own = mpm._impact_pressure(proj, era, v)
    deckwide = _deck_contact_pressure(scen)

    assert deckwide > 5.0 * own, (
        f"era_filler's confining neighbours no longer strike much harder than the "
        f"filler's own impedance match ({deckwide:.0f} vs {own:.0f} MPa). Deck-wide "
        f"sizing was conservative BECAUSE of that gap; without it, the ERA decks' "
        f"design state means something different. See PHYSICS §3.12."
    )
