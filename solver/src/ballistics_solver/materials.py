"""Material library — every physical constant, defined once, in mm-ms-g.

CLAUDE.md §7: never mix raw SI into kernels. All constants here are in the
mm-ms-g system (density g/mm^3, modulus MPa). Values are **representative and
illustrative**, order-of-magnitude from public literature (CLAUDE.md §10) —
not spec-sheet numbers for any real system.

STATUS: data library, fully consumed by mpm.py. Every field drives the solver:
elasticity (``youngs_modulus``/``poisson_ratio``), ``yield_strength`` (von Mises
return mapping — and the fracture strength for brittle materials), ductile
``damage_threshold`` (plastic-strain spall), ``brittle`` (stress-triggered
shatter), and the reactive block (``reactive`` + ``detonation_pressure`` /
``burn_time`` / ``ignition_compression``) that drives the ERA/NERA impulse layer
(milestone 5). See mpm.py and docs/PHYSICS.md §3.

Note there is no "fluid" material and none is needed: the shaped-charge jet
(``copper_jet``, milestone 7) flows hydrodynamically purely because its yield is
~1000x below its own stagnation pressure, so the ordinary von Mises return mapping
caps its deviatoric stress near zero. What makes a jet a jet lives in the *deck*
(``Projectile.tail_velocity`` — a velocity gradient), not here.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Equation of state (milestone 8, PHYSICS §3.5) --------------------------
# Pressure derivative of the bulk modulus, K' = dK/dp at p=0, for the Murnaghan
# EOS that supplies the volumetric stress in mpm.py:
#
#     p(J) = (K0/K') (J^-K' - 1)        K0 = lam + mu, from E/nu (see mpm._lame)
#
# K' ~ 4 is the textbook value for metals and most dense solids, and it is close
# enough to universal that it is the standard default when a material's own value
# is unmeasured — which is exactly our situation (root §10: representative, not
# spec-sheet). One number, shared by every material, so the EOS costs ZERO new
# per-material constants: K0 already follows from the elastic moduli we have.
#
# Deliberately a module constant rather than a `Material` field: nothing here
# varies it, and a per-particle array would thread a parameter through five
# kernel signatures to carry the same 4.0 everywhere. Promote it to a field the
# day a material genuinely needs its own (soft polymer fillers are the plausible
# case at ~7-12, but they ignite at J=0.98 and never reach compressions where K'
# is distinguishable).
EOS_KP: float = 4.0


@dataclass(frozen=True)
class Material:
    """Illustrative material parameters in the mm-ms-g unit system."""

    name: str
    material_id: int  # stored as float32 in the cache; see CACHE_FORMAT §2
    density: float  # g/mm^3
    youngs_modulus: float  # MPa
    poisson_ratio: float
    yield_strength: float  # MPa (von Mises); also the fracture strength if brittle
    damage_threshold: float  # ductile spall: equiv. plastic strain at detachment
    #                          (ignored for brittle materials — they use a stress trigger)
    brittle: bool = False  # ceramics/composites: shatter on stress, ~zero plastic flow

    # --- Reactive block (ERA/NERA, milestone 5) -----------------------------
    # A reactive filler is an impulse layer that degrades the penetrator: when
    # the impact shock reaches it, it ignites and releases an isotropic pressure
    # that flings the sandwiching plates apart (an ERA detonation). Modelled as a
    # source term in mpm.py — reactive particles bypass the ductile-spall path
    # entirely and run their own elastic → detonation-pressure → debris state
    # machine, so this must not be confused with `brittle`/`damage_threshold`.
    # A persistent NERA-style inert bulging layer is a filler that NEVER ignites
    # (`ignition_compression=0`, so it stays on the soft-elastic branch and the
    # plates move from the shock alone) — NOT merely `detonation_pressure=0`,
    # which still ignites on the impact shock and then collapses to limp debris.
    reactive: bool = False  # ERA/NERA filler: ignites and drives an impulse
    detonation_pressure: float = 0.0  # MPa, isotropic outward pulse while burning
    burn_time: float = 0.0  # ms the pulse lasts once ignited (physical time, not steps)
    ignition_compression: float = 0.0  # ignites when det(F) drops below this (0 = never)


# Representative library. Numbers are public-literature order-of-magnitude.
LIBRARY: dict[str, Material] = {
    "tungsten_rod": Material(
        name="tungsten_rod", material_id=0,
        density=17.6e-3, youngs_modulus=3.9e5, poisson_ratio=0.28,
        yield_strength=1.5e3, damage_threshold=2.0,
    ),
    "rha": Material(
        name="rha", material_id=1,
        density=7.85e-3, youngs_modulus=2.0e5, poisson_ratio=0.29,
        yield_strength=1.0e3, damage_threshold=0.8,
    ),
    # Shaped-charge jet material (milestone 7). Representative COPPER — the
    # classic liner metal — as an ALREADY-FORMED jet: we seed the jet, never the
    # liner collapse or the explosive that drives it (out of scope by
    # construction, root §10, and unnecessary — the velocity gradient is the part
    # that matters downstream; see `Projectile.tail_velocity`).
    #
    # Density/modulus/Poisson are public textbook copper. `yield_strength` is the
    # interesting one: at a 7 km/s tip the stagnation pressure is ~0.5*rho*v^2 ~
    # 2e5 MPa, a THOUSAND times this yield, so strength is negligible and the jet
    # flows hydrodynamically without any special "fluid" path — von Mises return
    # mapping caps deviatoric stress near zero all by itself. That is why a jet
    # needs no new constitutive model here.
    #
    # HONEST LIMIT (PHYSICS §3.5): yield caps only the DEVIATORIC response. The
    # volumetric response is a separate law, and until milestone 8 it had no
    # equation of state at all — it *under*-resisted at extreme compression, so a
    # 220 GPa stagnation point equilibrated at J~0.15 where real copper gives
    # ~0.61. It is a Murnaghan EOS now (monotone, stiffening, K0 = lam+mu from the
    # moduli above, K' = EOS_KP), which measured the tip back to J~0.43. (Quote that
    # to two figures: it is dt-dependent, because what it now measures is the
    # undamped shock ring rather than the EOS — PHYSICS §3.5.)
    #
    # What is STILL true here: Murnaghan is a *cold* curve carrying no shock
    # heating, so against copper's public Hugoniot it still under-reads pressure —
    # ~0.93x at J=0.9 (a KE deck: negligible) but ~0.68x at a 7 km/s tip. The
    # error is smaller and better-behaved, not gone, and it still depends on
    # velocity. Lowering the yield never touched any of this. Plausible, not
    # predictive (root §1).
    #
    # `damage_threshold` is ductile copper's plastic-strain reserve, and it is the
    # particulation knob: a stretching jet accumulates plastic strain, and when it
    # crosses this the jet breaks into a fragment train. Set from copper's large
    # ductility rather than tuned to force particulation on cue (root §10).
    "copper_jet": Material(
        name="copper_jet", material_id=6,
        density=8.96e-3, youngs_modulus=1.17e5, poisson_ratio=0.34,
        yield_strength=2.0e2, damage_threshold=1.5,
    ),
    "ceramic": Material(
        name="ceramic", material_id=2,
        density=3.9e-3, youngs_modulus=3.7e5, poisson_ratio=0.22,
        yield_strength=3.0e3, damage_threshold=0.05, brittle=True,
    ),
    "era_filler": Material(
        name="era_filler", material_id=3,
        density=1.6e-3, youngs_modulus=5.0e3, poisson_ratio=0.40,
        yield_strength=0.05e3, damage_threshold=0.02,
        # Reactive: ignites at ~2% shock compression and releases a ~4 GPa pulse
        # for ~5 us. Illustrative order-of-magnitude (root §10): real detonation
        # pressures are ~10x higher, but this is tuned to fling a few-mm steel
        # flyer to a few hundred m/s, not to match an explosive formulation.
        reactive=True,
        detonation_pressure=4.0e3, burn_time=5.0e-3, ignition_compression=0.98,
    ),
    # Inert twin of era_filler: identical mass/stiffness, reactivity OFF. Exists
    # solely for the equal-areal-mass A/B baseline (reactive vs inert filler in
    # the same geometry), so the reactive contribution to penetrator degradation
    # can be isolated from "there is simply more material in the path".
    "era_filler_inert": Material(
        name="era_filler_inert", material_id=4,
        density=1.6e-3, youngs_modulus=5.0e3, poisson_ratio=0.40,
        yield_strength=0.05e3, damage_threshold=0.02,
        reactive=False,
    ),
    # NERA (non-explosive reactive armor) interlayer: a soft filler that carries no
    # explosive, so the sandwich plates bulge apart on the impact shock alone and
    # the cohesive interlayer *holds the bulge open* rather than collapsing.
    #
    # Same mass/stiffness as era_filler / era_filler_inert, so all three are the
    # equal-areal-mass arms of one A/B family; they differ ONLY in the filler's
    # response path:
    #   era_filler       reactive, ignites   -> detonation overpressure
    #   era_filler_inert non-reactive, ductile, damage_threshold=0.02
    #                                        -> the soft filler shreds and gets
    #                                           out of the way
    #   nera_filler      non-reactive, ductile, damage_threshold=3.0
    #                                        -> yields and flows, but survives:
    #                                           the cohesive bulge
    #
    # MILESTONE 12 CHANGED THIS MATERIAL. It used to be `reactive=True` with
    # `ignition_compression=0` — a filler that never ignites. That was a
    # mis-encoding: `reactive=True` exists to run the ERA state machine, but mpm.py
    # ALSO uses it to gate out `_return_mapping` (L538) and `_update_damage` (L653).
    # The stated reason for those gates is that a filler "must not spall before it
    # detonates" — which cannot apply to a filler that never detonates. nera_filler
    # inherited a gate written for its igniting twin, and the cost was that it could
    # neither yield nor break: no dissipation path at all.
    #
    # The fix is the one apfsds_vs_nera.yaml already prescribed in its own header —
    # "a NON-reactive filler with a high damage_threshold" — and it needs ZERO
    # kernel code. Both gates key off `reactive>0.5`, so going non-reactive turns
    # plasticity and the spall path back on; the raised threshold is what keeps the
    # interlayer cohesive rather than letting it shred like era_filler_inert.
    #
    # NOTHING PHYSICAL WAS LOST by dropping `reactive=True`, verified path by path:
    # `_update_reactive` was a true no-op here (ic=0, burn=0, damage=0 -> every
    # branch falls through) and `_p2g` takes the identical elastic stress term for
    # an unignited particle. ONE real difference: `_clamp_reactive_v` no longer caps
    # this filler at REACTIVE_VMAX. Measured on the pre-M12 bake, that clamp bound on
    # exactly ONE particle across frames 158-159 of 550 — negligible, but not zero,
    # and its justification (runaway from the F-independent detonation source) never
    # applied to a filler with no detonation.
    #
    # The fix removed the clamp's REASON, not just its effect, and the measurement is
    # a tidy statement of what was wrong before: pre-M12 the filler reached the full
    # 3000 mm/ms clamp — nearly 2x the 1600 mm/ms rod that drove it, because a solid
    # that cannot yield stores the shock and springs. Post-M12 it peaks at 1586, i.e.
    # 53 % of the removed clamp and just under the rod's own speed. A filler that
    # dissipates does not out-run the thing hitting it.
    #
    # BOTH FIELDS BELOW ARE NOW LIVE — they were dead before, and neither is tuned:
    #   yield_strength  50 MPa — unchanged from the value the dead field already
    #                   carried, deliberately NOT re-picked to land an answer.
    #   damage_threshold 3.0  — representative elastomer elongation-to-failure
    #                   (public order-of-magnitude, root §10). Real NERA interlayers
    #                   are rubber, which stretches hundreds of percent before it
    #                   tears; 3.0 is that reserve, ~150x era_filler_inert's 0.02.
    #                   It is NOT "infinity": debris dragged downrange and crushed
    #                   between the rod and the main plate SHOULD fail, and does.
    #
    # HONEST LIMIT (PHYSICS §3.3): a real elastomer dissipates viscoelastically, not
    # by von Mises plastic flow. Plasticity is the dissipation path this solver has,
    # and it is the right *kind* of thing (irreversible, isochoric, cohesion-
    # preserving) rather than the right constitutive model. Plausible, not
    # predictive (root §1).
    "nera_filler": Material(
        name="nera_filler", material_id=5,
        density=1.6e-3, youngs_modulus=5.0e3, poisson_ratio=0.40,
        yield_strength=0.05e3, damage_threshold=3.0,  # both LIVE — see above
        reactive=False,
    ),
}


def get(name: str) -> Material:
    """Look up a material by name, with a helpful error if it's missing."""
    try:
        return LIBRARY[name]
    except KeyError as exc:
        known = ", ".join(sorted(LIBRARY))
        raise KeyError(f"unknown material {name!r}; known: {known}") from exc


def id_to_name() -> dict[str, str]:
    """Map stringified material_id -> name, for the manifest `materials` field."""
    return {str(m.material_id): m.name for m in LIBRARY.values()}
