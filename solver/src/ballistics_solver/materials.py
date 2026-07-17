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
#
# MILESTONE 13 KEPT THIS, in a smaller role. Murnaghan is no longer the EOS: it is
# the POLE GUARD's fallback branch, used only below `mpm.MG_F_SWITCH` of the way to
# the Hugoniot's pole (see mpm._mg_pressure). K'=4 still sets that branch's shape.
EOS_KP: float = 4.0

# --- Mie-Grueneisen shock EOS (milestone 13, PHYSICS §3.10) -----------------
# The volumetric law is now
#
#     p(J,e) = p_H(eta)*(1 - Gamma0*eta/2)  +  Gamma0*rho0*e        eta = 1 - J
#     p_H(eta) = rho0*c0^2*eta / (1 - s*eta)^2                      (shock Hugoniot)
#
# GAMMA CLOSURE, chosen on purpose: Gamma/V = Gamma0/V0, i.e. Gamma*rho = Gamma0*rho0
# (equivalently Gamma = Gamma0*J). This is NOT a free choice made alongside the
# formula — it is the assumption that DERIVES the formula, so the cold factor
# (1 - Gamma0*eta/2) and the thermal term (Gamma0*rho0*e) must share it. Constant-Gamma
# would be a different law. Note the standard identity `p(J, e_H) = p_H` holds for
# EITHER closure and will therefore NOT catch a mistake here.
#
# `c0` IS NOT A FIELD, DELIBERATELY: it is derived as c0 = sqrt(K0/rho0) with
# K0 = lam+mu, the same K0 milestone 8 used. Two reasons, and the second is the
# important one:
#   * It costs zero new constants, and
#   * it keeps the EOS TANGENT-MATCHED AT REST: K(1) = rho0*c0^2 = K0 exactly, which
#     is what makes milestone 13 a LARGE-STRAIN-ONLY change (a 1600 m/s KE deck barely
#     moves; a 7 km/s jet tip moves a lot). Reading c0 from a shock table instead would
#     silently break that and move every deck.
# It is corroborated, not assumed: the derived c0 lands at 0.99x copper's public bulk
# sound speed (3902 vs 3940 m/s), 1.06x RHA's, 1.10x tungsten's. Nothing was tuned.
#
# So Mie-Grueneisen costs TWO new per-material constants, not three. Both are public,
# textbook, order-of-magnitude values (root §10) and neither is tuned to an outcome:
#
#   shock_s          slope of the linear u_s = c0 + s*u_p fit. Clusters ~1.2-1.5 for
#                    metals, ~1.5-2.5 for polymers. It also fixes the Hugoniot's POLE
#                    at J = 1 - 1/s, which is why it is the constant that decides
#                    where the guard lives (mpm.MG_F_SWITCH).
#   gruneisen_gamma  Gamma0, the Grueneisen coefficient at the reference state. ~2 for
#                    most metals; it scales the thermal pressure Gamma0*rho0*e, i.e.
#                    how strongly shock heating stiffens the material.


@dataclass(frozen=True)
class ShockEOS:
    """Public Hugoniot fit constants for a material. See the block above."""

    s: float  # u_s = c0 + s*u_p slope; pole at J = 1 - 1/s
    gamma0: float  # Grueneisen coefficient at the reference state


@dataclass(frozen=True)
class Material:
    """Illustrative material parameters in the mm-ms-g unit system."""

    name: str
    # One-line human prose, carried to the cache as `material_descriptions`
    # (CACHE_FORMAT §2.1) so the viewer can say what it is drawing — it knows only
    # the cache format (root §2) and has no other way to find out.
    #
    # REQUIRED — no default, for the same reason `shock` has none: a new material
    # must say what it is rather than inheriting silence. It is the cheapest field
    # here to fill in and the only one a reader sees directly.
    #
    # Keep it to ~70 characters: this renders in a HUD panel beside a color
    # swatch, not in a document. What it is and why it is in the deck — never a
    # restatement of the constants below, which are right there, and never a real
    # system's specification (root §10).
    description: str
    material_id: int  # stored as float32 in the cache; see CACHE_FORMAT §2
    density: float  # g/mm^3
    youngs_modulus: float  # MPa
    poisson_ratio: float
    yield_strength: float  # MPa (von Mises); also the fracture strength if brittle
    damage_threshold: float  # ductile spall: equiv. plastic strain at detachment
    #                          (ignored for brittle materials — they use a stress trigger)
    # Mie-Grueneisen shock constants (milestone 13). REQUIRED — no default, on purpose:
    # a new material must state its Hugoniot rather than silently inherit copper's.
    shock: ShockEOS
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
    # shock=: tungsten's public Hugoniot fit. s~1.24 / Gamma0~1.54 are pure-W values;
    # this rod is a 17.6 g/mm^3 heavy alloy rather than pure W (19.3), so they are
    # representative rather than exact — which is the standard this repo works to
    # (root §10). Pole at J = 1 - 1/1.24 = 0.194, the deepest of any material here.
    "tungsten_rod": Material(
        name="tungsten_rod", material_id=0,
        description="Tungsten heavy alloy: very dense, tough KE long-rod penetrator.",
        density=17.6e-3, youngs_modulus=3.9e5, poisson_ratio=0.28,
        yield_strength=1.5e3, damage_threshold=2.0,
        shock=ShockEOS(s=1.24, gamma0=1.54),
    ),
    # shock=: representative steel. Public fits scatter (iron ~1.92, 4340 ~1.33), so
    # 1.49 is a mid-range steel value, not a spec-sheet number for any alloy.
    "rha": Material(
        name="rha", material_id=1,
        description="Rolled homogeneous armor: the baseline steel plate.",
        density=7.85e-3, youngs_modulus=2.0e5, poisson_ratio=0.29,
        yield_strength=1.0e3, damage_threshold=0.8,
        shock=ShockEOS(s=1.49, gamma0=1.93),
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
    # shock=: copper is the best-characterised Hugoniot in the public literature and
    # the one this repo measures against (PHYSICS §3.5/§3.10). s=1.489, Gamma0=1.99.
    # Pole at J = 1 - 1/1.489 = 0.328 — real copper essentially cannot be compressed
    # past that, and the jet tip runs closest to its own pole of anything here.
    "copper_jet": Material(
        name="copper_jet", material_id=6,
        description="Copper shaped-charge jet: soft enough to flow hydrodynamically.",
        density=8.96e-3, youngs_modulus=1.17e5, poisson_ratio=0.34,
        yield_strength=2.0e2, damage_threshold=1.5,
        shock=ShockEOS(s=1.489, gamma0=1.99),
    ),
    # shock=: representative alumina-like ceramic (s~1.0, Gamma0~1.3). s=1.0 puts the
    # pole at J=0, i.e. this material has NO pole in the physical range — the guard is
    # unreachable for it by construction. That is a real property of low-s materials,
    # not a special case. Ceramics are poorly described by a linear u_s-u_p fit
    # anyway; it does not matter here, because §3.5 predicted a priori (and measured)
    # that ceramic fails at J~0.98, where every monotone volumetric law agrees to <1 %.
    "ceramic": Material(
        name="ceramic", material_id=2,
        description="Alumina-like ceramic: stiff and hard, shatters with little flow.",
        density=3.9e-3, youngs_modulus=3.7e5, poisson_ratio=0.22,
        yield_strength=3.0e3, damage_threshold=0.05, brittle=True,
        shock=ShockEOS(s=1.0, gamma0=1.30),
    ),
    "era_filler": Material(
        name="era_filler", material_id=3,
        description="Explosive reactive filler: ignites on shock, flings the plates apart.",
        density=1.6e-3, youngs_modulus=5.0e3, poisson_ratio=0.40,
        yield_strength=0.05e3, damage_threshold=0.02,
        shock=ShockEOS(s=2.0, gamma0=1.0),
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
        description="Inert ERA twin: same mass and stiffness, never detonates (A/B baseline).",
        density=1.6e-3, youngs_modulus=5.0e3, poisson_ratio=0.40,
        yield_strength=0.05e3, damage_threshold=0.02,
        shock=ShockEOS(s=2.0, gamma0=1.0),
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
        description="NERA elastomer interlayer: yields and flows, but stays cohesive.",
        density=1.6e-3, youngs_modulus=5.0e3, poisson_ratio=0.40,
        yield_strength=0.05e3, damage_threshold=3.0,  # both LIVE — see above
        shock=ShockEOS(s=2.0, gamma0=1.0),
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


def id_to_description() -> dict[str, str]:
    """Map stringified material_id -> prose, for `material_descriptions`.

    Keyed identically to :func:`id_to_name` — CACHE_FORMAT §2.1 requires the two
    maps to share a key set, and building both from the same LIBRARY is what makes
    that true by construction rather than by discipline.
    """
    return {str(m.material_id): m.description for m in LIBRARY.values()}
