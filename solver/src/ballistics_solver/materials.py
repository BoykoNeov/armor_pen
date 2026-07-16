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
"""

from __future__ import annotations

from dataclasses import dataclass


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
    # NERA (non-explosive reactive armor) interlayer: a soft elastic filler that
    # NEVER detonates but stays cohesive, so the sandwich plates bulge apart on
    # the shock alone and the bulge is *held open* rather than collapsing.
    #
    # Same mass/stiffness as era_filler / era_filler_inert, so all three are the
    # equal-areal-mass arms of one A/B family; they differ ONLY in the filler's
    # response path:
    #   era_filler       reactive, ignites   -> detonation overpressure
    #   era_filler_inert non-reactive        -> plasticity + ductile spall at
    #                                           damage_threshold=0.02, i.e. the
    #                                           soft filler shreds and gets out
    #                                           of the way
    #   nera_filler      reactive, no ignite -> mpm.py skips BOTH return-mapping
    #                                           and the ductile-spall gate for
    #                                           reactive particles, so this stays
    #                                           soft-elastic and cohesive forever
    #
    # `ignition_compression=0` is what makes it never ignite (`_update_reactive`
    # gates ignition on `ic > 0`). That, NOT `detonation_pressure=0`, is the
    # persistent-bulge branch: zero pressure still ignites on the impact shock,
    # latches the particle spent, and collapses it to limp debris.
    #
    # DEAD FIELDS — do not tune these here, they are not consumed. `reactive=True`
    # makes mpm.py skip BOTH `_return_mapping` (plasticity) and `_update_damage`
    # (ductile spall) for this particle. So `yield_strength` and
    # `damage_threshold` below are inert, kept only for parity with the twins:
    # this filler is governed by E / nu / density alone. Consequence to be honest
    # about (PHYSICS §3.3): it can neither yield nor break, so it stores elastic
    # energy without dissipating it — stiffer-than-real, and any "the cohesive
    # filler resists the rod better" A/B against era_filler_inert is confounded by
    # that, not just by cohesion. Verified stable (thickness plateaus, no NaN).
    "nera_filler": Material(
        name="nera_filler", material_id=5,
        density=1.6e-3, youngs_modulus=5.0e3, poisson_ratio=0.40,
        yield_strength=0.05e3, damage_threshold=0.02,  # both DEAD — see above
        reactive=True,
        detonation_pressure=0.0, burn_time=0.0, ignition_compression=0.0,
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
