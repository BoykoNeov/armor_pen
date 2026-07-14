"""Material library — every physical constant, defined once, in mm-ms-g.

CLAUDE.md §7: never mix raw SI into kernels. All constants here are in the
mm-ms-g system (density g/mm^3, modulus MPa). Values are **representative and
illustrative**, order-of-magnitude from public literature (CLAUDE.md §10) —
not spec-sheet numbers for any real system.

STATUS: data-only stub. The constitutive kernels (elastic + von Mises return
mapping + damage detach) live in mpm.py and are not implemented yet.
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
    yield_strength: float  # MPa (von Mises)
    damage_threshold: float  # accumulated plastic strain at which a particle detaches
    brittle: bool = False  # ceramics/composites fail with little plastic flow


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
