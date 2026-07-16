"""Scenario schema — scenarios are *data, not code* (CLAUDE.md §9).

A scenario deck (YAML under ``scenarios/``) is parsed into these dataclasses.
Kernels must not hardcode scenario specifics; they read them from here.

This module is intentionally dependency-light (stdlib + PyYAML) so the schema
can be inspected without importing Taichi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Domain:
    """World-space simulation bounds, in the manifest's unit system (mm)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def validate(self) -> None:
        if not (self.xmax > self.xmin and self.ymax > self.ymin):
            raise ValueError(f"degenerate domain: {self}")


@dataclass
class Projectile:
    """The incoming shell: a KE penetrator or a shaped-charge jet."""

    kind: str  # "kinetic" | "heat_jet"
    material: str  # key into the material library (materials.py)
    # Geometry is illustrative/representative only (CLAUDE.md §10).
    length: float  # mm
    diameter: float  # mm
    velocity: float  # m/s (== mm/ms in the mm-ms-g system)
    angle_deg: float = 0.0  # obliquity from horizontal
    # Nose profile. A real APFSDS long rod is POINTED, not flat-faced. Note what
    # this does and does not buy: the nose exists mainly for flight aerodynamics
    # and initial bite, and at ordnance velocity it is consumed within the first
    # microsecond, after which penetration is the eroding/hydrodynamic regime
    # (Tate-Alekseevskii, PHYSICS §2) in which final depth is nearly
    # nose-shape-independent. So this buys a plausible initial crater and honest
    # bite-vs-skid behaviour at obliquity — NOT a change in penetration accuracy.
    # Profiles are illustrative, not any real system's geometry (root §10).
    nose_shape: str = "conical"  # "conical" | "ogive" | "blunt"
    nose_length: float | None = None  # mm; None = 1.5 calibers (1.5 * diameter)
    # Height (mm) at which the rod tip meets the armor face. None = mid-domain,
    # which is what normal-incidence decks want. An oblique rod plunges in -y as
    # it advances, so those decks aim high and spend the domain below the impact
    # rather than wasting half of it as unused headroom.
    impact_y: float | None = None

    @property
    def nose_len(self) -> float:
        """Nose length in mm, resolving the `None` default to 1.5 calibers.

        1.5 rather than the 2-3 calibers a real long rod carries: at this deck's
        illustrative L/D of 7.5 (real APFSDS is ~20-30) a 2-caliber nose would
        eat a quarter of the rod, and the nose is carved OUT of the rod rather
        than added in front of it, so a longer nose is paid for in lost mass.
        """
        return 1.5 * self.diameter if self.nose_length is None else self.nose_length


NOSE_SHAPES = ("conical", "ogive", "blunt")


@dataclass
class ArmorLayer:
    """One layer in the target stack, front to back."""

    material: str  # key into the material library
    thickness: float  # mm
    standoff: float = 0.0  # mm of air in front of this layer (for spaced/NERA)


@dataclass
class SolverParams:
    """Numerics. `dt` is the substep dt; it is NOT frame_dt (CLAUDE.md §6)."""

    grid_resolution: int = 512  # background grid cells along the long axis
    particles_per_cell: int = 4
    dt: float = 1.0e-7  # substep dt (SI seconds by convention here) — CFL-bound
    total_time: float = 4.0e-5  # simulated seconds to bake (~tens of microseconds)
    frame_count: int = 90  # render frames dumped (target 60-120)


@dataclass
class Scenario:
    """A full input deck: projectile + armor stack + numerics + domain."""

    name: str
    domain: Domain
    projectile: Projectile
    armor: list[ArmorLayer]
    solver: SolverParams = field(default_factory=SolverParams)

    def validate(self) -> None:
        self.domain.validate()
        if not self.armor:
            raise ValueError("scenario has no armor layers")
        if self.solver.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        iy = self.projectile.impact_y
        if iy is not None and not (self.domain.ymin < iy < self.domain.ymax):
            raise ValueError(
                f"impact_y={iy} is outside the domain "
                f"(ymin={self.domain.ymin}, ymax={self.domain.ymax})"
            )
        proj = self.projectile
        if proj.nose_shape not in NOSE_SHAPES:
            raise ValueError(
                f"nose_shape={proj.nose_shape!r} is not one of {NOSE_SHAPES}"
            )
        # The nose is carved out of the rod, so it cannot outrun the rod itself.
        if not 0.0 <= proj.nose_len < proj.length:
            raise ValueError(
                f"nose_length={proj.nose_len} must be >= 0 and shorter than the "
                f"rod length={proj.length}"
            )


def load_scenario(path: str | Path) -> Scenario:
    """Parse a YAML deck into a validated :class:`Scenario`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    scenario = Scenario(
        name=raw.get("name", path.stem),
        domain=Domain(**raw["domain"]),
        projectile=Projectile(**raw["projectile"]),
        armor=[ArmorLayer(**layer) for layer in raw["armor"]],
        solver=SolverParams(**raw.get("solver", {})),
    )
    scenario.validate()
    return scenario
