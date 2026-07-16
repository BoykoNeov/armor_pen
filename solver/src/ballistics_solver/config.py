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
    # Height (mm) at which the rod tip meets the armor face. None = mid-domain,
    # which is what normal-incidence decks want. An oblique rod plunges in -y as
    # it advances, so those decks aim high and spend the domain below the impact
    # rather than wasting half of it as unused headroom.
    impact_y: float | None = None


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
