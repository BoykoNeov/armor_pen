"""Smoke tests for the scenario schema. No GPU, no Taichi required.

Run: cd solver && pip install -e ".[dev]" && pytest
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ballistics_solver.config import Domain, Projectile, load_scenario

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _rod(**overrides) -> Projectile:
    kw = dict(kind="kinetic", material="tungsten_rod", length=60.0,
              diameter=8.0, velocity=1600.0)
    return Projectile(**(kw | overrides))


@pytest.mark.parametrize("deck", ["apfsds_vs_rha.yaml", "heat_vs_composite.yaml"])
def test_shipped_scenarios_load_and_validate(deck: str) -> None:
    scenario = load_scenario(SCENARIOS / deck)
    assert scenario.armor, "scenario must have at least one armor layer"
    assert scenario.solver.frame_count > 0
    scenario.validate()  # must not raise


def test_degenerate_domain_rejected() -> None:
    with pytest.raises(ValueError):
        Domain(xmin=0, xmax=0, ymin=0, ymax=10).validate()


def test_rod_is_pointed_by_default() -> None:
    """A real APFSDS is sharp, so every deck gets a nose without asking."""
    assert _rod().nose_shape == "conical"
    assert _rod().nose_len == 12.0  # 1.5 calibers on an 8 mm rod
    assert _rod(nose_length=5.0).nose_len == 5.0  # explicit wins
    for deck in SCENARIOS.glob("*.yaml"):
        assert load_scenario(deck).projectile.nose_shape != "blunt", deck.stem


@pytest.mark.parametrize("bad", [{"nose_shape": "pointy"}, {"nose_length": 60.0},
                                 {"nose_length": 99.0}, {"nose_length": -1.0}])
def test_bad_nose_rejected(bad: dict) -> None:
    from ballistics_solver.config import ArmorLayer, Scenario
    scenario = Scenario(
        name="t",
        domain=Domain(xmin=0, xmax=300, ymin=0, ymax=120),
        projectile=_rod(**bad),
        armor=[ArmorLayer(material="rha", thickness=40.0)],
    )
    with pytest.raises(ValueError):
        scenario.validate()
