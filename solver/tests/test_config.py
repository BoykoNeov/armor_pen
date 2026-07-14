"""Smoke tests for the scenario schema. No GPU, no Taichi required.

Run: cd solver && pip install -e ".[dev]" && pytest
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ballistics_solver.config import Domain, load_scenario

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


@pytest.mark.parametrize("deck", ["apfsds_vs_rha.yaml", "heat_vs_composite.yaml"])
def test_shipped_scenarios_load_and_validate(deck: str) -> None:
    scenario = load_scenario(SCENARIOS / deck)
    assert scenario.armor, "scenario must have at least one armor layer"
    assert scenario.solver.frame_count > 0
    scenario.validate()  # must not raise


def test_degenerate_domain_rejected() -> None:
    with pytest.raises(ValueError):
        Domain(xmin=0, xmax=0, ymin=0, ymax=10).validate()
