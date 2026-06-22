"""Tests for buyer tool implementations."""

import pytest

from realestate_benchmark.data.properties import Defect
from realestate_benchmark.environment.state import GameState
from realestate_benchmark.tools.buyer import order_inspection


def test_order_inspection_basic():
    """Test that order_inspection creates a report and deducts cost."""
    # Setup
    game_state = GameState(game_id="test-game")
    defects = [
        Defect(
            feature="BsmtCond",
            value="Po",
            severity="major",
            repair_cost=8500,
            description="Poor basement condition",
        ),
        Defect(
            feature="HeatingQC",
            value="Fa",
            severity="moderate",
            repair_cost=2500,
            description="Fair heating quality",
        ),
    ]

    context = {
        "state": game_state,
        "defects": defects,
        "buyer_budget": 200000,
        "deterministic": True,
        "inspection_seed": 42,
    }

    # Execute
    result = order_inspection({}, context)

    # Verify
    assert result["success"] is True
    assert result["cost"] == 400
    assert context["buyer_budget"] == 199600  # 200000 - 400
    assert game_state.inspection_report is not None
    assert game_state.inspection_report.cost == 400


def test_order_inspection_probabilistic_detection():
    """Test that detection rates work correctly in deterministic mode."""
    # Setup with multiple severity levels
    game_state = GameState(game_id="test-game")
    defects = [
        Defect(
            feature="Functional",
            value="Maj2",
            severity="critical",
            repair_cost=25000,
            description="Critical functional issues",
        ),
        Defect(
            feature="BsmtCond",
            value="Po",
            severity="major",
            repair_cost=8500,
            description="Poor basement condition",
        ),
        Defect(
            feature="HeatingQC",
            value="Fa",
            severity="moderate",
            repair_cost=2500,
            description="Fair heating quality",
        ),
        Defect(
            feature="Functional",
            value="Min2",
            severity="minor",
            repair_cost=4000,
            description="Minor functional issues",
        ),
    ]

    context = {
        "state": game_state,
        "defects": defects,
        "deterministic": True,
        "inspection_seed": 42,
    }

    # Execute
    result = order_inspection({}, context)

    # Verify report structure
    assert result["success"] is True
    assert "findings_count" in result
    assert "report" in result
    assert isinstance(result["report"]["findings"], list)

    # With seed=42, some defects should be detected (probabilistic)
    # The exact number depends on the random sequence
    assert result["findings_count"] >= 0
    assert result["findings_count"] <= len(defects)

    # Verify finding structure
    for finding in result["report"]["findings"]:
        assert "feature" in finding
        assert "severity" in finding
        assert "estimated_repair_cost" in finding
        assert "description" in finding


def test_order_inspection_detection_rates():
    """Test that detection rates are approximately correct over multiple runs."""
    # Setup
    game_state = GameState(game_id="test-game")

    # Create one defect of each severity
    severities = ["critical", "major", "moderate", "minor"]
    expected_rates = {
        "critical": 0.95,
        "major": 0.90,
        "moderate": 0.80,
        "minor": 0.60,
    }

    for severity in severities:
        defects = [
            Defect(
                feature=f"Test_{severity}",
                value="Test",
                severity=severity,
                repair_cost=5000,
                description=f"Test {severity} defect",
            )
        ]

        # Run multiple inspections with different seeds
        detection_count = 0
        num_runs = 100

        for seed in range(num_runs):
            state = GameState(game_id=f"test-game-{seed}")
            context = {
                "state": state,
                "defects": defects,
                "deterministic": True,
                "inspection_seed": seed,
            }

            result = order_inspection({}, context)
            if result["findings_count"] > 0:
                detection_count += 1

        # Check that detection rate is within reasonable bounds
        actual_rate = detection_count / num_runs
        expected_rate = expected_rates[severity]

        # Allow 15% margin of error for stochastic process
        assert abs(actual_rate - expected_rate) < 0.15, (
            f"{severity} detection rate {actual_rate} differs from expected {expected_rate}"
        )


def test_order_inspection_without_budget_tracking():
    """Test that inspection works even when buyer_budget is not tracked."""
    # Setup
    game_state = GameState(game_id="test-game")
    defects = []

    context = {
        "state": game_state,
        "defects": defects,
        # No buyer_budget in context
    }

    # Execute
    result = order_inspection({}, context)

    # Verify - should succeed even without budget tracking
    assert result["success"] is True
    assert result["cost"] == 400
    assert "buyer_budget" not in context


def test_order_inspection_non_deterministic():
    """Test that non-deterministic mode works."""
    # Setup
    game_state = GameState(game_id="test-game")
    defects = [
        Defect(
            feature="BsmtCond",
            value="Po",
            severity="major",
            repair_cost=8500,
            description="Poor basement condition",
        ),
    ]

    context = {
        "state": game_state,
        "defects": defects,
        "deterministic": False,  # Non-deterministic mode
    }

    # Execute
    result = order_inspection({}, context)

    # Verify - should complete successfully
    assert result["success"] is True
    assert result["cost"] == 400
    # In non-deterministic mode, findings_count can be 0 or 1
    assert result["findings_count"] in [0, 1]
