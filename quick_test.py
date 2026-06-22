#!/usr/bin/env python3
"""Quick integration test to verify all modules can be imported."""

import sys

def test_imports():
    """Test that all major modules can be imported."""
    errors = []

    modules_to_test = [
        # Models
        ("realestate_benchmark.models.interface", "ModelInterface"),
        ("realestate_benchmark.models.anthropic", "AnthropicModel"),
        ("realestate_benchmark.models.mock", "MockModel"),

        # Environment
        ("realestate_benchmark.environment.state", "GameState"),
        ("realestate_benchmark.environment.database", "Database"),
        ("realestate_benchmark.environment.controller", "GameController"),

        # Data
        ("realestate_benchmark.data.ames", "load_ames_data"),
        ("realestate_benchmark.data.properties", "partition_property"),
        ("realestate_benchmark.data.descriptions", "generate_description"),

        # Agents
        ("realestate_benchmark.agents.memory", "Memory"),
        ("realestate_benchmark.agents.base", "ReActAgent"),
        ("realestate_benchmark.agents.seller", "SellerAgent"),
        ("realestate_benchmark.agents.buyer", "BuyerAgent"),

        # Tools
        ("realestate_benchmark.tools.registry", "ToolRegistry"),

        # Evaluation
        ("realestate_benchmark.evaluation.p1_informational", "compute_p1_commission"),
        ("realestate_benchmark.evaluation.p2_welfare", "compute_p2_welfare_gap"),
    ]

    for module_name, symbol in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[symbol])
            getattr(module, symbol)
            print(f"✓ {module_name}.{symbol}")
        except Exception as e:
            error_msg = f"✗ {module_name}.{symbol}: {e}"
            print(error_msg)
            errors.append(error_msg)

    if errors:
        print(f"\n❌ {len(errors)} import errors found")
        return 1
    else:
        print(f"\n✅ All {len(modules_to_test)} imports successful!")
        return 0

if __name__ == "__main__":
    sys.exit(test_imports())
