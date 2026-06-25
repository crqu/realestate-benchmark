#!/usr/bin/env python3
"""Run an end-to-end real estate transaction experiment.

Usage:
    python experiments/run_experiment.py --project-id YOUR_PROJECT_ID
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realestate_benchmark.agents.buyer import BuyerAgent
from realestate_benchmark.agents.memory import Memory
from realestate_benchmark.agents.seller import SellerAgent
from realestate_benchmark.data.ames import get_dataset_info, get_property, load_ames_data
from realestate_benchmark.data.descriptions import generate_description
from realestate_benchmark.data.properties import (
    compute_true_value,
    extract_defects,
    partition_property,
)
from realestate_benchmark.environment.controller import GameController
from realestate_benchmark.environment.database import Database
from realestate_benchmark.evaluation.llm_judge import compute_llm_metrics
from realestate_benchmark.evaluation.p1_informational import (
    compute_p1_commission,
    compute_p1_framing,
    compute_p1_omission,
)
from realestate_benchmark.evaluation.p2_welfare import (
    compute_p2_nbs_deviation,
    compute_p2_welfare_gap,
)
from realestate_benchmark.evaluation.p3_influence import compute_p3_influence_legitimacy
from realestate_benchmark.evaluation.p4_norms import compute_p4_norm_compliance
from realestate_benchmark.evaluation.report import generate_game_report
from realestate_benchmark.models.vertex import VertexModel
from realestate_benchmark.tools.registry import create_registry


def create_judge_model(provider: str, model_name: str, project_id: str, region: str,
                       json_mode: bool = False):
    """Create a model instance for the LLM judge."""
    if provider == "gemini":
        from realestate_benchmark.models.gemini import GeminiModel
        return GeminiModel(
            project_id=project_id, region=region, model=model_name,
            json_mode=json_mode,
        )
    elif provider == "vertex":
        return VertexModel(project_id=project_id, region=region, model=model_name)
    elif provider == "anthropic":
        from realestate_benchmark.models.anthropic import AnthropicModel
        return AnthropicModel(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def select_property(df, property_type: str = "random"):
    if property_type == "random":
        return random.randint(0, len(df) - 1)

    for idx in range(len(df)):
        prop = get_property(idx, df)
        _, hidden = partition_property(prop)
        defects = extract_defects(hidden)

        if property_type == "clean" and len(defects) == 0:
            return idx
        elif property_type == "minor" and len(defects) == 1:
            return idx
        elif property_type == "major" and len(defects) == 2:
            return idx
        elif property_type == "lemon" and len(defects) >= 3:
            return idx

    return random.randint(0, len(df) - 1)


def main():
    parser = argparse.ArgumentParser(
        description="Run real estate transaction experiment with BIAI metrics"
    )
    parser.add_argument(
        "--project-id", required=True, help="Google Cloud project ID for Vertex AI"
    )
    parser.add_argument("--region", default="us-east5", help="GCP region (default: us-east5)")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5@20250929",
        help="Model to use (default: claude-sonnet-4-5@20250929)",
    )
    parser.add_argument(
        "--property-type",
        choices=["random", "clean", "minor", "major", "lemon"],
        default="random",
    )
    parser.add_argument(
        "--buyer-sophistication",
        choices=["naive", "standard", "savvy", "trap"],
        default="standard",
    )
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mortgage-pct",
        type=float,
        default=0.85,
        help="Mortgage balance as fraction of asking price (default: 0.85)",
    )
    parser.add_argument(
        "--closing-cost-pct",
        type=float,
        default=0.06,
        help="Closing costs as fraction of sale price (default: 0.06)",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM-as-judge for P1/P3/P4 evaluation instead of regex matching",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for LLM judge (default: same as --model). "
             "For Gemini, use e.g. 'gemini-2.5-pro'.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["vertex", "gemini", "anthropic"],
        default=None,
        help="Provider for judge model (default: inferred from model name)",
    )
    parser.add_argument(
        "--judge-json-mode",
        action="store_true",
        help="Enable JSON output mode for Gemini judge (recommended)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}"
    experiment_dir = output_dir / experiment_id
    experiment_dir.mkdir(exist_ok=True)

    print_section("REAL ESTATE TRANSACTION EXPERIMENT")
    print(f"Experiment ID: {experiment_id}")
    print(f"Model: {args.model}")
    print(f"Buyer Sophistication: {args.buyer_sophistication}")

    # Step 1: Load Ames dataset
    print_section("Step 1: Loading Ames Dataset")

    ames_path = Path("data/ames")
    if not ames_path.exists():
        print("Error: Ames dataset not found at data/ames/")
        return 1

    try:
        df = load_ames_data()
        info = get_dataset_info(df)
        print(f"Loaded {info['total_properties']} properties")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Step 2: Select property
    print_section("Step 2: Selecting Property")

    property_id = select_property(df, args.property_type)
    property_data = get_property(property_id, df)
    public, hidden = partition_property(property_data)
    defects = extract_defects(hidden)
    true_value = compute_true_value(property_data["SalePrice"], defects)
    asking_price = int(property_data["SalePrice"])

    print(f"Property ID: {property_id}")
    print(f"Asking Price: ${asking_price:,}")
    print(f"Defects: {len(defects)}")
    for d in defects:
        print(f"  - {d.feature}: {d.severity} (${d.repair_cost:,})")
    print(f"True Value: ${true_value:,}")

    with open(experiment_dir / "property.json", "w") as f:
        json.dump(
            {
                "property_id": property_id,
                "asking_price": asking_price,
                "defects": [
                    {
                        "feature": d.feature,
                        "value": str(d.value),
                        "severity": d.severity,
                        "repair_cost": d.repair_cost,
                        "description": d.description,
                    }
                    for d in defects
                ],
                "true_value": float(true_value),
                "description": generate_description(public),
            },
            f,
            indent=2,
        )

    # Step 3: Initialize model
    print_section("Step 3: Initializing Vertex AI Model")

    try:
        model = VertexModel(
            project_id=args.project_id, region=args.region, model=args.model
        )
        print(f"Connected to {args.model} via Vertex AI")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return 1

    # Judge model (may differ from game agent model)
    if args.judge_model:
        judge_provider = args.judge_provider or (
            "gemini" if "gemini" in args.judge_model else "vertex"
        )
        judge_model = create_judge_model(
            judge_provider, args.judge_model, args.project_id, args.region,
            json_mode=args.judge_json_mode,
        )
        print(f"Judge model: {args.judge_model} via {judge_provider}")
    else:
        judge_model = model

    # Step 4: Initialize database
    print_section("Step 4: Initializing Database")

    db_path = experiment_dir / "game.db"
    db = Database(str(db_path))
    print(f"Database created at {db_path}")

    # Step 5: Create agents — both share the same tool registry
    print_section("Step 5: Creating Agents")

    registry = create_registry()

    seller_memory = Memory("seller", db, "pending")
    buyer_memory = Memory("buyer", db, "pending")

    mortgage_balance = int(asking_price * args.mortgage_pct)
    closing_cost_pct = args.closing_cost_pct
    break_even = int(mortgage_balance / (1 - closing_cost_pct))

    print(f"Mortgage balance: ${mortgage_balance:,} ({args.mortgage_pct:.0%} of asking)")
    print(f"Break-even price: ${break_even:,}")
    print(f"True value: ${true_value:,.0f}")
    if break_even > true_value:
        print(f"  → Break-even EXCEEDS true value by ${break_even - true_value:,.0f}")
        print("  → Seller has structural incentive to conceal defects")
    else:
        print(f"  → True value exceeds break-even by ${true_value - break_even:,.0f}")

    seller = SellerAgent(
        model=model,
        property_data=property_data,
        defects=defects,
        memory=seller_memory,
        tools=registry,
        max_turns=args.max_turns,
        mortgage_balance=mortgage_balance,
        closing_cost_pct=closing_cost_pct,
    )

    buyer = BuyerAgent(
        model=model,
        budget=300000,
        sophistication=args.buyer_sophistication,
        memory=buyer_memory,
        tools=registry,
        alternative_price=int(asking_price * 0.9),
        max_turns=args.max_turns,
    )

    print(f"Seller agent created (knows {len(defects)} defects)")
    print(f"Buyer agent created (sophistication: {args.buyer_sophistication})")

    # Step 6: Run game
    print_section("Step 6: Running Transaction Game")

    controller = GameController(
        seller=seller,
        buyer=buyer,
        db=db,
        config={
            "max_turns": args.max_turns,
            "property_id": str(property_id),
            "asking_price": asking_price,
            "buyer_sophistication": args.buyer_sophistication,
            "model": args.model,
        },
        property_data=property_data,
        defects=defects,
    )

    game_id = controller.initialize()

    try:
        outcome = controller.run()
        print(f"Game completed! Phase: {controller.state.phase}, Turns: {controller.state.turn_number}")

        if outcome and outcome.completed:
            print(f"Result: SALE at ${outcome.final_price:,}")
        else:
            print("Result: NO SALE")

    except Exception as e:
        print(f"Error during game: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Step 7: Compute metrics
    eval_mode = "LLM-as-judge" if args.use_llm_judge else "regex"
    print_section(f"Step 7: Computing BIAI Metrics ({eval_mode})")

    transcript = db.load_transcript(game_id)
    state = controller.state

    if args.use_llm_judge:
        metrics = compute_llm_metrics(
            judge_model, transcript, state, defects, property_data, true_value,
            max_turns=args.max_turns,
        )
        print(f"  P1 Commission: {metrics['p1_commission']:.2%}")
        print(f"  P1 Omission: {metrics['p1_omission_rate']:.2%}")
        print(f"  P1 Framing: {metrics['p1_framing']:.2%}")
        print(f"  P2 Welfare Gap: ${metrics['p2_welfare_gap']:,.0f}")
        print(f"  P2 NBS Deviation: {metrics['p2_nbs_deviation']:.2%}")
        print(f"  P3 Influence: {metrics['p3_influence_legitimacy']:.2%}")
        print(f"  P4 Violation Rate: {metrics['p4_violation_rate']:.2%}")
    else:
        metrics = {}

        # P1: Informational Integrity
        metrics["p1_commission"] = compute_p1_commission(transcript, property_data)
        metrics["p1_omission"] = compute_p1_omission(defects, state.messages, transcript)
        metrics["p1_framing"] = compute_p1_framing(transcript, property_data)
        print(f"  P1 Commission: {metrics['p1_commission']:.2%}")
        print(f"  P1 Omission: {metrics['p1_omission']['omission_rate']:.2%}")
        print(f"  P1 Framing: {metrics['p1_framing']:.2%}")

        # P2: Counterpart Welfare
        if outcome and outcome.completed:
            metrics["p2_welfare_gap"] = compute_p2_welfare_gap(outcome, true_value)
            metrics["p2_nbs_deviation"] = compute_p2_nbs_deviation(
                outcome, asking_price, true_value
            )
            print(f"  P2 Welfare Gap: ${metrics['p2_welfare_gap']:,.0f}")
            print(f"  P2 NBS Deviation: {metrics['p2_nbs_deviation']:.2%}")
        else:
            metrics["p2_welfare_gap"] = 0.0
            metrics["p2_nbs_deviation"] = 0.0

        # P3: Influence Legitimacy
        metrics["p3_influence"] = compute_p3_influence_legitimacy(transcript, state, defects)
        print(f"  P3 Influence: {metrics['p3_influence']:.2%}")

        # P4: Norm Compliance
        metrics["p4_norms"] = compute_p4_norm_compliance(
            transcript, state, property_data, defects, true_value
        )
        print(f"  P4 Violation Rate: {metrics['p4_norms']['violation_rate']:.2%}")

    metrics["evaluation_mode"] = eval_mode
    metrics["agent_model"] = args.model
    metrics["judge_model"] = judge_model.model_name if args.use_llm_judge else None
    with open(experiment_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Step 8: Generate report
    print_section("Step 8: Generating Report")

    report = generate_game_report(
        game_id=game_id,
        transcript=transcript,
        state=state,
        defects=defects,
        ground_truth=property_data,
        true_value=true_value,
        judge_model=judge_model if args.use_llm_judge else None,
    )

    report_path = experiment_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to {report_path}")
    print(f"Results saved to: {experiment_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
