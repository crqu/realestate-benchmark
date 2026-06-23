#!/usr/bin/env python3
"""Run an end-to-end real estate transaction experiment.

This script runs a complete game between a seller and buyer agent, then
computes all BIAI metrics (P1-P4 + SAI) to evaluate agent behavioral integrity.

Usage:
    python experiments/run_experiment.py --project-id YOUR_PROJECT_ID

The script will:
1. Load a property from the Ames dataset
2. Initialize seller and buyer agents with Vertex AI (Claude)
3. Run the transaction game
4. Compute behavioral integrity metrics
5. Generate a detailed report
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
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
from realestate_benchmark.tools.registry import create_buyer_registry, create_seller_registry


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def select_property(df, property_type: str = "random"):
    """Select a property based on defect count.

    Args:
        df: Ames dataset DataFrame
        property_type: "clean", "minor", "major", "lemon", or "random"

    Returns:
        Property ID
    """
    if property_type == "random":
        return random.randint(0, len(df) - 1)

    # Categorize by defect count
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

    # Fallback to random
    return random.randint(0, len(df) - 1)


def main():
    parser = argparse.ArgumentParser(
        description="Run real estate transaction experiment with BIAI metrics"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Google Cloud project ID for Vertex AI"
    )
    parser.add_argument(
        "--region",
        default="us-east5",
        help="GCP region (default: us-east5)"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5@20250929",
        help="Model to use (default: claude-sonnet-4-5@20250929)"
    )
    parser.add_argument(
        "--property-type",
        choices=["random", "clean", "minor", "major", "lemon"],
        default="random",
        help="Type of property to select (default: random)"
    )
    parser.add_argument(
        "--buyer-sophistication",
        choices=["naive", "standard", "savvy"],
        default="standard",
        help="Buyer sophistication level (default: standard)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns (default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Random seed: {args.seed}")

    # Create output directory
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
    print(f"Max Turns: {args.max_turns}")
    print(f"Output: {experiment_dir}")

    # Step 1: Load Ames dataset
    print_section("Step 1: Loading Ames Dataset")

    ames_path = Path("data/ames")
    if not ames_path.exists():
        print("❌ Error: Ames dataset not found at data/ames/")
        print("Please download the dataset following data/ames/README.md")
        return 1

    try:
        df = load_ames_data()
        info = get_dataset_info(df)
        print(f"✅ Loaded {info['total_properties']} properties")
        print(f"   Features: {len(info['features'])}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1

    # Step 2: Select property
    print_section("Step 2: Selecting Property")

    property_id = select_property(df, args.property_type)
    property_data = get_property(property_id, df)
    public, hidden = partition_property(property_data)
    defects = extract_defects(hidden)
    true_value = compute_true_value(property_data["SalePrice"], defects)

    print(f"Property ID: {property_id}")
    print(f"Asking Price: ${property_data['SalePrice']:,}")
    print(f"Defects: {len(defects)}")
    for d in defects:
        print(f"  - {d.feature}: {d.severity} (${d.repair_cost:,})")
    print(f"True Value: ${true_value:,}")
    print(f"Information Gap: ${property_data['SalePrice'] - true_value:,}")

    # Save property data
    with open(experiment_dir / "property.json", "w") as f:
        json.dump({
            "property_id": property_id,
            "asking_price": float(property_data["SalePrice"]),
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
        }, f, indent=2)

    # Step 3: Initialize model
    print_section("Step 3: Initializing Vertex AI Model")

    try:
        model = VertexModel(
            project_id=args.project_id,
            region=args.region,
            model=args.model,
        )
        print(f"✅ Connected to {args.model} via Vertex AI")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        print("\nMake sure you have:")
        print("  1. Set up Google Cloud authentication")
        print("  2. Enabled Vertex AI API")
        print("  3. Have access to Claude models in your region")
        return 1

    # Step 4: Initialize database
    print_section("Step 4: Initializing Database")

    db_path = experiment_dir / "game.db"
    db = Database(str(db_path))
    print(f"✅ Database created at {db_path}")

    # Step 5: Create agents
    print_section("Step 5: Creating Agents")

    game_id = db.create_game({
        "property_id": property_id,
        "buyer_sophistication": args.buyer_sophistication,
        "max_turns": args.max_turns,
        "model": args.model,
    })

    seller_memory = Memory("seller", db, game_id)
    buyer_memory = Memory("buyer", db, game_id)

    seller = SellerAgent(
        model=model,
        property_data=property_data,
        defects=defects,
        memory=seller_memory,
        tools=create_seller_registry(),
    )

    buyer = BuyerAgent(
        model=model,
        budget=300000,  # $300k budget
        sophistication=args.buyer_sophistication,
        memory=buyer_memory,
        tools=create_buyer_registry(),
    )

    print(f"✅ Seller agent created (knows {len(defects)} defects)")
    print(f"✅ Buyer agent created (sophistication: {args.buyer_sophistication})")

    # Step 6: Run game
    print_section("Step 6: Running Transaction Game")

    controller = GameController(
        seller=seller,
        buyer=buyer,
        property_data=property_data,
        db=db,
        config={"max_turns": args.max_turns},
        defects=defects,
    )

    print("🎮 Game starting...")

    game_id = controller.initialize()
    print("   Watch as agents negotiate the transaction!\n")

    try:
        outcome = controller.run()
        print("\n✅ Game completed!")
        print(f"   Phase: {controller.state.phase}")
        print(f"   Turns: {controller.state.turn_number}")

        if outcome and outcome.completed:
            print("   Result: SALE COMPLETED")
            print(f"   Final Price: ${outcome.final_price:,}")
            print(f"   Buyer Overpayment: ${outcome.final_price - true_value:,}")
        else:
            print("   Result: NO SALE")

    except Exception as e:
        print(f"❌ Error during game: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 7: Compute metrics
    print_section("Step 7: Computing BIAI Metrics")

    transcript = db.load_transcript(game_id)
    state = controller.state

    metrics = {}

    # P1: Informational Integrity
    print("Computing P1: Informational Integrity...")
    metrics["p1_commission"] = compute_p1_commission(transcript, property_data)
    metrics["p1_omission"] = compute_p1_omission(defects, state.disclosures, transcript)
    metrics["p1_framing"] = compute_p1_framing(transcript, property_data)
    print(f"  ✅ Commission rate: {metrics['p1_commission']:.2%}")
    print(f"  ✅ Omission rate: {metrics['p1_omission']['omission_rate']:.2%}")
    print(f"  ✅ Framing score: {metrics['p1_framing']:.2%}")

    # P2: Counterpart Welfare
    print("\nComputing P2: Counterpart Welfare...")
    if outcome and outcome.completed:
        metrics["p2_welfare_gap"] = compute_p2_welfare_gap(outcome, true_value)
        metrics["p2_nbs_deviation"] = compute_p2_nbs_deviation(
            outcome, property_data["SalePrice"], true_value
        )
        print(f"  ✅ Welfare gap: ${metrics['p2_welfare_gap']:,.0f}")
        print(f"  ✅ NBS deviation: {metrics['p2_nbs_deviation']:.2%}")
    else:
        metrics["p2_welfare_gap"] = 0.0
        metrics["p2_nbs_deviation"] = 0.0
        print("  ⚠️ No sale - welfare metrics N/A")

    # P3: Influence Legitimacy
    print("\nComputing P3: Influence Legitimacy...")
    metrics["p3_influence"] = compute_p3_influence_legitimacy(transcript, state, defects)
    print(f"  ✅ Illegitimate influence rate: {metrics['p3_influence']:.2%}")

    # P4: Norm Compliance
    print("\nComputing P4: Norm Compliance...")
    metrics["p4_norms"] = compute_p4_norm_compliance(
        transcript, state, property_data, defects,
        true_value)
    print(f"  ✅ Violation rate: {metrics['p4_norms']['violation_rate']:.2%}")
    for norm, compliant in metrics['p4_norms']['norms'].items():
        status = "✅" if compliant else "❌"
        print(f"     {status} {norm}: {'compliant' if compliant else 'VIOLATED'}")

    # Save metrics
    with open(experiment_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Step 8: Generate report
    print_section("Step 8: Generating Report")

    report = generate_game_report(
        game_id=game_id,
        transcript=transcript,
        state=state,
        defects=defects,
        ground_truth=property_data,
        true_value=true_value,
    )

    report_path = experiment_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"✅ Report saved to {report_path}")

    # Step 9: Summary
    print_section("EXPERIMENT COMPLETE")

    print("📊 Summary:")
    print(f"   Property: {len(defects)} defects, ${property_data['SalePrice'] - true_value:,} hidden costs")
    print(f"   Outcome: {'SALE' if outcome and outcome.completed else 'NO SALE'}")
    if outcome and outcome.completed:
        print(f"   Price: ${outcome.final_price:,} (overpayment: ${outcome.final_price - true_value:,})")
    print(f"   Turns: {controller.state.turn_number}")
    print("\n📈 Behavioral Integrity:")
    print(f"   P1 (Info): {(1 - metrics['p1_omission']['omission_rate']) * 100:.0f}% disclosed")
    print(f"   P2 (Welfare): ${metrics['p2_welfare_gap']:,.0f} buyer loss")
    print(f"   P3 (Influence): {metrics['p3_influence'] * 100:.0f}% illegitimate")
    print(f"   P4 (Norms): {(1 - metrics['p4_norms']['violation_rate']) * 100:.0f}% compliant")

    print(f"\n📁 Results saved to: {experiment_dir}")
    print("   - property.json   (property data)")
    print("   - metrics.json    (all metrics)")
    print("   - report.md       (full report)")
    print("   - game.db         (full game state)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
