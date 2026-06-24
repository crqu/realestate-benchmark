#!/usr/bin/env python3
"""Batch experiment runner for cross-condition analysis.

Runs multiple games across property types and buyer sophistication levels,
collecting all metrics including reasoning analysis and negotiation dynamics.

Usage:
    venv/bin/python experiments/run_batch.py \
        --project-id YOUR_PROJECT_ID \
        --num-seeds 3 \
        --property-types major lemon \
        --buyer-levels naive savvy
"""

import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realestate_benchmark.agents.buyer import BuyerAgent
from realestate_benchmark.agents.memory import Memory
from realestate_benchmark.agents.seller import SellerAgent
from realestate_benchmark.data.ames import get_property, load_ames_data
from realestate_benchmark.data.descriptions import generate_description
from realestate_benchmark.data.properties import (
    compute_true_value,
    extract_defects,
    partition_property,
)
from realestate_benchmark.environment.controller import GameController
from realestate_benchmark.environment.database import Database
from realestate_benchmark.evaluation.sai import compute_game_metrics
from realestate_benchmark.evaluation.report import generate_game_report
from realestate_benchmark.models.vertex import VertexModel


def select_properties_by_type(df, property_type: str, count: int, seed: int) -> list[int]:
    """Select multiple property indices matching the given type."""
    rng = random.Random(seed)
    candidates = []

    for idx in range(len(df)):
        prop = get_property(idx, df)
        _, hidden = partition_property(prop)
        defects = extract_defects(hidden)
        n = len(defects)

        if property_type == "clean" and n == 0:
            candidates.append(idx)
        elif property_type == "minor" and n == 1:
            candidates.append(idx)
        elif property_type == "major" and n == 2:
            candidates.append(idx)
        elif property_type == "lemon" and n >= 3:
            candidates.append(idx)

    rng.shuffle(candidates)
    return candidates[:count]


def run_single_game(
    model,
    df,
    property_idx: int,
    buyer_sophistication: str,
    max_turns: int,
    experiment_dir: Path,
    game_label: str,
    use_llm_judge: bool = True,
) -> dict | None:
    """Run a single game and return results dict."""
    property_data = get_property(property_idx, df)
    public, hidden = partition_property(property_data)
    defects = extract_defects(hidden)
    true_value = compute_true_value(property_data["SalePrice"], defects)
    asking_price = int(property_data["SalePrice"])

    game_dir = experiment_dir / game_label
    game_dir.mkdir(parents=True, exist_ok=True)

    db_path = game_dir / "game.db"
    db = Database(str(db_path))

    from realestate_benchmark.tools.registry import create_registry

    registry = create_registry()
    seller_memory = Memory("seller", db, "pending")
    buyer_memory = Memory("buyer", db, "pending")

    seller = SellerAgent(
        model=model,
        property_data=property_data,
        defects=defects,
        memory=seller_memory,
        tools=registry,
        max_turns=max_turns,
    )

    buyer = BuyerAgent(
        model=model,
        budget=300000,
        sophistication=buyer_sophistication,
        memory=buyer_memory,
        tools=registry,
        alternative_price=int(asking_price * 0.9),
        max_turns=max_turns,
    )

    controller = GameController(
        seller=seller,
        buyer=buyer,
        db=db,
        config={
            "max_turns": max_turns,
            "property_id": str(property_idx),
            "asking_price": asking_price,
            "buyer_sophistication": buyer_sophistication,
            "model": model.model_name,
        },
        property_data=property_data,
        defects=defects,
    )

    game_id = controller.initialize()

    try:
        outcome = controller.run()
    except Exception as e:
        print(f"    ERROR in game: {e}")
        traceback.print_exc()
        return None

    transcript = db.load_transcript(game_id)
    state = controller.state

    judge = model if use_llm_judge else None
    metrics = compute_game_metrics(
        transcript, state, defects, property_data, true_value,
        judge_model=judge, max_turns=max_turns,
    )

    result = {
        "game_id": game_id,
        "game_label": game_label,
        "property_id": property_idx,
        "property_type": _classify_property(defects),
        "asking_price": asking_price,
        "true_value": float(true_value),
        "defect_count": len(defects),
        "buyer_sophistication": buyer_sophistication,
        "completed": outcome.completed if outcome else False,
        "final_price": outcome.final_price if outcome and outcome.completed else None,
        "total_turns": state.turn_number,
        "metrics": metrics,
    }

    with open(game_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    report = generate_game_report(
        game_id=game_id,
        transcript=transcript,
        state=state,
        defects=defects,
        ground_truth=property_data,
        true_value=true_value,
        judge_model=judge,
    )
    with open(game_dir / "report.md", "w") as f:
        f.write(report)

    return result


def _classify_property(defects) -> str:
    n = len(defects)
    if n == 0:
        return "clean"
    elif n == 1:
        return "minor"
    elif n == 2:
        return "major"
    else:
        return "lemon"


def print_summary(all_results: list[dict], output_path: Path):
    """Print and save experiment summary."""
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"Total games: {len(all_results)}")
    completed = [r for r in all_results if r["completed"]]
    print(f"Completed sales: {len(completed)} ({len(completed)/len(all_results)*100:.0f}%)")

    # Group by condition
    by_condition: dict[str, list[dict]] = {}
    for r in all_results:
        key = f"{r['property_type']}_{r['buyer_sophistication']}"
        by_condition.setdefault(key, []).append(r)

    print(f"\n{'Condition':<25} {'N':>3} {'Sale%':>6} {'P1_omit':>8} {'P1_comm':>8} "
          f"{'P3_infl':>8} {'P4_viol':>8} {'WelGap':>10} {'Decept':>8}")
    print("-" * 100)

    for condition in sorted(by_condition.keys()):
        results = by_condition[condition]
        n = len(results)
        sale_pct = sum(1 for r in results if r["completed"]) / n * 100

        m = [r["metrics"] for r in results]
        avg_omission = sum(x.get("p1_omission_rate", 0) for x in m) / n
        avg_commission = sum(x.get("p1_commission", 0) for x in m) / n
        avg_p3 = sum(x.get("p3_influence_legitimacy", 0) for x in m) / n
        avg_p4 = sum(x.get("p4_violation_rate", 0) for x in m) / n
        avg_welfare = sum(x.get("p2_welfare_gap", 0) for x in m) / n
        avg_deception = sum(x.get("deception_rate", 0) for x in m) / n

        print(f"{condition:<25} {n:>3} {sale_pct:>5.0f}% {avg_omission:>7.2f} "
              f"{avg_commission:>7.2f} {avg_p3:>7.2f} {avg_p4:>7.2f} "
              f"${avg_welfare:>8,.0f} {avg_deception:>7.2f}")

    # SAI analysis if we have both naive and savvy
    naive_results = [r for r in all_results if r["buyer_sophistication"] == "naive"]
    savvy_results = [r for r in all_results if r["buyer_sophistication"] == "savvy"]

    if naive_results and savvy_results:
        print(f"\n{'=' * 70}")
        print("  SAI ANALYSIS (naive - savvy)")
        print(f"{'=' * 70}\n")

        sai_metrics = [
            ("p1_omission_rate", "P1 Omission"),
            ("p1_commission", "P1 Commission"),
            ("p1_framing", "P1 Framing"),
            ("p3_influence_legitimacy", "P3 Influence"),
            ("p4_violation_rate", "P4 Violations"),
            ("p2_welfare_gap", "P2 Welfare Gap"),
            ("deception_rate", "Deception Rate"),
            ("alignment_rate", "Alignment Rate"),
        ]

        for key, label in sai_metrics:
            naive_vals = [r["metrics"].get(key, 0) for r in naive_results]
            savvy_vals = [r["metrics"].get(key, 0) for r in savvy_results]
            naive_avg = sum(naive_vals) / len(naive_vals) if naive_vals else 0
            savvy_avg = sum(savvy_vals) / len(savvy_vals) if savvy_vals else 0
            sai = naive_avg - savvy_avg

            if key == "p2_welfare_gap":
                print(f"  {label:<20} naive={naive_avg:>10,.0f}  savvy={savvy_avg:>10,.0f}  "
                      f"SAI=${sai:>+10,.0f}")
            else:
                print(f"  {label:<20} naive={naive_avg:>8.3f}  savvy={savvy_avg:>8.3f}  "
                      f"SAI={sai:>+8.3f}")

    # Negotiation dynamics summary
    print(f"\n{'=' * 70}")
    print("  NEGOTIATION DYNAMICS")
    print(f"{'=' * 70}\n")

    for condition in sorted(by_condition.keys()):
        results = by_condition[condition]
        dynamics_list = [r["metrics"].get("negotiation_dynamics", {}) for r in results]
        if not any(dynamics_list):
            continue

        zopa_vals = [d.get("zopa_position") for d in dynamics_list if d.get("zopa_position") is not None]
        rounds_vals = [d.get("efficiency_offer_rounds", 0) for d in dynamics_list]
        recip_vals = [d.get("concession_reciprocity") for d in dynamics_list if d.get("concession_reciprocity") is not None]

        line = f"  {condition:<25}"
        if zopa_vals:
            line += f" ZOPA={sum(zopa_vals)/len(zopa_vals):.2f}"
        if rounds_vals:
            line += f"  Rounds={sum(rounds_vals)/len(rounds_vals):.1f}"
        if recip_vals:
            line += f"  Reciprocity={sum(recip_vals)/len(recip_vals):.2f}"
        print(line)

    # Save full summary
    summary = {
        "total_games": len(all_results),
        "completed_sales": len(completed),
        "results": all_results,
        "by_condition": {
            k: {
                "count": len(v),
                "sale_rate": sum(1 for r in v if r["completed"]) / len(v),
                "avg_metrics": {
                    "p1_omission_rate": sum(r["metrics"].get("p1_omission_rate", 0) for r in v) / len(v),
                    "p1_commission": sum(r["metrics"].get("p1_commission", 0) for r in v) / len(v),
                    "p3_influence_legitimacy": sum(r["metrics"].get("p3_influence_legitimacy", 0) for r in v) / len(v),
                    "p4_violation_rate": sum(r["metrics"].get("p4_violation_rate", 0) for r in v) / len(v),
                    "p2_welfare_gap": sum(r["metrics"].get("p2_welfare_gap", 0) for r in v) / len(v),
                    "deception_rate": sum(r["metrics"].get("deception_rate", 0) for r in v) / len(v),
                },
            }
            for k, v in by_condition.items()
        },
    }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nFull summary saved to {output_path / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-east5")
    parser.add_argument("--model", default="claude-sonnet-4-5@20250929")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--property-types", nargs="+",
        default=["major", "lemon"],
        choices=["clean", "minor", "major", "lemon"],
    )
    parser.add_argument(
        "--buyer-levels", nargs="+",
        default=["naive", "savvy"],
        choices=["naive", "standard", "savvy"],
    )
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--use-llm-judge", action="store_true", default=True)
    parser.add_argument("--no-llm-judge", dest="use_llm_judge", action="store_false")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.output_dir) / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch experiment: {batch_dir}")
    print(f"Model: {args.model}")
    print(f"Property types: {args.property_types}")
    print(f"Buyer levels: {args.buyer_levels}")
    print(f"Seeds: {args.num_seeds} (base: {args.base_seed})")
    print(f"Max turns: {args.max_turns}")
    print(f"LLM Judge: {args.use_llm_judge}")

    model = VertexModel(project_id=args.project_id, region=args.region, model=args.model)
    df = load_ames_data()
    print(f"Loaded {len(df)} properties from Ames dataset")

    total_games = len(args.property_types) * len(args.buyer_levels) * args.num_seeds
    print(f"Total games to run: {total_games}\n")

    with open(batch_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    all_results = []
    game_num = 0

    for prop_type in args.property_types:
        property_indices = select_properties_by_type(df, prop_type, args.num_seeds, args.base_seed)
        if len(property_indices) < args.num_seeds:
            print(f"WARNING: Only found {len(property_indices)} '{prop_type}' properties "
                  f"(requested {args.num_seeds})")

        for buyer_level in args.buyer_levels:
            for seed_idx, prop_idx in enumerate(property_indices):
                game_num += 1
                label = f"{prop_type}_{buyer_level}_seed{seed_idx}"
                print(f"[{game_num}/{total_games}] {label} (property={prop_idx})", end=" ... ", flush=True)

                t0 = time.time()
                result = run_single_game(
                    model=model,
                    df=df,
                    property_idx=prop_idx,
                    buyer_sophistication=buyer_level,
                    max_turns=args.max_turns,
                    experiment_dir=batch_dir,
                    game_label=label,
                    use_llm_judge=args.use_llm_judge,
                )
                elapsed = time.time() - t0

                if result:
                    all_results.append(result)
                    status = f"SALE ${result['final_price']:,}" if result["completed"] else "NO SALE"
                    print(f"{status} ({elapsed:.0f}s)")
                else:
                    print(f"FAILED ({elapsed:.0f}s)")

    if all_results:
        print_summary(all_results, batch_dir)
    else:
        print("\nNo games completed successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
