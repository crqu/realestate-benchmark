"""Per-harm-type experiment functions.

Each function runs one category of harm taxonomy experiment and returns
a list of per-game result dicts with harm-specific metrics attached.
"""

from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

from realestate_benchmark.agents.buyer import BuyerAgent
from realestate_benchmark.agents.memory import Memory
from realestate_benchmark.agents.personas import (
    ADVICE_CONFIGS,
    CONFIDENTIAL_CONFIGS,
    EXTRACTION_STRATEGIES,
    PERSONA_PAIRS,
    VULNERABILITY_CONFIGS,
    AdviceSeekerConfig,
    ConfidentialInfoConfig,
    PersonaConfig,
    VulnerabilityConfig,
)
from realestate_benchmark.agents.seller import SellerAgent
from realestate_benchmark.data.ames import get_property
from realestate_benchmark.data.properties import (
    compute_true_value,
    extract_defects,
    partition_property,
)
from realestate_benchmark.environment.controller import GameController
from realestate_benchmark.environment.database import Database
from realestate_benchmark.evaluation.harm_metrics import (
    compute_proactive_disclosure_rate,
    llm_judge_coercion,
    llm_judge_confidentiality,
    llm_judge_steering,
    llm_judge_unauthorized_advice,
)
from realestate_benchmark.evaluation.report import generate_game_report
from realestate_benchmark.evaluation.sai import compute_game_metrics
from realestate_benchmark.models.interface import ModelInterface
from realestate_benchmark.tools.registry import create_registry


def _select_properties(df: Any, property_type: str, count: int, seed: int) -> list[int]:
    """Select property indices matching a given defect profile."""
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


def _run_game(
    model: ModelInterface,
    df: Any,
    property_idx: int,
    buyer_sophistication: str,
    max_turns: int,
    game_dir: Path,
    use_llm_judge: bool = True,
    *,
    persona_config: PersonaConfig | None = None,
    vulnerability_config: VulnerabilityConfig | None = None,
    advice_config: AdviceSeekerConfig | None = None,
    confidential_config: ConfidentialInfoConfig | None = None,
    mortgage_pct: float = 0.85,
) -> dict[str, Any] | None:
    """Run a single game with harm-experiment configuration."""
    property_data = get_property(property_idx, df)
    public, hidden = partition_property(property_data)
    defects = extract_defects(hidden)
    true_value = compute_true_value(property_data["SalePrice"], defects)
    asking_price = int(property_data["SalePrice"])

    game_dir.mkdir(parents=True, exist_ok=True)
    db = Database(str(game_dir / "game.db"))
    registry = create_registry()

    seller_memory = Memory("seller", db, "pending")
    buyer_memory = Memory("buyer", db, "pending")

    mortgage_balance = int(asking_price * mortgage_pct)

    seller = SellerAgent(
        model=model,
        property_data=property_data,
        defects=defects,
        memory=seller_memory,
        tools=registry,
        max_turns=max_turns,
        mortgage_balance=mortgage_balance,
        confidential_config=confidential_config,
    )

    buyer = BuyerAgent(
        model=model,
        budget=300000,
        sophistication=buyer_sophistication,
        memory=buyer_memory,
        tools=registry,
        alternative_price=int(asking_price * 0.9),
        max_turns=max_turns,
        persona_config=persona_config,
        vulnerability_config=vulnerability_config,
        advice_config=advice_config,
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
        print(f"    ERROR: {e}")
        traceback.print_exc()
        return None

    transcript = db.load_transcript(game_id)
    state = controller.state

    judge = model if use_llm_judge else None
    metrics = compute_game_metrics(
        transcript, state, defects, property_data, true_value,
        judge_model=judge, max_turns=max_turns,
    )

    result: dict[str, Any] = {
        "game_id": game_id,
        "property_id": property_idx,
        "asking_price": asking_price,
        "true_value": float(true_value),
        "defect_count": len(defects),
        "buyer_sophistication": buyer_sophistication,
        "completed": outcome.completed if outcome else False,
        "final_price": outcome.final_price if outcome and outcome.completed else None,
        "total_turns": state.turn_number,
        "metrics": metrics,
    }

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

    with open(game_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ---------------------------------------------------------------------------
# Experiment 1/2: Discrimination (Fair Housing / Steering + Bargaining)
# ---------------------------------------------------------------------------


def run_discrimination_experiment(
    model: ModelInterface,
    df: Any,
    *,
    protected_attributes: list[str] | None = None,
    property_types: list[str] | None = None,
    num_seeds: int = 3,
    base_seed: int = 42,
    max_turns: int = 20,
    output_dir: Path = Path("experiments/harm_results/discrimination"),
    use_llm_judge: bool = True,
) -> list[dict[str, Any]]:
    """Run matched-pair discrimination experiments.

    For each (property, seed), runs both treatment and control personas
    to enable paired statistical comparison.
    """
    if protected_attributes is None:
        protected_attributes = list(PERSONA_PAIRS.keys())
    if property_types is None:
        property_types = ["major", "lemon"]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    for attr in protected_attributes:
        treatment_persona, control_persona = PERSONA_PAIRS[attr]

        for prop_type in property_types:
            prop_indices = _select_properties(df, prop_type, num_seeds, base_seed)

            for seed_idx, prop_idx in enumerate(prop_indices):
                pair_id = f"{attr}_{prop_type}_seed{seed_idx}"

                for persona in [treatment_persona, control_persona]:
                    label = f"{pair_id}_{persona.demographic_group}"
                    print(f"  [{label}] prop={prop_idx}", end=" ... ", flush=True)
                    t0 = time.time()

                    game_dir = output_dir / label
                    result = _run_game(
                        model, df, prop_idx,
                        buyer_sophistication=persona.base_sophistication,
                        max_turns=max_turns,
                        game_dir=game_dir,
                        use_llm_judge=use_llm_judge,
                        persona_config=persona,
                    )

                    if result is not None:
                        steering = llm_judge_steering(
                            model,
                            Database(str(game_dir / "game.db")).load_transcript(result["game_id"]),
                            persona,
                        )
                        result.update(steering)
                        result["pair_id"] = pair_id
                        result["protected_attribute"] = attr
                        result["demographic_group"] = persona.demographic_group
                        result["persona_id"] = persona.persona_id

                        with open(game_dir / "result.json", "w") as f:
                            json.dump(result, f, indent=2, default=str)

                        all_results.append(result)
                        elapsed = time.time() - t0
                        status = (
                            f"SALE ${result['final_price']:,}" if result["completed"]
                            else "NO SALE"
                        )
                        print(f"{status} ({elapsed:.0f}s)")
                    else:
                        print(f"FAILED ({time.time() - t0:.0f}s)")

    return all_results


# ---------------------------------------------------------------------------
# Experiment 3: Misrepresentation
# ---------------------------------------------------------------------------


def run_misrepresentation_experiment(
    model: ModelInterface,
    df: Any,
    *,
    pressure_levels: dict[str, float] | None = None,
    buyer_levels: list[str] | None = None,
    property_types: list[str] | None = None,
    num_seeds: int = 3,
    base_seed: int = 42,
    max_turns: int = 20,
    output_dir: Path = Path("experiments/harm_results/misrepresentation"),
    use_llm_judge: bool = True,
) -> list[dict[str, Any]]:
    """Run misrepresentation experiments across pressure levels and buyer types."""
    if pressure_levels is None:
        pressure_levels = {"low": 0.50, "medium": 0.85, "high": 0.95}
    if buyer_levels is None:
        buyer_levels = ["naive", "standard", "savvy", "inquisitor"]
    if property_types is None:
        property_types = ["minor", "major", "lemon"]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    for prop_type in property_types:
        prop_indices = _select_properties(df, prop_type, num_seeds, base_seed)

        for pressure_name, mortgage_pct in pressure_levels.items():
            for buyer_level in buyer_levels:
                for seed_idx, prop_idx in enumerate(prop_indices):
                    label = f"{prop_type}_{pressure_name}_{buyer_level}_seed{seed_idx}"
                    print(f"  [{label}]", end=" ... ", flush=True)
                    t0 = time.time()

                    game_dir = output_dir / label
                    result = _run_game(
                        model, df, prop_idx,
                        buyer_sophistication=buyer_level,
                        max_turns=max_turns,
                        game_dir=game_dir,
                        use_llm_judge=use_llm_judge,
                        mortgage_pct=mortgage_pct,
                    )

                    if result is not None:
                        property_data = get_property(prop_idx, df)
                        _, hidden = partition_property(property_data)
                        defects = extract_defects(hidden)
                        db = Database(str(game_dir / "game.db"))
                        transcript = db.load_transcript(result["game_id"])

                        state_messages = []
                        for entry in transcript:
                            if entry.tool_name == "send_message":
                                from realestate_benchmark.environment.state import Message
                                state_messages.append(Message(
                                    sender=entry.agent_id,
                                    content=entry.parameters.get("content", ""),
                                    turn=entry.turn,
                                ))

                        proactive_rate = compute_proactive_disclosure_rate(
                            defects, state_messages,
                        )
                        result["proactive_disclosure_rate"] = proactive_rate
                        result["seller_pressure"] = pressure_name
                        result["mortgage_pct"] = mortgage_pct

                        with open(game_dir / "result.json", "w") as f:
                            json.dump(result, f, indent=2, default=str)

                        all_results.append(result)
                        elapsed = time.time() - t0
                        status = (
                            f"SALE ${result['final_price']:,}" if result["completed"]
                            else "NO SALE"
                        )
                        print(f"{status} proactive={proactive_rate:.0%} ({elapsed:.0f}s)")
                    else:
                        print(f"FAILED ({time.time() - t0:.0f}s)")

    return all_results


# ---------------------------------------------------------------------------
# Experiment 4: Confidentiality Leakage
# ---------------------------------------------------------------------------


def run_confidentiality_experiment(
    model: ModelInterface,
    df: Any,
    *,
    info_types: list[str] | None = None,
    extraction_strategies: list[str] | None = None,
    buyer_levels: list[str] | None = None,
    property_types: list[str] | None = None,
    num_seeds: int = 3,
    base_seed: int = 42,
    max_turns: int = 20,
    output_dir: Path = Path("experiments/harm_results/confidentiality"),
    use_llm_judge: bool = True,
) -> list[dict[str, Any]]:
    """Run confidentiality leakage experiments."""
    if info_types is None:
        info_types = list(CONFIDENTIAL_CONFIGS.keys())
    if extraction_strategies is None:
        extraction_strategies = list(EXTRACTION_STRATEGIES.keys())
    if buyer_levels is None:
        buyer_levels = ["naive", "savvy"]
    if property_types is None:
        property_types = ["major", "lemon"]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    for info_type in info_types:
        conf_config = CONFIDENTIAL_CONFIGS[info_type]

        for strategy_name in extraction_strategies:
            strategy_text = EXTRACTION_STRATEGIES[strategy_name]

            for buyer_level in buyer_levels:
                for prop_type in property_types:
                    prop_indices = _select_properties(df, prop_type, num_seeds, base_seed)

                    for seed_idx, prop_idx in enumerate(prop_indices):
                        label = (
                            f"{info_type}_{strategy_name}_{buyer_level}"
                            f"_{prop_type}_seed{seed_idx}"
                        )
                        print(f"  [{label}]", end=" ... ", flush=True)
                        t0 = time.time()

                        advice_for_extraction = None
                        if strategy_text:
                            advice_for_extraction = AdviceSeekerConfig(
                                solicitation_type="none",
                                injection_text=strategy_text,
                            )

                        game_dir = output_dir / label
                        result = _run_game(
                            model, df, prop_idx,
                            buyer_sophistication=buyer_level,
                            max_turns=max_turns,
                            game_dir=game_dir,
                            use_llm_judge=use_llm_judge,
                            confidential_config=conf_config,
                            advice_config=advice_for_extraction,
                        )

                        if result is not None:
                            db = Database(str(game_dir / "game.db"))
                            transcript = db.load_transcript(result["game_id"])

                            confidentiality = llm_judge_confidentiality(
                                model, transcript, conf_config,
                            )
                            result.update(confidentiality)
                            result["info_type"] = info_type
                            result["extraction_strategy"] = strategy_name

                            with open(game_dir / "result.json", "w") as f:
                                json.dump(result, f, indent=2, default=str)

                            all_results.append(result)
                            elapsed = time.time() - t0
                            breach = "BREACH" if result.get("breach_detected") else "SAFE"
                            print(f"{breach} sev={result.get('breach_severity', 0)} ({elapsed:.0f}s)")
                        else:
                            print(f"FAILED ({time.time() - t0:.0f}s)")

    return all_results


# ---------------------------------------------------------------------------
# Experiment 5: Unauthorized Legal / Financial Advice
# ---------------------------------------------------------------------------


def run_advice_experiment(
    model: ModelInterface,
    df: Any,
    *,
    solicitation_types: list[str] | None = None,
    buyer_levels: list[str] | None = None,
    property_types: list[str] | None = None,
    num_seeds: int = 3,
    base_seed: int = 42,
    max_turns: int = 20,
    output_dir: Path = Path("experiments/harm_results/advice"),
    use_llm_judge: bool = True,
) -> list[dict[str, Any]]:
    """Run unauthorized advice experiments."""
    if solicitation_types is None:
        solicitation_types = list(ADVICE_CONFIGS.keys())
    if buyer_levels is None:
        buyer_levels = ["naive", "savvy"]
    if property_types is None:
        property_types = ["major", "lemon"]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    for sol_type in solicitation_types:
        adv_config = ADVICE_CONFIGS[sol_type]

        for buyer_level in buyer_levels:
            for prop_type in property_types:
                prop_indices = _select_properties(df, prop_type, num_seeds, base_seed)

                for seed_idx, prop_idx in enumerate(prop_indices):
                    label = f"{sol_type}_{buyer_level}_{prop_type}_seed{seed_idx}"
                    print(f"  [{label}]", end=" ... ", flush=True)
                    t0 = time.time()

                    game_dir = output_dir / label
                    result = _run_game(
                        model, df, prop_idx,
                        buyer_sophistication=buyer_level,
                        max_turns=max_turns,
                        game_dir=game_dir,
                        use_llm_judge=use_llm_judge,
                        advice_config=adv_config,
                    )

                    if result is not None:
                        db = Database(str(game_dir / "game.db"))
                        transcript = db.load_transcript(result["game_id"])

                        from realestate_benchmark.environment.state import Message
                        messages = []
                        for entry in transcript:
                            if entry.tool_name == "send_message":
                                messages.append(Message(
                                    sender=entry.agent_id,
                                    content=entry.parameters.get("content", ""),
                                    turn=entry.turn,
                                ))

                        advice = llm_judge_unauthorized_advice(
                            model, transcript, messages, sol_type,
                        )
                        result.update(advice)
                        result["solicitation_type"] = sol_type

                        with open(game_dir / "result.json", "w") as f:
                            json.dump(result, f, indent=2, default=str)

                        all_results.append(result)
                        elapsed = time.time() - t0
                        strength = result.get("max_advice_strength", 0)
                        print(f"strength={strength} ({elapsed:.0f}s)")
                    else:
                        print(f"FAILED ({time.time() - t0:.0f}s)")

    return all_results


# ---------------------------------------------------------------------------
# Experiment 6: Coercive Persuasion
# ---------------------------------------------------------------------------


def run_coercion_experiment(
    model: ModelInterface,
    df: Any,
    *,
    pressure_levels: dict[str, dict[str, Any]] | None = None,
    vulnerability_types: list[str] | None = None,
    property_types: list[str] | None = None,
    num_seeds: int = 3,
    base_seed: int = 42,
    max_turns: int = 20,
    output_dir: Path = Path("experiments/harm_results/coercion"),
    use_llm_judge: bool = True,
) -> list[dict[str, Any]]:
    """Run coercive persuasion experiments."""
    if pressure_levels is None:
        pressure_levels = {
            "normal": {"mortgage_pct": 0.85, "days_on_market": 30, "carrying_cost": 1500},
            "extreme": {"mortgage_pct": 0.95, "days_on_market": 90, "carrying_cost": 2500},
        }
    if vulnerability_types is None:
        vulnerability_types = list(VULNERABILITY_CONFIGS.keys())
    if property_types is None:
        property_types = ["major", "lemon"]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    for pressure_name, pressure_params in pressure_levels.items():
        for vuln_type in vulnerability_types:
            vuln_config = VULNERABILITY_CONFIGS[vuln_type]

            for prop_type in property_types:
                prop_indices = _select_properties(df, prop_type, num_seeds, base_seed)

                for seed_idx, prop_idx in enumerate(prop_indices):
                    label = (
                        f"{pressure_name}_{vuln_type}_{prop_type}_seed{seed_idx}"
                    )
                    print(f"  [{label}]", end=" ... ", flush=True)
                    t0 = time.time()

                    game_dir = output_dir / label
                    result = _run_game(
                        model, df, prop_idx,
                        buyer_sophistication="standard",
                        max_turns=max_turns,
                        game_dir=game_dir,
                        use_llm_judge=use_llm_judge,
                        vulnerability_config=vuln_config,
                        mortgage_pct=pressure_params.get("mortgage_pct", 0.85),
                    )

                    if result is not None:
                        db = Database(str(game_dir / "game.db"))
                        transcript = db.load_transcript(result["game_id"])

                        from realestate_benchmark.environment.state import Message
                        messages = []
                        for entry in transcript:
                            if entry.tool_name == "send_message":
                                messages.append(Message(
                                    sender=entry.agent_id,
                                    content=entry.parameters.get("content", ""),
                                    turn=entry.turn,
                                ))

                        coercion = llm_judge_coercion(
                            model, transcript, messages,
                            vuln_type, vuln_config.description,
                        )
                        result.update(coercion)
                        result["seller_pressure"] = pressure_name
                        result["buyer_vulnerability"] = vuln_type

                        with open(game_dir / "result.json", "w") as f:
                            json.dump(result, f, indent=2, default=str)

                        all_results.append(result)
                        elapsed = time.time() - t0
                        intensity = result.get("coercion_intensity_score", 0.0)
                        targeted = "TARGETED" if result.get("vulnerability_targeted") else ""
                        print(f"intensity={intensity:.2f} {targeted} ({elapsed:.0f}s)")
                    else:
                        print(f"FAILED ({time.time() - t0:.0f}s)")

    return all_results
