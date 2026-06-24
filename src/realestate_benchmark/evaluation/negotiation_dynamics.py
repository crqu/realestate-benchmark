"""Negotiation dynamics analysis.

Purely computational metrics that analyze offer sequences to characterize
negotiation behavior: concession patterns, anchoring effects, price convergence,
ZOPA utilization, and negotiation efficiency.
"""

from typing import Any

from realestate_benchmark.environment.state import GameState, TranscriptEntry


def extract_offer_trajectory(state: GameState) -> list[dict[str, Any]]:
    """Extract chronological offer trajectory from game state.

    Args:
        state: Game state containing historical and current offers.

    Returns:
        List of offer dicts sorted by turn, each with from_party, amount, turn, status.
    """
    offers: list[dict[str, Any]] = []
    for o in state.offers:
        offers.append(
            {
                "from_party": o.from_party,
                "amount": o.amount,
                "turn": o.turn,
                "status": o.status,
            }
        )
    if state.current_offer is not None:
        o = state.current_offer
        offers.append(
            {
                "from_party": o.from_party,
                "amount": o.amount,
                "turn": o.turn,
                "status": o.status,
            }
        )
    offers.sort(key=lambda x: x["turn"])
    return offers


def compute_concession_analysis(state: GameState) -> dict[str, Any]:
    """Analyze concession patterns for each party.

    A seller concession is a decrease in consecutive seller offer amounts.
    A buyer concession is an increase in consecutive buyer offer amounts.

    Args:
        state: Game state with offer history.

    Returns:
        Dict with per-party concession details and reciprocity index.
    """
    trajectory = extract_offer_trajectory(state)

    result: dict[str, Any] = {}
    for party in ("seller", "buyer"):
        party_offers = [o for o in trajectory if o["from_party"] == party]
        concessions: list[dict[str, Any]] = []
        for i in range(1, len(party_offers)):
            prev = party_offers[i - 1]["amount"]
            curr = party_offers[i]["amount"]
            if party == "seller":
                size = prev - curr
            else:
                size = curr - prev
            if size > 0:
                concessions.append(
                    {
                        "from_amount": prev,
                        "to_amount": curr,
                        "size": size,
                        "turn": party_offers[i]["turn"],
                    }
                )
        total = sum(c["size"] for c in concessions)
        result[party] = {
            "concessions": concessions,
            "total_concession": total,
            "avg_concession_size": total / len(concessions) if concessions else 0.0,
            "num_concessions": len(concessions),
        }

    seller_avg = result["seller"]["avg_concession_size"]
    buyer_avg = result["buyer"]["avg_concession_size"]
    if seller_avg == 0.0 and buyer_avg == 0.0:
        reciprocity = 0.0
    else:
        reciprocity = 1.0 - abs(seller_avg - buyer_avg) / max(seller_avg, buyer_avg, 1)

    result["reciprocity_index"] = reciprocity
    return result


def compute_anchoring_effect(state: GameState) -> dict[str, Any]:
    """Analyze anchoring effect of first offer and asking price on final price.

    Args:
        state: Game state with offer history and outcome.

    Returns:
        Dict with first offer details, asking price, final price, and anchor ratios.
    """
    trajectory = extract_offer_trajectory(state)

    first_offer_party: str | None = None
    first_offer_amount: int | None = None
    if trajectory:
        first_offer_party = trajectory[0]["from_party"]
        first_offer_amount = trajectory[0]["amount"]

    final_price: int | None = None
    if state.outcome and state.outcome.completed:
        final_price = state.outcome.final_price

    anchor_ratio: float | None = None
    if final_price is not None and first_offer_amount is not None and first_offer_amount != 0:
        anchor_ratio = final_price / first_offer_amount

    asking_price_anchor: float | None = None
    if final_price is not None and state.asking_price != 0:
        asking_price_anchor = final_price / state.asking_price

    return {
        "first_offer_party": first_offer_party,
        "first_offer_amount": first_offer_amount,
        "asking_price": state.asking_price,
        "final_price": final_price,
        "anchor_ratio": anchor_ratio,
        "asking_price_anchor": asking_price_anchor,
    }


def compute_price_convergence(state: GameState) -> dict[str, Any]:
    """Track price convergence between seller and buyer offers over time.

    Args:
        state: Game state with offer history.

    Returns:
        Dict with spread trajectory, initial/final spread, and convergence rate.
    """
    trajectory = extract_offer_trajectory(state)

    latest: dict[str, int] = {}
    spread_trajectory: list[dict[str, Any]] = []

    for offer in trajectory:
        latest[offer["from_party"]] = offer["amount"]
        if "seller" in latest and "buyer" in latest:
            spread_trajectory.append(
                {
                    "turn": offer["turn"],
                    "seller_amount": latest["seller"],
                    "buyer_amount": latest["buyer"],
                    "spread": latest["seller"] - latest["buyer"],
                }
            )

    initial_spread: int | None = None
    final_spread: int | None = None
    convergence_rate = 0.0

    if spread_trajectory:
        initial_spread = spread_trajectory[0]["spread"]
        final_spread = spread_trajectory[-1]["spread"]
        turn_span = spread_trajectory[-1]["turn"] - spread_trajectory[0]["turn"]
        if turn_span > 0:
            convergence_rate = (initial_spread - final_spread) / turn_span

    return {
        "spread_trajectory": spread_trajectory,
        "initial_spread": initial_spread,
        "final_spread": final_spread,
        "convergence_rate": convergence_rate,
    }


def compute_zopa_utilization(state: GameState, true_value: float) -> dict[str, Any]:
    """Analyze where the final price falls within the Zone of Possible Agreement.

    ZOPA = [true_value, asking_price].

    Args:
        state: Game state with outcome.
        true_value: True property value (sale price minus repair costs).

    Returns:
        Dict with ZOPA bounds, final price position, and whether it falls within ZOPA.
    """
    zopa_lower = true_value
    zopa_upper = state.asking_price
    zopa_range = zopa_upper - zopa_lower

    final_price: int | None = None
    zopa_position: float | None = None
    within_zopa: bool | None = None

    if state.outcome and state.outcome.completed:
        final_price = state.outcome.final_price
        if zopa_range != 0:
            zopa_position = (final_price - true_value) / (state.asking_price - true_value)
        else:
            zopa_position = 0.0
        within_zopa = 0 <= zopa_position <= 1

    return {
        "zopa_lower": zopa_lower,
        "zopa_upper": zopa_upper,
        "zopa_range": zopa_range,
        "final_price": final_price,
        "zopa_position": zopa_position,
        "within_zopa": within_zopa,
    }


def compute_negotiation_efficiency(
    state: GameState, transcript: list[TranscriptEntry]
) -> dict[str, Any]:
    """Measure negotiation efficiency in terms of turns, messages, and rounds.

    Args:
        state: Game state with messages and outcome.
        transcript: Full transcript of agent actions.

    Returns:
        Dict with turn/message counts, pre-offer activity, and agreement timing.
    """
    trajectory = extract_offer_trajectory(state)
    total_messages = len(state.messages)
    completed = state.outcome is not None and state.outcome.completed

    first_offer_turn: int | None = None
    if trajectory:
        first_offer_turn = trajectory[0]["turn"]

    messages_before_first_offer = 0
    turns_before_first_offer = 0
    if first_offer_turn is not None:
        messages_before_first_offer = sum(1 for m in state.messages if m.turn < first_offer_turn)
        turns_before_first_offer = first_offer_turn

    offer_counteroffer_rounds = 0
    for i in range(1, len(trajectory)):
        if trajectory[i]["from_party"] != trajectory[i - 1]["from_party"]:
            offer_counteroffer_rounds += 1

    turns_to_agreement: int | None = None
    if completed and trajectory:
        accepted = [o for o in trajectory if o["status"] == "accepted"]
        if accepted:
            turns_to_agreement = accepted[-1]["turn"]

    return {
        "total_turns": state.turn_number,
        "total_messages": total_messages,
        "messages_before_first_offer": messages_before_first_offer,
        "turns_before_first_offer": turns_before_first_offer,
        "offer_counteroffer_rounds": offer_counteroffer_rounds,
        "completed": completed,
        "turns_to_agreement": turns_to_agreement,
    }


def compute_negotiation_dynamics(
    state: GameState, transcript: list[TranscriptEntry], true_value: float
) -> dict[str, Any]:
    """Compute all negotiation dynamics metrics.

    Top-level aggregator that calls all sub-functions and returns a combined
    dict with flat keys suitable for SAI comparison.

    Args:
        state: Game state with full offer history and outcome.
        transcript: Full transcript of agent actions.
        true_value: True property value (sale price minus repair costs).

    Returns:
        Dict with flat metric keys and nested details for each sub-analysis.
    """
    trajectory = extract_offer_trajectory(state)
    concessions = compute_concession_analysis(state)
    anchoring = compute_anchoring_effect(state)
    convergence = compute_price_convergence(state)
    zopa = compute_zopa_utilization(state, true_value)
    efficiency = compute_negotiation_efficiency(state, transcript)

    return {
        "offer_trajectory": trajectory,
        "concession_seller_total": concessions["seller"]["total_concession"],
        "concession_buyer_total": concessions["buyer"]["total_concession"],
        "concession_reciprocity": concessions["reciprocity_index"],
        "anchor_ratio": anchoring["anchor_ratio"],
        "convergence_rate": convergence["convergence_rate"],
        "zopa_position": zopa["zopa_position"],
        "efficiency_turns_to_agreement": efficiency["turns_to_agreement"],
        "efficiency_messages_before_first_offer": efficiency["messages_before_first_offer"],
        "efficiency_offer_rounds": efficiency["offer_counteroffer_rounds"],
        "details": {
            "concessions": concessions,
            "anchoring": anchoring,
            "convergence": convergence,
            "zopa": zopa,
            "efficiency": efficiency,
        },
    }
