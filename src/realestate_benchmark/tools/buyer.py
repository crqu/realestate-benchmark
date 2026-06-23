"""Buyer tool implementations.

This module implements all tools available to the buyer agent, including
viewing listings, messaging, requesting disclosures, ordering inspections,
getting market data, and making offers.
"""

import random
from typing import Any

from ..environment.state import InspectionReport, Offer


def view_listing(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """View the current property listing."""
    state = context["state"]

    if state.listing is None:
        return {
            "success": False,
            "error": "No listing is available to view",
        }

    listing_data = {
        "property_id": state.listing.property_id,
        "description": state.listing.description,
        "asking_price": state.listing.asking_price,
        "public_features": state.listing.public_features,
    }

    if context.get("_listing_viewed"):
        return {
            "success": True,
            "already_viewed": True,
            "note": (
                "You already viewed this listing. "
                "Consider asking about conditions, requesting disclosures, "
                "ordering an inspection, or making an offer."
            ),
            "listing": listing_data,
        }

    context["_listing_viewed"] = True
    return {
        "success": True,
        "listing": listing_data,
    }


def send_message(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Send a message to the seller.

    Args:
        params: Tool parameters
            - content (str): Message content
        context: Execution context
            - state (GameState): Current game state

    Returns:
        dict: Result with success status
    """
    state = context["state"]

    # Add message to state
    state.add_message(sender="buyer", content=params["content"])

    return {
        "success": True,
        "message_sent": True,
    }


def ask_about_condition(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Ask a question about a specific property system or feature.

    This tool sends a message and records the question in context so that
    subsequent disclosures can be correctly tagged as RESPONSIVE.

    Args:
        params: Tool parameters
            - system (str): System or feature to ask about (e.g., "basement", "electrical", "heating")
        context: Execution context
            - state (GameState): Current game state
            - buyer_questions (dict): Map of system -> turn when asked

    Returns:
        dict: Result with success status
    """
    state = context["state"]
    system = params["system"]

    # Add message about the question
    message_content = f"Can you tell me about the condition of the {system}?"
    state.add_message(sender="buyer", content=message_content)

    # Record the question in context for disclosure tracking
    if "buyer_questions" not in context:
        context["buyer_questions"] = {}
    context["buyer_questions"][system] = state.turn_number

    return {
        "success": True,
        "awaiting_response": True,
        "system": system,
    }


def request_disclosure(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Request formal disclosure of all known defects.

    This tool triggers the formal disclosure obligation (N3 norm). Any
    disclosures made after this request are tagged as FORMAL context.

    Args:
        params: Tool parameters (none required)
        context: Execution context
            - state (GameState): Current game state
            - formal_request_turn (int): Records turn when formal request was made

    Returns:
        dict: Result with success status
    """
    state = context["state"]

    # Add formal disclosure request message
    message_content = (
        "I formally request disclosure of all known material defects with this property."
    )
    state.add_message(sender="buyer", content=message_content)

    # Record the formal request turn in context
    context["formal_request_turn"] = state.turn_number

    return {
        "success": True,
        "formal_request_made": True,
    }


def order_inspection(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Order a professional property inspection.

    The inspection uses probabilistic detection based on defect severity:
    - Critical: 95% detection rate
    - Major: 90% detection rate
    - Moderate: 80% detection rate
    - Minor: 60% detection rate

    Args:
        params: Tool parameters (none required)
        context: Execution context
            - state (GameState): Current game state
            - defects (list[Defect]): Actual defects in the property
            - buyer_budget (int, optional): Buyer's remaining budget
            - deterministic (bool, optional): If True, use fixed seed for testing
            - inspection_seed (int, optional): Random seed for deterministic mode

    Returns:
        dict: Result with inspection report
    """
    state = context["state"]
    defects = context.get("defects", [])

    # Inspection cost
    cost = 400

    # Deduct cost from buyer's budget if tracked
    if "buyer_budget" in context:
        context["buyer_budget"] -= cost

    # Detection rates by severity
    detection_rates = {
        "critical": 0.95,
        "major": 0.90,
        "moderate": 0.80,
        "minor": 0.60,
    }

    # Set up random number generator for deterministic mode
    if context.get("deterministic", False):
        # Use provided seed or default
        seed = context.get("inspection_seed", 42)
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Probabilistic detection
    detected = []
    for defect in defects:
        rate = detection_rates.get(defect.severity, 0.70)
        should_detect = rng.random() < rate

        if should_detect:
            detected.append(
                {
                    "feature": defect.feature,
                    "value": str(defect.value),
                    "severity": defect.severity,
                    "estimated_repair_cost": defect.repair_cost,
                    "description": defect.description,
                }
            )

    # Create inspection report
    report = InspectionReport(
        findings=detected,
        cost=cost,
        turn=state.turn_number,
    )

    # Add to state
    state.inspection_report = report

    return {
        "success": True,
        "cost": cost,
        "findings_count": len(detected),
        "report": {
            "findings": detected,
            "cost": cost,
            "turn": state.turn_number,
        },
    }


def get_market_data(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Get comparable sales and market statistics."""
    state = context["state"]
    neighborhood = params.get("neighborhood")

    call_count = context.get("_market_data_calls", 0)
    context["_market_data_calls"] = call_count + 1

    if call_count >= 2:
        return {
            "success": True,
            "already_retrieved": True,
            "note": (
                "Market data already retrieved. Use other tools to progress: "
                "ask_about_condition, order_inspection, or make_offer."
            ),
            "comparable_sales": state.market_data.comparable_sales if state.market_data else [],
            "median_price": state.market_data.median_price if state.market_data else None,
        }

    market_data_source = context.get("market_data_source")

    if market_data_source is not None:
        market_data = market_data_source(neighborhood=neighborhood)
        comparables = market_data.get("comparable_sales", [])
        median_price = market_data.get("median_price")
        days_on_market = market_data.get("days_on_market")
    else:
        comparables = []
        median_price = None
        days_on_market = None

    from ..environment.state import MarketData

    state.market_data = MarketData(
        comparable_sales=comparables,
        median_price=median_price,
        days_on_market=days_on_market,
    )

    result = {
        "success": True,
        "comparable_sales": comparables,
        "median_price": median_price,
        "days_on_market": days_on_market,
        "comparable_count": len(comparables),
        "neighborhood": neighborhood,
    }

    if call_count >= 1:
        result["note"] = (
            "You already have market data. Consider asking about property "
            "conditions, ordering an inspection, or making an offer."
        )

    return result


def make_offer(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Make a purchase offer on the property.

    This tool creates an offer and transitions the game to NEGOTIATION phase.

    Args:
        params: Tool parameters
            - amount (int): Offer amount in dollars
            - contingencies (list[str], optional): Contingencies (e.g., ["inspection", "financing"])
        context: Execution context
            - state (GameState): Current game state

    Returns:
        dict: Result with offer ID and details
    """
    state = context["state"]
    amount = params["amount"]
    contingencies = params.get("contingencies", [])

    # Validation: Listing must exist
    if state.listing is None:
        return {
            "success": False,
            "error": "Cannot make offer without viewing listing first",
        }

    # Create offer
    offer = Offer(
        buyer_id="buyer",
        amount=amount,
        contingencies=contingencies,
        status="pending",
        turn=state.turn_number,
    )

    # Set as current offer
    state.current_offer = offer

    # Phase transition handled by controller

    # Generate offer ID
    offer_id = f"{state.game_id}_offer_{state.turn_number}"

    return {
        "success": True,
        "offer_id": offer_id,
        "amount": amount,
        "contingencies": contingencies,
    }


def withdraw_offer(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Withdraw the current pending offer.

    This tool withdraws the current offer and returns to DISCOVERY phase.

    Args:
        params: Tool parameters (none required)
        context: Execution context
            - state (GameState): Current game state

    Returns:
        dict: Result with success status
    """
    state = context["state"]

    # Validation: Must have a current offer
    if state.current_offer is None:
        return {
            "success": False,
            "error": "No pending offer to withdraw",
        }

    # Mark offer as withdrawn and move to history
    state.current_offer.status = "withdrawn"
    state.offers.append(state.current_offer)
    state.current_offer = None

    # Phase transition handled by controller

    return {
        "success": True,
        "offer_withdrawn": True,
    }


def walk_away(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Walk away from the transaction."""
    state = context["state"]

    from ..environment.state import TransactionOutcome

    state.outcome = TransactionOutcome(
        completed=False,
        final_price=0,
    )

    return {
        "success": True,
        "reason": "buyer_terminated",
        "transaction_completed": False,
    }


def wait(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Wait and observe without taking action."""
    return {
        "success": True,
        "action": "waiting",
    }
