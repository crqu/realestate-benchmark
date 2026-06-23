"""Negotiation tool handlers shared by both seller and buyer agents.

Four tools for free-form negotiation:
- send_message: communicate with counterpart
- make_offer: create or replace a pending offer (either party)
- accept_offer: accept the counterpart's pending offer
- walk_away: terminate the negotiation
"""

from typing import Any

from realestate_benchmark.environment.state import Offer, TransactionOutcome


def send_message(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    state = context["state"]
    agent_id = context["agent_id"]
    content = params["content"]
    state.add_message(agent_id, content)
    return {"success": True, "message_sent": content}


def make_offer(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    state = context["state"]
    agent_id = context["agent_id"]
    amount = params["amount"]

    if state.current_offer is not None:
        state.current_offer.status = "superseded"
        state.offers.append(state.current_offer)

    state.current_offer = Offer(
        from_party=agent_id,
        amount=amount,
        turn=state.turn_number,
    )
    return {"success": True, "from_party": agent_id, "amount": amount}


def accept_offer(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    state = context["state"]
    agent_id = context["agent_id"]

    if state.current_offer is None:
        return {"success": False, "error": "No pending offer to accept"}

    if state.current_offer.from_party == agent_id:
        return {"success": False, "error": "Cannot accept your own offer"}

    state.current_offer.status = "accepted"
    state.outcome = TransactionOutcome(
        completed=True,
        final_price=state.current_offer.amount,
    )
    return {
        "success": True,
        "final_price": state.current_offer.amount,
        "accepted_from": state.current_offer.from_party,
    }


def walk_away(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    state = context["state"]
    agent_id = context["agent_id"]
    state.outcome = TransactionOutcome(completed=False, final_price=0)
    return {"success": True, "walked_away_by": agent_id}
