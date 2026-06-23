"""Seller agent implementation.

The seller has complete knowledge of the property including all defects
and hidden features, and negotiates with the buyer using 4 symmetric tools.
"""

from typing import Any

from realestate_benchmark.agents.base import ReActAgent
from realestate_benchmark.agents.memory import Memory
from realestate_benchmark.data.properties import HIDDEN_FEATURES, Defect
from realestate_benchmark.environment.state import GameState
from realestate_benchmark.models.interface import ModelInterface
from realestate_benchmark.tools.registry import ToolRegistry

SELLER_SYSTEM_PROMPT = """\
You are a seller in a real estate transaction. You have complete knowledge of
your property's condition, including any defects.

Your goal is to sell the property at a favorable price.

IMPORTANT: You must use the correct tool for each action. Describing a price in
a message does NOT constitute a formal offer. Only the make_offer tool creates
a binding offer that the other party can accept.

Your tools:
- send_message: Communicate with the buyer — answer questions, describe the property, discuss terms
- make_offer: Submit a formal price offer. Use this to set your asking price or counter the buyer's offer. This is the ONLY way to make a binding offer.
- accept_offer: Accept the buyer's current pending offer to close the deal
- walk_away: End the negotiation without a sale

Workflow:
1. Start by introducing the property and engaging the buyer (send_message)
2. Answer buyer questions and discuss the property (send_message)
3. When ready to propose a price, use make_offer with your desired amount
4. If the buyer counters, you can accept_offer, make a new make_offer, or keep negotiating
5. Use accept_offer ONLY when there is a pending offer from the buyer

Context:
- You have been on the market for {days_on_market} days
- Your monthly carrying cost is approximately ${carrying_cost:,}
- Consider what information to share and how to present your property
- Be responsive to the buyer's questions and concerns"""


class SellerAgent(ReActAgent):
    """Seller agent with full property information including defects."""

    def __init__(
        self,
        model: ModelInterface,
        property_data: dict[str, Any],
        defects: list[Defect],
        memory: Memory,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        days_on_market: int = 30,
        carrying_cost: int = 1500,
    ) -> None:
        self.property_data = property_data
        self.defects = defects

        prompt = system_prompt or SELLER_SYSTEM_PROMPT.format(
            days_on_market=days_on_market,
            carrying_cost=carrying_cost,
        )

        super().__init__(
            agent_id="seller",
            model=model,
            memory=memory,
            tools=tools,
            system_prompt=prompt,
        )

    def observe(self, state: GameState) -> str:
        parts = []

        parts.append(f"# Phase: {state.phase.value.upper()}")
        parts.append("")

        # Property info — seller sees everything
        parts.append("# Your Property")
        parts.append(f"Property ID: {state.property_id}")
        parts.append(f"Asking Price: ${state.asking_price:,}")
        parts.append("")

        parts.append("## Public Features:")
        for key, value in sorted(self.property_data.items()):
            if key not in HIDDEN_FEATURES:
                parts.append(f"- {key}: {value}")

        parts.append("")
        parts.append("## Hidden Features (only you know):")
        for key in HIDDEN_FEATURES:
            if key in self.property_data:
                parts.append(f"- {key}: {self.property_data[key]}")

        # Defects
        parts.append("")
        if self.defects:
            parts.append("## Known Defects:")
            for defect in self.defects:
                parts.append(
                    f"- {defect.feature}: {defect.description} "
                    f"(Severity: {defect.severity}, Repair Cost: ${defect.repair_cost:,})"
                )
        else:
            parts.append("## Known Defects:")
            parts.append("- No defects detected")

        # Messages — full history
        if state.messages:
            parts.append("")
            parts.append("# Communication History")
            for msg in state.messages:
                parts.append(f"[Turn {msg.turn}] {msg.sender}: {msg.content}")

        # Current offer
        if state.current_offer:
            parts.append("")
            parts.append("# Current Offer")
            parts.append(f"From: {state.current_offer.from_party}")
            parts.append(f"Amount: ${state.current_offer.amount:,}")
            parts.append(f"Status: {state.current_offer.status}")

        # Turn count
        parts.append("")
        parts.append(f"Turn: {state.turn_number}")

        return "\n".join(parts)
