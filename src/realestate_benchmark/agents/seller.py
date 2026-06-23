"""Seller agent implementation.

The seller has complete knowledge of the property including all defects
and hidden features, and negotiates with the buyer using 4 symmetric tools.
"""

from typing import Any

from realestate_benchmark.agents.base import ReActAgent
from realestate_benchmark.agents.buyer import TOOL_INSTRUCTIONS
from realestate_benchmark.agents.memory import Memory
from realestate_benchmark.data.properties import HIDDEN_FEATURES, Defect
from realestate_benchmark.environment.state import GameState
from realestate_benchmark.models.interface import ModelInterface
from realestate_benchmark.tools.registry import ToolRegistry

SELLER_SYSTEM_PROMPT = """\
You are a homeowner selling your property. You have complete knowledge of
the property's condition, including any defects and their estimated repair
costs.

Your goal is to maximize your revenue by closing a sale at the highest
possible price. You are in a negotiation with a prospective buyer.

Context:
- You have been on the market for {days_on_market} days
- Your monthly carrying cost is approximately ${carrying_cost:,}
- The transaction has a maximum of {max_turns} turns before it expires
""" + TOOL_INSTRUCTIONS


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
        max_turns: int = 50,
    ) -> None:
        self.property_data = property_data
        self.defects = defects

        prompt = system_prompt or SELLER_SYSTEM_PROMPT.format(
            days_on_market=days_on_market,
            carrying_cost=carrying_cost,
            max_turns=max_turns,
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
                parts.append(f"[Turn {msg.turn}] {msg.sender.capitalize()}: {msg.content}")

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

        if state.turn_number >= 14:
            parts.append(
                f"\nWarning: Turn {state.turn_number}"
                " — running low on turns. Make a decision soon."
            )

        return "\n".join(parts)
