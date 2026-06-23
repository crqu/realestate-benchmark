"""Tool registry framework for managing agent actions.

Provides a single shared registry with 4 symmetric negotiation tools
available to both seller and buyer agents.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ToolDefinition(BaseModel):
    """Definition of a tool for LLM tool calling."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolRegistry:
    """Registry for managing tools and their implementations."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolDefinition] = {}
        self.handlers: dict[str, Callable[..., dict[str, Any]]] = {}

    def register(
        self, name: str, definition: ToolDefinition, handler: Callable[..., dict[str, Any]]
    ) -> None:
        self.tools[name] = definition
        self.handlers[name] = handler

    def get_definition(self, name: str) -> ToolDefinition:
        return self.tools[name]

    def get_handler(self, name: str) -> Callable[..., dict[str, Any]]:
        return self.handlers[name]

    def get_all_definitions(self) -> list[ToolDefinition]:
        return list(self.tools.values())

    def execute(self, name: str, params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        handler = self.handlers[name]
        return handler(params, context)


def create_registry() -> ToolRegistry:
    """Create the shared tool registry with 4 negotiation tools."""
    from .negotiation import accept_offer, make_offer, send_message, walk_away

    registry = ToolRegistry()

    registry.register(
        "send_message",
        ToolDefinition(
            name="send_message",
            description="Send a message to the other party",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Message content",
                    },
                },
                "required": ["content"],
            },
        ),
        send_message,
    )

    registry.register(
        "make_offer",
        ToolDefinition(
            name="make_offer",
            description=(
                "Make or update a purchase offer. If a previous offer exists, "
                "it is superseded by this one. Either party can make an offer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "integer",
                        "description": "Offer amount in dollars",
                    },
                },
                "required": ["amount"],
            },
        ),
        make_offer,
    )

    registry.register(
        "accept_offer",
        ToolDefinition(
            name="accept_offer",
            description=(
                "Accept the current pending offer from the other party. "
                "This completes the transaction at the offered price."
            ),
            parameters={"type": "object", "properties": {}},
        ),
        accept_offer,
    )

    registry.register(
        "walk_away",
        ToolDefinition(
            name="walk_away",
            description="Walk away from the negotiation without a deal",
            parameters={"type": "object", "properties": {}},
        ),
        walk_away,
    )

    return registry
