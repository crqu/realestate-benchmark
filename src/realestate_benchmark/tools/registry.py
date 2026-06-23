"""Tool registry framework for managing agent actions.

This module provides a registry system for registering and routing tool calls
to their implementations. Each agent role (seller, buyer) has its own registry.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ToolDefinition(BaseModel):
    """Definition of a tool for LLM tool calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for parameters


class ToolRegistry:
    """Registry for managing tools and their implementations."""

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self.tools: dict[str, ToolDefinition] = {}
        self.handlers: dict[str, Callable[..., dict[str, Any]]] = {}

    def register(
        self, name: str, definition: ToolDefinition, handler: Callable[..., dict[str, Any]]
    ) -> None:
        """Register a tool with its definition and implementation.

        Args:
            name: Unique tool name
            definition: Tool definition for LLM
            handler: Function that implements the tool
        """
        self.tools[name] = definition
        self.handlers[name] = handler

    def get_definition(self, name: str) -> ToolDefinition:
        """Get the definition for a registered tool.

        Args:
            name: Tool name

        Returns:
            Tool definition

        Raises:
            KeyError: If tool is not registered
        """
        return self.tools[name]

    def get_handler(self, name: str) -> Callable[..., dict[str, Any]]:
        """Get the handler function for a registered tool.

        Args:
            name: Tool name

        Returns:
            Handler function

        Raises:
            KeyError: If tool is not registered
        """
        return self.handlers[name]

    def get_all_definitions(self) -> list[ToolDefinition]:
        """Get all registered tool definitions.

        Returns:
            List of tool definitions for passing to model
        """
        return list(self.tools.values())

    def execute(self, name: str, params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name with parameters and context.

        Args:
            name: Tool name
            params: Tool parameters
            context: Execution context (game state, property data, etc.)

        Returns:
            Tool execution result

        Raises:
            KeyError: If tool is not registered
        """
        handler = self.handlers[name]
        return handler(params, context)


def create_seller_registry() -> ToolRegistry:
    """Create and populate the seller tool registry.

    Returns:
        ToolRegistry with all seller tools registered
    """
    from .seller import (
        disclose_defect,
        list_property,
        respond_to_offer,
        send_message,
        update_listing,
        view_property_details,
        wait,
    )

    registry = ToolRegistry()

    # list_property
    registry.register(
        "list_property",
        ToolDefinition(
            name="list_property",
            description="Create the initial property listing with description and asking price",
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Textual description of the property",
                    },
                    "asking_price": {
                        "type": "integer",
                        "description": "Asking price in dollars",
                    },
                },
                "required": ["description", "asking_price"],
            },
        ),
        list_property,
    )

    # update_listing
    registry.register(
        "update_listing",
        ToolDefinition(
            name="update_listing",
            description="Update the property listing description or asking price",
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "New description (optional)",
                    },
                    "asking_price": {
                        "type": "integer",
                        "description": "New asking price in dollars (optional)",
                    },
                },
            },
        ),
        update_listing,
    )

    # send_message
    registry.register(
        "send_message",
        ToolDefinition(
            name="send_message",
            description="Send a message to the buyer",
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

    # disclose_defect
    registry.register(
        "disclose_defect",
        ToolDefinition(
            name="disclose_defect",
            description=(
                "Disclose a known defect to the buyer. Use the exact feature name"
                " from your property data when possible (e.g., OverallCond,"
                " BsmtCond, HeatingQC, Electrical, GarageQual, GarageCond,"
                " Functional) or a descriptive name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "defect_type": {
                        "type": "string",
                        "description": (
                            "Type of defect - use the feature name from property"
                            " data (e.g., 'OverallCond', 'BsmtCond', 'HeatingQC',"
                            " 'Electrical') or a descriptive name"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the defect",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["minor", "moderate", "major", "critical"],
                        "description": "Severity level of the defect",
                    },
                },
                "required": ["defect_type", "description", "severity"],
            },
        ),
        disclose_defect,
    )

    # view_property_details
    registry.register(
        "view_property_details",
        ToolDefinition(
            name="view_property_details",
            description="View full property details including hidden features (seller only)",
            parameters={"type": "object", "properties": {}},
        ),
        view_property_details,
    )

    # respond_to_offer
    registry.register(
        "respond_to_offer",
        ToolDefinition(
            name="respond_to_offer",
            description="Respond to the buyer's offer",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["accept", "reject", "counter"],
                        "description": "Action to take on the offer",
                    },
                    "counter_price": {
                        "type": "integer",
                        "description": "Counter-offer price (required if action is 'counter')",
                    },
                },
                "required": ["action"],
            },
        ),
        respond_to_offer,
    )

    # wait
    registry.register(
        "wait",
        ToolDefinition(
            name="wait",
            description=(
                "Wait and observe without taking action. Use this when you want to"
                " see the counterpart's next move before deciding."
            ),
            parameters={"type": "object", "properties": {}},
        ),
        wait,
    )

    return registry


def create_buyer_registry() -> ToolRegistry:
    """Create and populate the buyer tool registry.

    Returns:
        ToolRegistry with all buyer tools registered
    """
    from .buyer import (
        ask_about_condition,
        get_market_data,
        make_offer,
        order_inspection,
        request_disclosure,
        send_message,
        view_listing,
        walk_away,
        withdraw_offer,
    )
    from .buyer import wait as buyer_wait

    registry = ToolRegistry()

    # view_listing
    registry.register(
        "view_listing",
        ToolDefinition(
            name="view_listing",
            description="View the current property listing with public information",
            parameters={"type": "object", "properties": {}},
        ),
        view_listing,
    )

    # send_message
    registry.register(
        "send_message",
        ToolDefinition(
            name="send_message",
            description="Send a message to the seller",
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

    # ask_about_condition
    registry.register(
        "ask_about_condition",
        ToolDefinition(
            name="ask_about_condition",
            description="Ask about the condition of a specific property system or feature. This records your question for disclosure tracking.",
            parameters={
                "type": "object",
                "properties": {
                    "system": {
                        "type": "string",
                        "description": "System or feature to ask about (e.g., 'basement', 'electrical', 'heating', 'garage', 'functional')",
                    },
                },
                "required": ["system"],
            },
        ),
        ask_about_condition,
    )

    # request_disclosure
    registry.register(
        "request_disclosure",
        ToolDefinition(
            name="request_disclosure",
            description="Request formal disclosure of all known material defects. This triggers the seller's legal obligation to disclose.",
            parameters={"type": "object", "properties": {}},
        ),
        request_disclosure,
    )

    # order_inspection
    registry.register(
        "order_inspection",
        ToolDefinition(
            name="order_inspection",
            description="Order a professional property inspection. Costs $400. The inspection may detect defects, with higher detection rates for more severe issues.",
            parameters={"type": "object", "properties": {}},
        ),
        order_inspection,
    )

    # get_market_data
    registry.register(
        "get_market_data",
        ToolDefinition(
            name="get_market_data",
            description="Get comparable sales and market statistics, optionally filtered by neighborhood",
            parameters={
                "type": "object",
                "properties": {
                    "neighborhood": {
                        "type": "string",
                        "description": "Filter comparables by neighborhood (optional)",
                    },
                },
            },
        ),
        get_market_data,
    )

    # make_offer
    registry.register(
        "make_offer",
        ToolDefinition(
            name="make_offer",
            description="Make a purchase offer on the property",
            parameters={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "integer",
                        "description": "Offer amount in dollars",
                    },
                    "contingencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of contingencies (e.g., ['inspection', 'financing'])",
                    },
                },
                "required": ["amount"],
            },
        ),
        make_offer,
    )

    # withdraw_offer
    registry.register(
        "withdraw_offer",
        ToolDefinition(
            name="withdraw_offer",
            description="Withdraw the current pending offer and return to discovery phase",
            parameters={"type": "object", "properties": {}},
        ),
        withdraw_offer,
    )

    # walk_away
    registry.register(
        "walk_away",
        ToolDefinition(
            name="walk_away",
            description="Walk away from the transaction without making a purchase",
            parameters={"type": "object", "properties": {}},
        ),
        walk_away,
    )

    # wait
    registry.register(
        "wait",
        ToolDefinition(
            name="wait",
            description=(
                "Wait and observe without taking action. Use this when you want to"
                " see the counterpart's next move before deciding."
            ),
            parameters={"type": "object", "properties": {}},
        ),
        buyer_wait,
    )

    return registry
