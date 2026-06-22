"""Game state data structures for real estate transaction benchmark.

This module defines all dataclasses for representing game state, including
listings, messages, offers, disclosures, and transaction outcomes.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GamePhase(StrEnum):
    """Phases of the transaction game."""

    INIT = "init"
    LISTING = "listing"
    DISCOVERY = "discovery"
    NEGOTIATION = "negotiation"
    CLOSED = "closed"
    TERMINATED = "terminated"


class DisclosureContext(StrEnum):
    """Context in which a disclosure was made (for P1 scoring)."""

    PROACTIVE = "proactive"  # Disclosed before buyer asked
    RESPONSIVE = "responsive"  # Disclosed after ask_about_condition
    FORMAL = "formal"  # Disclosed after request_disclosure


class Listing(BaseModel):
    """Property listing with public information."""

    property_id: str
    description: str
    asking_price: int
    public_features: dict[str, Any]


class Message(BaseModel):
    """A message between agents."""

    sender: str  # "seller" | "buyer"
    content: str
    turn: int
    timestamp: datetime = Field(default_factory=datetime.now)


class Offer(BaseModel):
    """A purchase offer from the buyer."""

    buyer_id: str
    amount: int
    contingencies: list[str] = Field(default_factory=list)
    status: str = "pending"  # "pending" | "accepted" | "rejected" | "countered"
    turn: int


class Disclosure(BaseModel):
    """A defect disclosure from the seller."""

    defect_type: str  # e.g., "basement", "electrical", "heating"
    description: str
    severity: str  # "minor" | "moderate" | "major" | "critical"
    context: DisclosureContext
    turn: int


class InspectionReport(BaseModel):
    """Results from a professional inspection."""

    findings: list[dict[str, Any]]  # List of detected defects
    cost: int
    turn: int


class MarketData(BaseModel):
    """Comparable sales and market statistics."""

    comparable_sales: list[dict[str, Any]] = Field(default_factory=list)
    median_price: float | None = None
    days_on_market: float | None = None


class TransactionOutcome(BaseModel):
    """Final outcome of the transaction."""

    completed: bool
    final_price: int = 0
    buyer_welfare_gap: float | None = None


class GameState(BaseModel):
    """Public game state visible to both agents."""

    # Game metadata
    game_id: str
    turn_number: int = 0
    phase: GamePhase = GamePhase.INIT

    # Property listing
    listing: Listing | None = None

    # Communication
    messages: list[Message] = Field(default_factory=list)

    # Offers
    offers: list[Offer] = Field(default_factory=list)
    current_offer: Offer | None = None

    # Disclosures (with context tracking)
    disclosures: list[Disclosure] = Field(default_factory=list)

    # Inspection results (if ordered)
    inspection_report: InspectionReport | None = None

    # Market reference
    market_data: MarketData = Field(default_factory=MarketData)

    # Outcome
    outcome: TransactionOutcome | None = None

    def add_message(self, sender: str, content: str) -> None:
        """Add a message to the message history.

        Args:
            sender: Agent ID ("seller" or "buyer")
            content: Message content
        """
        message = Message(sender=sender, content=content, turn=self.turn_number)
        self.messages.append(message)

    def add_disclosure(self, disclosure: Disclosure) -> None:
        """Add a disclosure to the disclosure history.

        Args:
            disclosure: Disclosure object to add
        """
        self.disclosures.append(disclosure)

    def transition_phase(self, new_phase: GamePhase) -> None:
        """Transition to a new game phase.

        Args:
            new_phase: The phase to transition to
        """
        self.phase = new_phase


class TranscriptEntry(BaseModel):
    """Single action record — the atomic unit of trajectory analysis."""

    entry_id: str
    game_id: str
    agent_id: str  # "seller" | "buyer"
    turn: int  # global sequence number
    phase: GamePhase  # which game phase
    tool_name: str  # which tool was called
    parameters: dict[str, Any]  # tool call parameters
    result: dict[str, Any]  # tool call result
    reasoning_trace: str  # agent's reasoning before this action
    timestamp: datetime = Field(default_factory=datetime.now)
    time_to_decide_ms: int = 0  # wall-clock time for this decision
    memory_snapshot: str | None = None  # agent's memory state at decision time
