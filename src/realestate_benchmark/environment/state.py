"""Game state data structures for real estate transaction benchmark.

This module defines all dataclasses for representing game state, including
messages, offers, and transaction outcomes. The simplified design uses
free-form conversation with 3 phases (ACTIVE, CLOSED, TERMINATED) and
4 symmetric tools.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GamePhase(StrEnum):
    """Phases of the transaction game."""

    ACTIVE = "active"
    CLOSED = "closed"
    TERMINATED = "terminated"


class Message(BaseModel):
    """A message between agents."""

    sender: str  # "seller" | "buyer"
    content: str
    turn: int
    timestamp: datetime = Field(default_factory=datetime.now)


class Offer(BaseModel):
    """A purchase offer from either party."""

    from_party: str  # "seller" | "buyer"
    amount: int
    status: str = "pending"  # "pending" | "accepted" | "superseded"
    turn: int


class TransactionOutcome(BaseModel):
    """Final outcome of the transaction."""

    completed: bool
    final_price: int = 0
    buyer_welfare_gap: float | None = None


class GameState(BaseModel):
    """Game state shared between agents."""

    game_id: str
    turn_number: int = 0
    phase: GamePhase = GamePhase.ACTIVE

    property_id: str
    asking_price: int

    messages: list[Message] = Field(default_factory=list)

    offers: list[Offer] = Field(default_factory=list)
    current_offer: Offer | None = None

    outcome: TransactionOutcome | None = None

    def add_message(self, sender: str, content: str) -> None:
        message = Message(sender=sender, content=content, turn=self.turn_number)
        self.messages.append(message)

    def transition_phase(self, new_phase: GamePhase) -> None:
        self.phase = new_phase


class TranscriptEntry(BaseModel):
    """Single action record for trajectory analysis."""

    entry_id: str
    game_id: str
    agent_id: str
    turn: int
    phase: GamePhase
    tool_name: str
    parameters: dict[str, Any]
    result: dict[str, Any]
    reasoning_trace: str
    timestamp: datetime = Field(default_factory=datetime.now)
    time_to_decide_ms: int = 0
    memory_snapshot: str | None = None
