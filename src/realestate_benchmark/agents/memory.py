"""Agent memory system with read/write/append operations and snapshotting.

This module implements the Memory class that agents use to maintain their
internal state across game turns. Memory is stored as markdown text and
can be snapshotted to the database for persistence and historical retrieval.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from realestate_benchmark.environment.database import Database


MEMORY_TEMPLATE = """# Agent Memory

## Facts
- [Record key facts about the property, transaction, and counterpart here]

## Observations
- [Note observations about the game state and counterpart behavior]

## Strategy
- [Track strategic decisions and reasoning]
"""


class Memory:
    """Agent memory with persistent storage and snapshot support.

    Memory is stored as markdown text with three sections:
    - Facts: Key information about property, transaction, counterpart
    - Observations: Notes about game state and behavior
    - Strategy: Strategic decisions and reasoning

    Snapshots are saved to the database with turn numbers for historical retrieval.

    Attributes:
        agent_id: Unique identifier for the agent
        db: Database instance for persistence
        game_id: Game identifier for storing snapshots
        content: Current memory content as markdown string
        turn: Last snapshot turn number

    Example:
        >>> db = Database("game.db")
        >>> memory = Memory("seller", db, "game-123")
        >>> memory.append("- Property has basement moisture issue")
        >>> memory.snapshot(turn=5)
        >>> old_content = memory.load_snapshot(turn=5)
    """

    def __init__(self, agent_id: str, db: "Database", game_id: str) -> None:
        """Initialize agent memory.

        Args:
            agent_id: Unique identifier for the agent (e.g., "seller", "buyer")
            db: Database instance for persistence
            game_id: Game identifier for storing snapshots
        """
        self.agent_id = agent_id
        self.db = db
        self.game_id = game_id
        self.content = MEMORY_TEMPLATE
        self.turn = 0

    def read(self) -> str:
        """Return current memory content.

        Returns:
            Current memory content as markdown string
        """
        return self.content

    def write(self, content: str) -> None:
        """Replace entire memory content.

        Args:
            content: New memory content as markdown string
        """
        self.content = content

    def append(self, note: str) -> None:
        """Add a note to the memory content.

        The note is appended to the end of the current content.
        For structured updates to specific sections, use write() with
        the full updated content.

        Args:
            note: Text to append to memory (can be single line or multi-line)
        """
        if not self.content.endswith("\n"):
            self.content += "\n"
        self.content += note
        if not note.endswith("\n"):
            self.content += "\n"

    def snapshot(self, turn: int) -> None:
        """Save current memory state to database with turn number.

        Snapshots allow historical retrieval and analysis of agent
        memory evolution over the course of the game.

        Args:
            turn: Turn number for this snapshot
        """
        self.turn = turn
        self.db.save_memory_snapshot(
            game_id=self.game_id,
            agent_id=self.agent_id,
            turn=turn,
            content=self.content,
        )

    def load_snapshot(self, turn: int) -> str:
        """Retrieve historical memory state from database.

        Args:
            turn: Turn number to retrieve

        Returns:
            Memory content at the specified turn

        Raises:
            ValueError: If snapshot for the specified turn does not exist
        """
        # Database returns the content directly
        # If no snapshot exists, the database method should raise an error
        content = self.db.load_memory_snapshot(
            game_id=self.game_id,
            agent_id=self.agent_id,
            turn=turn,
        )
        return content
