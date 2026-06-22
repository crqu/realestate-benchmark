"""Database interface for game state persistence.

This module implements the full database schema with SQLite + WAL mode
for concurrent access and crash recovery.

The database stores:
- Game metadata (games table)
- Game state snapshots (game_state table)
- Full action transcripts (transcript table)
- Agent memory snapshots (memory_snapshots table)

WAL (Write-Ahead Logging) mode is enabled for better concurrency and crash recovery.
"""

import json
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .state import GameState, TranscriptEntry


class Database:
    """Database interface for persisting game state, transcripts, and memory snapshots.

    Uses SQLite with WAL (Write-Ahead Logging) mode for concurrent access.
    All JSON fields are stored as TEXT with proper serialization.

    Attributes:
        db_path: Path to the SQLite database file
        _connection: Active SQLite connection
    """

    # SQL schema definitions
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        config TEXT,
        status TEXT
    );

    CREATE TABLE IF NOT EXISTS game_state (
        game_id TEXT PRIMARY KEY,
        turn INTEGER,
        phase TEXT,
        listing TEXT,
        messages TEXT,
        offers TEXT,
        disclosures TEXT,
        inspection_report TEXT,
        market_data TEXT,
        outcome TEXT,
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    );

    CREATE TABLE IF NOT EXISTS transcript (
        entry_id TEXT PRIMARY KEY,
        game_id TEXT,
        agent_id TEXT,
        turn INTEGER,
        phase TEXT,
        tool_name TEXT,
        parameters TEXT,
        result TEXT,
        reasoning_trace TEXT,
        timestamp TIMESTAMP,
        time_to_decide_ms INTEGER,
        memory_snapshot TEXT,
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    );

    CREATE TABLE IF NOT EXISTS memory_snapshots (
        game_id TEXT,
        agent_id TEXT,
        turn INTEGER,
        content TEXT,
        PRIMARY KEY (game_id, agent_id, turn),
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    );

    CREATE INDEX IF NOT EXISTS idx_transcript_game_turn ON transcript(game_id, turn);
    CREATE INDEX IF NOT EXISTS idx_transcript_game_agent ON transcript(game_id, agent_id);
    """

    def __init__(self, db_path: str) -> None:
        """Initialize database connection with WAL mode.

        Args:
            db_path: Path to SQLite database file. Will be created if it doesn't exist.
        """
        self.db_path = db_path

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Open connection and enable WAL mode
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row  # Enable dict-like access

        # Enable WAL mode for better concurrency
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")  # Balance safety and performance

        # Create schema if it doesn't exist
        self._connection.executescript(self.SCHEMA)
        self._connection.commit()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for atomic transactions.

        Yields:
            A cursor for executing SQL commands

        Raises:
            Any exception that occurs during the transaction
        """
        cursor = self._connection.cursor()
        try:
            yield cursor
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        finally:
            cursor.close()

    def create_game(self, config: dict[str, Any]) -> str:
        """Create a new game and return game_id.

        Args:
            config: Game configuration dictionary (will be serialized to JSON)

        Returns:
            Newly created game_id (UUID4 string)
        """
        game_id = str(uuid.uuid4())
        config_json = json.dumps(config)
        status = "running"

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO games (game_id, config, status)
                VALUES (?, ?, ?)
                """,
                (game_id, config_json, status),
            )

        return game_id

    def update_game_status(self, game_id: str, status: str) -> None:
        """Update game status.

        Args:
            game_id: Game identifier
            status: New status ("running" | "completed" | "terminated")
        """
        with self._transaction() as cursor:
            cursor.execute(
                "UPDATE games SET status = ? WHERE game_id = ?",
                (status, game_id),
            )

    def save_state(self, game_id: str, state: GameState) -> None:
        """Save game state to database.

        Uses INSERT OR REPLACE to handle both initial save and updates.

        Args:
            game_id: Game identifier
            state: GameState object to persist
        """
        # Serialize all complex fields to JSON (mode='json' handles datetime serialization)
        listing_json = (
            json.dumps(state.listing.model_dump(mode="json")) if state.listing else None
        )
        messages_json = json.dumps([m.model_dump(mode="json") for m in state.messages])
        offers_json = json.dumps([o.model_dump(mode="json") for o in state.offers])
        disclosures_json = json.dumps([d.model_dump(mode="json") for d in state.disclosures])
        inspection_json = (
            json.dumps(state.inspection_report.model_dump(mode="json"))
            if state.inspection_report
            else None
        )
        market_data_json = json.dumps(state.market_data.model_dump(mode="json"))
        outcome_json = (
            json.dumps(state.outcome.model_dump(mode="json")) if state.outcome else None
        )

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO game_state (
                    game_id, turn, phase, listing, messages, offers,
                    disclosures, inspection_report, market_data, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    state.turn_number,
                    state.phase.value,
                    listing_json,
                    messages_json,
                    offers_json,
                    disclosures_json,
                    inspection_json,
                    market_data_json,
                    outcome_json,
                ),
            )

    def load_state(self, game_id: str) -> GameState:
        """Load game state from database.

        Args:
            game_id: Game identifier

        Returns:
            GameState object

        Raises:
            ValueError: If game_id does not exist
        """
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT * FROM game_state WHERE game_id = ?",
            (game_id,),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            raise ValueError(f"Game state not found for game_id: {game_id}")

        # Deserialize JSON fields
        state_dict: dict[str, Any] = {
            "game_id": game_id,
            "turn_number": row["turn"],
            "phase": row["phase"],
            "listing": json.loads(row["listing"]) if row["listing"] else None,
            "messages": json.loads(row["messages"]) if row["messages"] else [],
            "offers": json.loads(row["offers"]) if row["offers"] else [],
            "disclosures": json.loads(row["disclosures"]) if row["disclosures"] else [],
            "inspection_report": (
                json.loads(row["inspection_report"]) if row["inspection_report"] else None
            ),
            "market_data": json.loads(row["market_data"]) if row["market_data"] else {},
            "outcome": json.loads(row["outcome"]) if row["outcome"] else None,
        }

        return GameState.model_validate(state_dict)

    def append_transcript(self, entry: TranscriptEntry) -> None:
        """Append entry to game transcript.

        Args:
            entry: TranscriptEntry object to persist
        """
        # Serialize complex fields
        parameters_json = json.dumps(entry.parameters)
        result_json = json.dumps(entry.result)

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO transcript (
                    entry_id, game_id, agent_id, turn, phase, tool_name,
                    parameters, result, reasoning_trace, timestamp,
                    time_to_decide_ms, memory_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.game_id,
                    entry.agent_id,
                    entry.turn,
                    entry.phase.value,
                    entry.tool_name,
                    parameters_json,
                    result_json,
                    entry.reasoning_trace,
                    entry.timestamp.isoformat(),
                    entry.time_to_decide_ms,
                    entry.memory_snapshot,
                ),
            )

    def load_transcript(self, game_id: str) -> list[TranscriptEntry]:
        """Load full transcript for a game.

        Args:
            game_id: Game identifier

        Returns:
            List of TranscriptEntry objects ordered by turn
        """
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT * FROM transcript
            WHERE game_id = ?
            ORDER BY turn ASC
            """,
            (game_id,),
        )
        rows = cursor.fetchall()
        cursor.close()

        entries = []
        for row in rows:
            entry_dict = {
                "entry_id": row["entry_id"],
                "game_id": row["game_id"],
                "agent_id": row["agent_id"],
                "turn": row["turn"],
                "phase": row["phase"],
                "tool_name": row["tool_name"],
                "parameters": json.loads(row["parameters"]),
                "result": json.loads(row["result"]),
                "reasoning_trace": row["reasoning_trace"],
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "time_to_decide_ms": row["time_to_decide_ms"],
                "memory_snapshot": row["memory_snapshot"],
            }
            entries.append(TranscriptEntry.model_validate(entry_dict))

        return entries

    def save_memory_snapshot(self, game_id: str, agent_id: str, turn: int, content: str) -> None:
        """Save agent memory snapshot to database.

        Uses INSERT OR REPLACE to handle both initial save and updates.

        Args:
            game_id: Game identifier
            agent_id: Agent identifier (e.g., "seller", "buyer")
            turn: Turn number
            content: Memory content as string
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO memory_snapshots (game_id, agent_id, turn, content)
                VALUES (?, ?, ?, ?)
                """,
                (game_id, agent_id, turn, content),
            )

    def load_memory_snapshot(self, game_id: str, agent_id: str, turn: int) -> str:
        """Load agent memory snapshot from database.

        Args:
            game_id: Game identifier
            agent_id: Agent identifier
            turn: Turn number

        Returns:
            Memory content at specified turn

        Raises:
            ValueError: If snapshot does not exist
        """
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT content FROM memory_snapshots
            WHERE game_id = ? AND agent_id = ? AND turn = ?
            """,
            (game_id, agent_id, turn),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            raise ValueError(
                f"Memory snapshot not found for game_id={game_id}, agent_id={agent_id}, turn={turn}"
            )

        return str(row["content"])

    def get_latest_memory_snapshot(self, game_id: str, agent_id: str) -> tuple[int, str] | None:
        """Get the most recent memory snapshot for an agent.

        Args:
            game_id: Game identifier
            agent_id: Agent identifier

        Returns:
            Tuple of (turn, content) for the latest snapshot, or None if no snapshots exist
        """
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT turn, content FROM memory_snapshots
            WHERE game_id = ? AND agent_id = ?
            ORDER BY turn DESC
            LIMIT 1
            """,
            (game_id, agent_id),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            return None

        return (row["turn"], row["content"])

    def close(self) -> None:
        """Close database connection.

        Should be called when done with the database to ensure all changes are flushed.
        """
        if self._connection:
            self._connection.commit()
            self._connection.close()

    def __enter__(self) -> "Database":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures connection is closed."""
        self.close()
