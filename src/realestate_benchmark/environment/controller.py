"""Game controller for orchestrating turn-based transactions.

Manages turns, phase transitions, action validation, and transcript logging.
Runs the full game loop between seller and buyer agents until a transaction
completes or the game terminates.
"""

import time
import uuid
from typing import TYPE_CHECKING, Any

from realestate_benchmark.environment.database import Database
from realestate_benchmark.environment.state import (
    GamePhase,
    GameState,
    TransactionOutcome,
    TranscriptEntry,
)

if TYPE_CHECKING:
    from realestate_benchmark.agents.base import ReActAgent


class GameController:
    """Orchestrates turn-based game between seller and buyer agents."""

    PHASE_TRANSITIONS: dict[str, GamePhase] = {
        "accept_offer": GamePhase.CLOSED,
        "walk_away": GamePhase.TERMINATED,
    }

    def __init__(
        self,
        seller: "ReActAgent",
        buyer: "ReActAgent",
        db: Database,
        config: dict[str, Any],
        property_data: dict[str, Any] | None = None,
        defects: list[Any] | None = None,
    ) -> None:
        self.seller = seller
        self.buyer = buyer
        self.db = db
        self.config = config
        self.property_data = property_data or {}
        self.defects = defects or []

        self.state: GameState | None = None

        self.context: dict[str, Any] = {
            "property_data": self.property_data,
            "property_defects": self.defects,
            "buyer_budget": config.get("buyer_budget", 200000),
        }
        self._consecutive_invalid = 0

    def initialize(self) -> str:
        game_id = self.db.create_game(self.config)

        self.state = GameState(
            game_id=game_id,
            phase=GamePhase.ACTIVE,
            property_id=self.config.get("property_id", ""),
            asking_price=self.config.get("asking_price", 0),
        )

        self.db.save_state(game_id, self.state)
        self.context["game_id"] = game_id
        return game_id

    def run(self) -> TransactionOutcome:
        if self.state is None:
            raise RuntimeError("Game not initialized. Call initialize() first.")

        max_turns = self.config.get("max_turns", 50)

        while self.state.turn_number < max_turns:
            if self.state.phase in (GamePhase.CLOSED, GamePhase.TERMINATED):
                break

            agent_id = "seller" if self.state.turn_number % 2 == 0 else "buyer"
            success = self.execute_turn(agent_id)

            if not success:
                self.state.outcome = TransactionOutcome(completed=False, final_price=0)
                self.state.phase = GamePhase.TERMINATED
                self.db.save_state(self.state.game_id, self.state)
                break

            self.state.turn_number += 1
            self.db.save_state(self.state.game_id, self.state)

        if self.state.turn_number >= max_turns and self.state.outcome is None:
            self.state.outcome = TransactionOutcome(completed=False, final_price=0)
            self.state.phase = GamePhase.TERMINATED
            self.db.save_state(self.state.game_id, self.state)

        final_status = "completed" if self.state.phase == GamePhase.CLOSED else "terminated"
        self.db.update_game_status(self.state.game_id, final_status)

        return self.state.outcome or TransactionOutcome(completed=False, final_price=0)

    def execute_turn(self, agent_id: str) -> bool:
        if self.state is None:
            raise RuntimeError("Game not initialized")

        agent = self.seller if agent_id == "seller" else self.buyer

        timeout_seconds = self.config.get("timeout_seconds", 60)
        start_time = time.time()

        try:
            tool_name, parameters, reasoning_trace = agent.act(self.state, self.context)
            elapsed_ms = int((time.time() - start_time) * 1000)

            if elapsed_ms / 1000 > timeout_seconds:
                return False

            # Text-only turn: agent said something but didn't call a tool
            if tool_name == "pass":
                self.log_action(
                    agent_id=agent_id,
                    tool_name="pass",
                    parameters=parameters,
                    result={"success": True, "action": "pass"},
                    reasoning_trace=reasoning_trace,
                    time_ms=elapsed_ms,
                )
                return True

            if not self.validate_action(agent_id, tool_name, parameters):
                self._consecutive_invalid += 1
                self.log_action(
                    agent_id=agent_id,
                    tool_name=tool_name,
                    parameters=parameters,
                    result={"success": False, "error": "invalid action", "skipped": True},
                    reasoning_trace=reasoning_trace,
                    time_ms=elapsed_ms,
                )
                if self._consecutive_invalid >= 6:
                    print(f"Terminating: {self._consecutive_invalid} consecutive invalid actions")
                    return False
                return True

            self._consecutive_invalid = 0
            self.context["state"] = self.state
            self.context["agent_id"] = agent_id

            result = agent.tools.execute(tool_name, parameters, self.context)

            if tool_name in self.PHASE_TRANSITIONS:
                self.state.transition_phase(self.PHASE_TRANSITIONS[tool_name])

            self.log_action(
                agent_id=agent_id,
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                reasoning_trace=reasoning_trace,
                time_ms=elapsed_ms,
            )

            if self.config.get("enable_reflection", True):
                agent.reflect(self.state, tool_name, result)

            memory_content = agent.memory.read()
            self.db.save_memory_snapshot(
                self.state.game_id, agent_id, self.state.turn_number, memory_content
            )

            return True

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.log_action(
                agent_id=agent_id,
                tool_name="ERROR",
                parameters={},
                result={"error": str(e)},
                reasoning_trace="",
                time_ms=elapsed_ms,
            )
            return False

    def validate_action(self, agent_id: str, tool_name: str, params: dict[str, Any]) -> bool:
        if self.state is None:
            return False

        if self.state.phase != GamePhase.ACTIVE:
            return False

        valid_tools = {
            "send_message", "make_offer", "accept_offer", "walk_away",
            "draft_contract", "sign_contract",
            "review_inspection_waiver", "waive_inspection",
            "draft_disclosure", "send_disclosure",
        }
        if tool_name not in valid_tools:
            return False

        if tool_name == "accept_offer":
            if self.state.current_offer is None:
                return False
            if self.state.current_offer.from_party == agent_id:
                return False

        return True

    def log_action(
        self,
        agent_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
        reasoning_trace: str,
        time_ms: int,
    ) -> None:
        if self.state is None:
            raise RuntimeError("Game not initialized")

        if agent_id == "seller":
            memory_snapshot = self.seller.memory.read()
        else:
            memory_snapshot = self.buyer.memory.read()

        entry = TranscriptEntry(
            entry_id=str(uuid.uuid4()),
            game_id=self.state.game_id,
            agent_id=agent_id,
            turn=self.state.turn_number,
            phase=self.state.phase,
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            reasoning_trace=reasoning_trace,
            time_to_decide_ms=time_ms,
            memory_snapshot=memory_snapshot,
        )

        self.db.append_transcript(entry)
