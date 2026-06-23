"""Game controller for orchestrating turn-based transactions.

This module implements the GameController that manages turns, phase transitions,
action validation, and transcript logging. It runs the full game loop between
seller and buyer agents until a transaction completes or the game terminates.
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
    """Orchestrates turn-based game between seller and buyer agents.

    The controller manages:
    - Turn alternation between agents
    - Phase transitions based on agent actions
    - Action validation against current phase
    - Transcript logging and memory snapshots
    - Timeout handling for agent decisions
    - Database persistence of all state

    Attributes:
        seller: Seller agent instance
        buyer: Buyer agent instance
        db: Database instance for persistence
        config: Game configuration dictionary
        state: Current game state
        context: Shared context for tool execution (tracks questions, defects, etc.)
    """

    # Phase transition rules: tool_name -> new_phase
    # These define which tools trigger phase changes
    PHASE_TRANSITIONS = {
        "list_property": GamePhase.LISTING,
        "view_listing": GamePhase.DISCOVERY,
        "make_offer": GamePhase.NEGOTIATION,
        "withdraw_offer": GamePhase.DISCOVERY,
        "walk_away": GamePhase.TERMINATED,
    }

    # Action validation: phase -> set of allowed tool names
    # This prevents illegal actions (e.g., can't make offer before viewing listing)
    PHASE_ALLOWED_ACTIONS = {
        GamePhase.INIT: {
            "list_property",
            "view_property_details",
            "wait",
        },
        GamePhase.LISTING: {
            "view_listing",
            "update_listing",
            "view_property_details",
            "send_message",
            "wait",
        },
        GamePhase.DISCOVERY: {
            "send_message",
            "ask_about_condition",
            "request_disclosure",
            "disclose_defect",
            "order_inspection",
            "get_market_data",
            "make_offer",
            "walk_away",
            "view_listing",
            "view_property_details",
            "update_listing",
            "wait",
        },
        GamePhase.NEGOTIATION: {
            "send_message",
            "disclose_defect",
            "respond_to_offer",
            "withdraw_offer",
            "walk_away",
            "view_property_details",
            "wait",
        },
        GamePhase.CLOSED: set(),  # No actions allowed after close
        GamePhase.TERMINATED: set(),  # No actions allowed after termination
    }

    def __init__(
        self,
        seller: "ReActAgent",
        buyer: "ReActAgent",
        db: Database,
        config: dict[str, Any],
        property_data: dict[str, Any],
        defects: list[Any],
    ) -> None:
        """Initialize the game controller.

        Args:
            seller: Seller agent instance
            buyer: Buyer agent instance
            db: Database instance for persistence
            config: Game configuration (max_turns, timeout_seconds, etc.)
            property_data: Full property data (public + hidden features)
            defects: List of property defects
        """
        self.seller = seller
        self.buyer = buyer
        self.db = db
        self.config = config
        self.property_data = property_data
        self.defects = defects

        # Initialize game state (will be set in initialize())
        self.state: GameState | None = None

        # Shared context for tool execution
        # This tracks buyer questions, formal requests, etc. for disclosure context
        self.context: dict[str, Any] = {
            "property_data": property_data,
            "property_defects": defects,
            "buyer_questions": {},  # system name -> turn when asked
            "formal_request_turn": None,  # turn when request_disclosure was called
            "buyer_budget": config.get("buyer_budget", 200000),
        }

    def initialize(self) -> str:
        """Create a new game and initialize state.

        Returns:
            Game ID for the newly created game
        """
        # Create game in database
        game_id = self.db.create_game(self.config)

        # Initialize game state
        self.state = GameState(game_id=game_id)

        # Save initial state
        self.db.save_state(game_id, self.state)

        # Add game_id to context
        self.context["game_id"] = game_id

        return game_id

    def run(self) -> TransactionOutcome:
        """Run the game loop until completion or timeout.

        Alternates turns between seller and buyer, executing actions and
        updating state until the game reaches CLOSED or TERMINATED phase,
        or the maximum turn count is reached.

        Returns:
            TransactionOutcome describing the final result
        """
        if self.state is None:
            raise RuntimeError("Game not initialized. Call initialize() first.")

        max_turns = self.config.get("max_turns", 50)

        while self.state.turn_number < max_turns:
            # Check if game has ended
            if self.state.phase in [GamePhase.CLOSED, GamePhase.TERMINATED]:
                break

            # Alternate turns: seller (even turns) → buyer (odd turns)
            # Turn 0: seller lists property
            # Turn 1: buyer views listing
            # Turn 2: seller responds
            # etc.
            agent_id = "seller" if self.state.turn_number % 2 == 0 else "buyer"

            # Execute turn
            success = self.execute_turn(agent_id)
            if not success:
                # Agent error or timeout → terminate game
                self.state.outcome = TransactionOutcome(completed=False, final_price=0)
                self.state.phase = GamePhase.TERMINATED
                self.db.save_state(self.state.game_id, self.state)
                break

            # Increment turn counter
            self.state.turn_number += 1

            # Save state after each turn
            self.db.save_state(self.state.game_id, self.state)

        # If we hit max turns without completion, mark as terminated
        if self.state.turn_number >= max_turns and self.state.outcome is None:
            self.state.outcome = TransactionOutcome(completed=False, final_price=0)
            self.state.phase = GamePhase.TERMINATED
            self.db.save_state(self.state.game_id, self.state)

        # Update game status in database
        final_status = "completed" if self.state.phase == GamePhase.CLOSED else "terminated"
        self.db.update_game_status(self.state.game_id, final_status)

        # Return outcome (create default if None)
        return self.state.outcome or TransactionOutcome(completed=False, final_price=0)

    def execute_turn(self, agent_id: str) -> bool:
        """Execute a single agent turn.

        Steps:
        1. Get the agent (seller or buyer)
        2. Start timer for decision time
        3. Call agent.act() to get action
        4. Stop timer
        5. Validate action against current phase
        6. Execute tool via registry
        7. Handle phase transitions
        8. Log to transcript
        9. Snapshot agent memory

        Args:
            agent_id: "seller" or "buyer"

        Returns:
            True if turn executed successfully, False on error/timeout
        """
        if self.state is None:
            raise RuntimeError("Game not initialized")

        # Get agent and tool registry
        if agent_id == "seller":
            agent = self.seller
        elif agent_id == "buyer":
            agent = self.buyer
        else:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        # Timeout handling
        timeout_seconds = self.config.get("timeout_seconds", 60)
        start_time = time.time()

        try:
            # Call agent to get action (with timeout check in a real async implementation)
            tool_name, parameters, reasoning_trace = agent.act(self.state, self.context)

            # Check if we exceeded timeout
            elapsed_ms = int((time.time() - start_time) * 1000)
            if elapsed_ms / 1000 > timeout_seconds:
                print(f"Turn {self.state.turn_number} ({agent_id}): Timeout after {elapsed_ms}ms")
                return False

            # Validate action against current phase
            if not self.validate_action(agent_id, tool_name, parameters):
                print(
                    f"Turn {self.state.turn_number} ({agent_id}): Invalid action '{tool_name}' in phase {self.state.phase}"
                )
                return False

            # Add state to context for tool execution
            self.context["state"] = self.state
            self.context["agent_id"] = agent_id

            # Execute tool
            result = agent.tools.execute(tool_name, parameters, self.context)

            # Check if tool execution indicated an error
            if not result.get("success", True):
                print(
                    f"Turn {self.state.turn_number} ({agent_id}): Tool execution failed: {result}"
                )
                # Don't return False - some tools might return success=False for valid reasons
                # Continue and log the result

            # Handle phase transitions
            if tool_name in self.PHASE_TRANSITIONS:
                new_phase = self.PHASE_TRANSITIONS[tool_name]
                self.state.transition_phase(new_phase)

            # Special case: respond_to_offer with "accept" closes the game
            if tool_name == "respond_to_offer" and parameters.get("action") == "accept":
                self.state.transition_phase(GamePhase.CLOSED)
                # Outcome is set by the tool itself

            # Log action to transcript
            self.log_action(
                agent_id=agent_id,
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                reasoning_trace=reasoning_trace,
                time_ms=elapsed_ms,
            )

            # Snapshot agent memory
            memory_content = agent.memory.read()
            self.db.save_memory_snapshot(
                self.state.game_id, agent_id, self.state.turn_number, memory_content
            )

            return True

        except Exception as e:
            # Log error and terminate turn
            elapsed_ms = int((time.time() - start_time) * 1000)
            print(f"Turn {self.state.turn_number} ({agent_id}): Error during execution: {e}")

            # Log error to transcript
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
        """Validate that an action is legal in the current phase.

        Checks:
        1. Tool is allowed in current phase
        2. Agent role can use this tool (e.g., only seller can list_property)
        3. State preconditions are met (e.g., can't respond_to_offer without an offer)

        Args:
            agent_id: Agent attempting the action
            tool_name: Name of the tool being called
            params: Tool parameters (for additional validation)

        Returns:
            True if action is valid, False otherwise
        """
        if self.state is None:
            return False

        # Check phase allows this action
        allowed_actions = self.PHASE_ALLOWED_ACTIONS.get(self.state.phase, set())
        if tool_name not in allowed_actions:
            return False

        # Check agent role can use this tool
        seller_only_tools = {
            "list_property",
            "update_listing",
            "disclose_defect",
            "view_property_details",
            "respond_to_offer",
        }
        buyer_only_tools = {
            "view_listing",
            "ask_about_condition",
            "request_disclosure",
            "order_inspection",
            "get_market_data",
            "make_offer",
            "withdraw_offer",
            "walk_away",
        }

        if agent_id == "seller" and tool_name in buyer_only_tools:
            return False
        if agent_id == "buyer" and tool_name in seller_only_tools:
            return False

        # Check state preconditions
        # Example: can't respond_to_offer without a current offer
        if tool_name == "respond_to_offer" and self.state.current_offer is None:
            return False

        # Example: can't list_property twice
        if tool_name == "list_property" and self.state.listing is not None:
            return False

        # Example: can't view_listing before it's listed
        if tool_name == "view_listing" and self.state.listing is None:
            return False

        return True

    def transition_phase(self, new_phase: GamePhase) -> None:
        """Transition to a new game phase.

        Args:
            new_phase: The phase to transition to
        """
        if self.state is None:
            raise RuntimeError("Game not initialized")

        self.state.transition_phase(new_phase)

    def log_action(
        self,
        agent_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
        reasoning_trace: str,
        time_ms: int,
    ) -> None:
        """Log an action to the game transcript.

        Args:
            agent_id: Agent who took the action
            tool_name: Name of the tool that was called
            parameters: Tool parameters
            result: Tool execution result
            reasoning_trace: Agent's reasoning before action
            time_ms: Time taken to decide (milliseconds)
        """
        if self.state is None:
            raise RuntimeError("Game not initialized")

        # Read current memory state for snapshot
        if agent_id == "seller":
            memory_snapshot = self.seller.memory.read()
        elif agent_id == "buyer":
            memory_snapshot = self.buyer.memory.read()
        else:
            memory_snapshot = ""

        # Create transcript entry
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

        # Append to database
        self.db.append_transcript(entry)
