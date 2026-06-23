"""ReAct (Reason + Act) agent base class.

This module implements the core observe-think-act cycle that all agents extend.
Agents use the ReAct framework to make decisions based on observations of the
game state, reasoning about their options, and selecting actions (tool calls).
"""

from abc import ABC, abstractmethod
from typing import Any

from realestate_benchmark.agents.memory import Memory
from realestate_benchmark.environment.state import GameState
from realestate_benchmark.models.interface import (
    Message,
    ModelInterface,
)
from realestate_benchmark.models.interface import (
    ToolDefinition as ModelToolDefinition,
)
from realestate_benchmark.tools.registry import ToolRegistry


class ReActAgent(ABC):
    """Abstract base class for agents using the Reason + Act framework.

    The ReAct framework implements a observe-think-act cycle:
    1. OBSERVE: Format the current game state into a natural language observation
    2. THINK: Use the LLM to reason about the observation and available actions
    3. ACT: Select and execute a tool based on the model's response

    Subclasses must implement the observe() method to define how they perceive
    the game state. The act() method implements the full cycle and is called
    by the game controller on each turn.

    Attributes:
        agent_id: Unique identifier for this agent (e.g., "seller", "buyer")
        model: LLM model interface for generating decisions
        memory: Memory system for maintaining agent state across turns
        tools: Registry of available tools/actions
        system_prompt: Role-specific instructions for the agent
    """

    def __init__(
        self,
        agent_id: str,
        model: ModelInterface,
        memory: Memory,
        tools: ToolRegistry,
        system_prompt: str,
    ) -> None:
        """Initialize the ReAct agent.

        Args:
            agent_id: Unique identifier for this agent
            model: LLM model implementation
            memory: Memory system for this agent
            tools: Registry of available tools
            system_prompt: System-level instructions defining the agent's role
        """
        self.agent_id = agent_id
        self.model = model
        self.memory = memory
        self.tools = tools
        self.system_prompt = system_prompt

    @abstractmethod
    def observe(self, state: GameState) -> str:
        """Convert the current game state into a natural language observation.

        This method defines how the agent perceives the world. Different agent
        types see different information (e.g., sellers see hidden property
        details, buyers only see public information).

        Args:
            state: Current game state

        Returns:
            Natural language description of what the agent observes
        """
        ...

    def act(
        self, state: GameState, context: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str]:
        """Execute one observe-think-act cycle.

        This is the main decision-making method called by the game controller.
        It implements the full ReAct cycle:
        1. OBSERVE: Call self.observe() to get state description
        2. THINK: Build messages and query model with available tools
        3. ACT: Parse model response for tool call or text

        Args:
            state: Current game state
            context: Additional context (property data, budget, etc.)

        Returns:
            Tuple of (tool_name, parameters, reasoning_trace):
                - tool_name: Name of the tool to execute
                - parameters: Dictionary of parameters for the tool
                - reasoning_trace: The model's reasoning before selecting the action

        Raises:
            ValueError: If model returns neither text nor tool calls
        """
        # OBSERVE: Get current state description
        observation = self.observe(state)

        # THINK: Build messages and query model
        messages = self._build_messages(observation)

        # Convert tool definitions to model interface format, excluding blocked tools
        blocked = self._blocked_tools(context)
        registry_tools = self.tools.get_all_definitions()
        tool_definitions = [
            ModelToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in registry_tools
            if tool.name not in blocked
        ]

        response = self.model.generate(
            messages=messages,
            tools=tool_definitions,
            temperature=0.7,
        )

        # Extract reasoning trace (the model's text response before tool call)
        reasoning_trace = response.content

        # ACT: Parse response for action
        if response.tool_calls:
            # Model requested a tool call
            tool_call = response.tool_calls[0]  # Take first tool call
            return (tool_call.name, tool_call.arguments, reasoning_trace)
        elif response.content:
            return ("wait", {}, reasoning_trace)
        else:
            raise ValueError("Model returned neither text nor tool calls")

    def _blocked_tools(self, context: dict[str, Any]) -> set[str]:
        """Return tool names that should be hidden from the model this turn.

        Subclasses can override for role-specific blocking. The base
        implementation removes tools that have exhausted their cooldowns.
        """
        blocked: set[str] = set()
        if context.get("_market_data_calls", 0) >= 1:
            blocked.add("get_market_data")
        if context.get("_listing_viewed"):
            blocked.add("view_listing")
        return blocked

    def _build_messages(self, observation: str) -> list[Message]:
        """Build the message list for model input.

        Constructs the conversation context including:
        - System prompt (role definition)
        - User message with current observation, memory, and action prompt

        Args:
            observation: Current state observation from observe()

        Returns:
            List of messages for the model
        """
        # Read current memory state
        memory_content = self.memory.read()

        # Build user message with observation, memory, and prompt
        user_content = f"""# Current Situation
{observation}

# Your Memory
{memory_content}

What do you do next? Use one of your available tools to take an action.
"""

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_content),
        ]

        return messages
