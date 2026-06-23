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
    """Abstract base class for agents using the Reason + Act framework."""

    def __init__(
        self,
        agent_id: str,
        model: ModelInterface,
        memory: Memory,
        tools: ToolRegistry,
        system_prompt: str,
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.memory = memory
        self.tools = tools
        self.system_prompt = system_prompt

    @abstractmethod
    def observe(self, state: GameState) -> str:
        """Convert the current game state into a natural language observation."""
        ...

    def act(
        self, state: GameState, context: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str]:
        """Execute one observe-think-act cycle.

        Returns:
            Tuple of (tool_name, parameters, reasoning_trace).
        """
        observation = self.observe(state)
        messages = self._build_messages(observation)

        registry_tools = self.tools.get_all_definitions()
        tool_definitions = [
            ModelToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in registry_tools
        ]

        response = self.model.generate(
            messages=messages,
            tools=tool_definitions,
            temperature=0.7,
        )

        reasoning_trace = response.content

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            return (tool_call.name, tool_call.arguments, reasoning_trace)
        elif response.content:
            return ("pass", {}, reasoning_trace)
        else:
            raise ValueError("Model returned neither text nor tool calls")

    def _build_messages(self, observation: str) -> list[Message]:
        """Build the message list for model input."""
        memory_content = self.memory.read()

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
