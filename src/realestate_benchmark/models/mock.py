"""Mock model implementation for deterministic testing.

This module provides a MockModel that implements the ModelInterface protocol
without making any external API calls. It returns pre-programmed responses
in sequence and tracks all calls for test assertions.

Example:
    Basic usage with text responses:

    >>> from realestate_benchmark.models.mock import MockModel
    >>> from realestate_benchmark.models.interface import Message
    >>>
    >>> model = MockModel(responses=["Hello!", "How can I help?"])
    >>> messages = [Message(role="user", content="Hi")]
    >>> response = model.generate(messages)
    >>> print(response.content)
    Hello!
    >>> print(len(model.call_history))
    1

    Using tool calls:

    >>> from realestate_benchmark.models.interface import ToolCall
    >>> model = MockModel()
    >>> model.add_response(ToolCall(name="search", arguments={"query": "test"}))
    >>> response = model.generate([Message(role="user", content="Search for test")])
    >>> print(response.tool_calls[0].name)
    search
"""

from collections import deque
from typing import Any

from realestate_benchmark.models.interface import (
    Message,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


class MockModel:
    """A deterministic mock model for testing.

    This model returns pre-programmed responses without making any external API calls.
    It can return both text responses and tool calls, making it suitable for testing
    agent behavior in a controlled, deterministic way.

    Attributes:
        call_history: List of all generate() calls made to this model. Each entry
            contains the messages, tools, and temperature passed to generate().
        responses: Queue of responses to return. Will be popped in FIFO order.
    """

    def __init__(self, responses: list[str | ToolCall] | None = None) -> None:
        """Initialize the mock model.

        Args:
            responses: Optional list of pre-programmed responses. Each response can be:
                - str: Will be returned as text content
                - ToolCall: Will be returned as a tool call
                Responses are consumed in FIFO order (first added, first returned).
        """
        self._model_name = "mock-model-v1"
        self.responses: deque[str | ToolCall] = deque(responses or [])
        self.call_history: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """Generate a response from the pre-programmed queue.

        This method pops the next response from the queue and returns it.
        The call is recorded in call_history for test assertions.

        Args:
            messages: Conversation history (recorded but not used for generation).
            tools: Optional tool definitions (recorded but not used for generation).
            temperature: Sampling temperature (recorded but not used for generation).

        Returns:
            ModelResponse with either text content or a tool call, depending on
            the next queued response.

        Raises:
            IndexError: If the response queue is empty. Tests should ensure
                enough responses are queued for all expected calls.
        """
        # Record this call
        self.call_history.append(
            {
                "messages": [msg.model_dump() for msg in messages],
                "tools": [tool.model_dump() for tool in tools] if tools else None,
                "temperature": temperature,
            }
        )

        # Get next response
        if not self.responses:
            raise IndexError(
                "MockModel response queue is empty. Add more responses with add_response()."
            )

        response = self.responses.popleft()

        # Build ModelResponse based on response type
        if isinstance(response, str):
            # Text response
            return ModelResponse(
                content=response,
                tool_calls=[],
                usage={"input_tokens": 10, "output_tokens": len(response.split())},
                model_name=self._model_name,
            )
        elif isinstance(response, ToolCall):
            # Tool call response
            return ModelResponse(
                content="",  # Tool calls typically have empty content
                tool_calls=[response],
                usage={"input_tokens": 10, "output_tokens": 5},
                model_name=self._model_name,
            )
        else:
            raise TypeError(f"Response must be str or ToolCall, got {type(response).__name__}")

    def add_response(self, response: str | ToolCall) -> None:
        """Add a response to the end of the queue.

        Args:
            response: Either a string for text responses or a ToolCall
                for tool call responses.
        """
        self.responses.append(response)

    def reset(self) -> None:
        """Clear call history and remaining responses.

        This is useful for resetting state between test cases when reusing
        the same MockModel instance.
        """
        self.call_history.clear()
        self.responses.clear()

    @property
    def model_name(self) -> str:
        """Return the model identifier.

        Returns:
            "mock-model-v1"
        """
        return self._model_name

    @property
    def supports_tool_use(self) -> bool:
        """Return whether this model supports tool calling.

        Returns:
            True - the mock model supports both text and tool call responses.
        """
        return True
