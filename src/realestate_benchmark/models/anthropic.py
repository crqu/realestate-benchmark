"""Anthropic API implementation of the ModelInterface.

This module provides an implementation of the ModelInterface using Anthropic's
Claude models via the official Python SDK.

Example:
    Basic usage:

    >>> from realestate_benchmark.models.anthropic import AnthropicModel
    >>> from realestate_benchmark.models.interface import Message
    >>>
    >>> model = AnthropicModel(api_key="your-api-key")
    >>> messages = [
    ...     Message(role="system", content="You are a helpful assistant."),
    ...     Message(role="user", content="What is 2+2?"),
    ... ]
    >>> response = model.generate(messages)
    >>> print(response.content)
"""

from typing import Any, Literal, cast

import anthropic
from anthropic.types import MessageParam, ToolParam

from realestate_benchmark.models.interface import (
    Message,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


class AnthropicModel:
    """Anthropic Claude model implementation.

    This class wraps the Anthropic API to conform to the ModelInterface protocol.
    It handles message format conversion, tool definitions, and response parsing.

    Attributes:
        client: The Anthropic API client instance.
        _model_name: The specific Claude model identifier to use.

    Args:
        api_key: Anthropic API key. If not provided, will use the ANTHROPIC_API_KEY
            environment variable.
        model: The Claude model identifier. Defaults to "claude-sonnet-4-5@20250929".
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5@20250929",
    ) -> None:
        """Initialize the Anthropic model client.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Claude model identifier. Defaults to "claude-sonnet-4-5@20250929".
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self._model_name = model

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """Generate a response using the Anthropic API.

        Converts internal message format to Anthropic's format, handles tool
        definitions, makes the API call, and parses the response back to our
        internal format.

        Args:
            messages: Conversation history including system, user, and assistant messages.
            tools: Optional list of tool definitions the model can use.
            temperature: Sampling temperature (0.0 to 1.0). Default is 0.7.

        Returns:
            ModelResponse with generated content, tool calls, usage stats, and model name.

        Raises:
            anthropic.APIError: For API-level errors (auth, rate limits, etc.).
            anthropic.APIConnectionError: For network-related errors.
            ValueError: If messages format is invalid or conversion fails.
        """
        try:
            # Convert messages to Anthropic format
            system_content, anthropic_messages = self._convert_messages(messages)

            # Build API call parameters
            api_params: dict[str, Any] = {
                "model": self._model_name,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": anthropic_messages,
            }

            # Add system message if present
            if system_content:
                api_params["system"] = system_content

            # Add tools if provided
            if tools:
                api_params["tools"] = self._convert_tools(tools)

            # Make API call
            response = self.client.messages.create(**api_params)

            # Parse response
            return self._parse_response(response)

        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"Anthropic API connection error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during model generation: {e}") from e

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[MessageParam]]:
        """Convert internal Message format to Anthropic's format.

        Anthropic separates system messages from the conversation. System messages
        go in a separate parameter, while user/assistant messages form the conversation.

        Args:
            messages: List of internal Message objects.

        Returns:
            Tuple of (system_content, anthropic_messages):
                - system_content: Combined system message text, or empty string if none.
                - anthropic_messages: List of user/assistant messages in Anthropic format.

        Raises:
            ValueError: If message roles are invalid or conversation structure is wrong.
        """
        system_messages: list[str] = []
        anthropic_messages: list[MessageParam] = []

        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg.content)
            elif msg.role in ("user", "assistant"):
                # Cast to Literal type for TypedDict compatibility
                role = cast(Literal["user", "assistant"], msg.role)
                anthropic_messages.append({"role": role, "content": msg.content})
            else:
                raise ValueError(f"Invalid message role: {msg.role}")

        # Combine all system messages into one
        system_content = "\n\n".join(system_messages)

        # Validate conversation structure (must start with user)
        if anthropic_messages and anthropic_messages[0]["role"] != "user":
            raise ValueError("Anthropic API requires first message to be from user role")

        return system_content, anthropic_messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[ToolParam]:
        """Convert internal ToolDefinition format to Anthropic's format.

        Args:
            tools: List of internal ToolDefinition objects.

        Returns:
            List of tool definitions in Anthropic's expected format.
        """
        anthropic_tools: list[ToolParam] = []

        for tool in tools:
            anthropic_tool: ToolParam = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def _parse_response(self, response: anthropic.types.Message) -> ModelResponse:
        """Parse Anthropic API response into internal ModelResponse format.

        Args:
            response: Raw response from Anthropic API.

        Returns:
            Parsed ModelResponse with content, tool calls, and usage stats.
        """
        # Extract text content
        text_content = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=block.input,
                    )
                )

        # Extract usage statistics
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return ModelResponse(
            content=text_content,
            tool_calls=tool_calls,
            usage=usage,
            model_name=response.model,
        )

    @property
    def model_name(self) -> str:
        """Return the model identifier.

        Returns:
            String identifier for the Claude model being used.
        """
        return self._model_name

    @property
    def supports_tool_use(self) -> bool:
        """Return whether this model supports tool calling.

        All Claude models support tool use, so this always returns True.

        Returns:
            True, as all Claude models support tool calling.
        """
        return True
