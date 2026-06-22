"""Vertex AI model implementation using Claude via Google Cloud.

This module provides a ModelInterface implementation for Claude models
running on Google Cloud Vertex AI.
"""

from typing import Any

from anthropic import AnthropicVertex

from realestate_benchmark.models.interface import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


class VertexModel(ModelInterface):
    """Vertex AI implementation of ModelInterface using Claude models.

    This implementation uses the Anthropic Vertex AI SDK to access Claude
    models hosted on Google Cloud Platform.

    Example:
        >>> model = VertexModel(
        ...     project_id="your-project-id",
        ...     region="us-east5",
        ...     model="claude-sonnet-4-5"
        ... )
        >>> response = model.generate([
        ...     Message(role="user", content="Hello!")
        ... ])
    """

    def __init__(
        self,
        project_id: str,
        region: str = "us-east5",
        model: str = "claude-sonnet-4-5@20250929",
    ):
        """Initialize Vertex AI model client.

        Args:
            project_id: Google Cloud project ID
            region: GCP region where Claude is available
            model: Model identifier (e.g., "claude-sonnet-4-5@20250929")
        """
        self.client = AnthropicVertex(
            project_id=project_id,
            region=region,
        )
        self._model = model

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

    @property
    def supports_tool_use(self) -> bool:
        """Return whether this model supports tool calling."""
        return True

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal Message format to Anthropic format.

        Args:
            messages: List of Message objects

        Returns:
            Tuple of (system_message, anthropic_messages)

        Raises:
            ValueError: If message format is invalid
        """
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                if system_message is not None:
                    raise ValueError("Only one system message is allowed")
                system_message = msg.content
            elif msg.role in ("user", "assistant"):
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
            else:
                raise ValueError(f"Invalid message role: {msg.role}")

        # Anthropic API requires first message to be from user
        if anthropic_messages and anthropic_messages[0]["role"] != "user":
            raise ValueError("First message must be from user")

        return system_message, anthropic_messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert internal ToolDefinition format to Anthropic format.

        Args:
            tools: List of ToolDefinition objects

        Returns:
            List of tool dictionaries in Anthropic format
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """Generate a response from the Vertex AI model.

        Args:
            messages: Conversation history
            tools: Available tools for the model to call
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            ModelResponse with content, tool calls, and usage

        Raises:
            Exception: If API call fails
        """
        try:
            # Convert messages
            system_message, anthropic_messages = self._convert_messages(messages)

            # Build API call parameters
            params: dict[str, Any] = {
                "model": self._model,
                "messages": anthropic_messages,
                "max_tokens": 4096,
                "temperature": temperature,
            }

            if system_message:
                params["system"] = system_message

            if tools:
                params["tools"] = self._convert_tools(tools)

            # Make API call
            response = self.client.messages.create(**params)

            # Extract text content
            text_content = ""
            tool_calls = []

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

            # Extract usage information
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            return ModelResponse(
                content=text_content,
                tool_calls=tool_calls,
                usage=usage,
                model_name=self._model,
            )

        except Exception as e:
            # Re-raise with context
            raise Exception(f"Vertex AI API error: {str(e)}") from e
