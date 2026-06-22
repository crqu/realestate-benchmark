"""Tests for model implementations."""

from unittest.mock import MagicMock, patch

import pytest

from realestate_benchmark.models import (
    AnthropicModel,
    Message,
    ModelResponse,
    ToolDefinition,
)


class TestAnthropicModel:
    """Tests for the AnthropicModel implementation."""

    def test_initialization(self) -> None:
        """Test that AnthropicModel initializes correctly."""
        model = AnthropicModel(api_key="test-key", model="claude-sonnet-4-5@20250929")
        assert model.model_name == "claude-sonnet-4-5@20250929"
        assert model.supports_tool_use is True

    def test_model_name_property(self) -> None:
        """Test the model_name property."""
        model = AnthropicModel(api_key="test-key", model="custom-model")
        assert model.model_name == "custom-model"

    def test_supports_tool_use_property(self) -> None:
        """Test the supports_tool_use property."""
        model = AnthropicModel(api_key="test-key")
        assert model.supports_tool_use is True

    def test_convert_messages_user_assistant(self) -> None:
        """Test message conversion with user and assistant messages."""
        model = AnthropicModel(api_key="test-key")
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
        ]

        system_content, anthropic_messages = model._convert_messages(messages)

        assert system_content == ""
        assert len(anthropic_messages) == 3
        assert anthropic_messages[0] == {"role": "user", "content": "Hello"}
        assert anthropic_messages[1] == {"role": "assistant", "content": "Hi there"}
        assert anthropic_messages[2] == {"role": "user", "content": "How are you?"}

    def test_convert_messages_with_system(self) -> None:
        """Test message conversion with system messages."""
        model = AnthropicModel(api_key="test-key")
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="system", content="You are concise."),
            Message(role="user", content="Hello"),
        ]

        system_content, anthropic_messages = model._convert_messages(messages)

        assert system_content == "You are helpful.\n\nYou are concise."
        assert len(anthropic_messages) == 1
        assert anthropic_messages[0] == {"role": "user", "content": "Hello"}

    def test_convert_messages_invalid_role(self) -> None:
        """Test that invalid message roles raise ValueError."""
        model = AnthropicModel(api_key="test-key")
        messages = [Message(role="invalid", content="Test")]

        with pytest.raises(ValueError, match="Invalid message role: invalid"):
            model._convert_messages(messages)

    def test_convert_messages_must_start_with_user(self) -> None:
        """Test that conversations must start with user role."""
        model = AnthropicModel(api_key="test-key")
        messages = [Message(role="assistant", content="Hello")]

        with pytest.raises(ValueError, match="first message to be from user role"):
            model._convert_messages(messages)

    def test_convert_tools(self) -> None:
        """Test tool definition conversion."""
        model = AnthropicModel(api_key="test-key")
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            ),
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        ]

        anthropic_tools = model._convert_tools(tools)

        assert len(anthropic_tools) == 2
        assert anthropic_tools[0]["name"] == "get_weather"
        assert anthropic_tools[0]["description"] == "Get current weather"
        assert "properties" in anthropic_tools[0]["input_schema"]

    @patch("anthropic.Anthropic")
    def test_generate_text_only(self, mock_anthropic: MagicMock) -> None:
        """Test generation with text-only response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello, how can I help?")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        mock_response.model = "claude-sonnet-4-5@20250929"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Create model and generate
        model = AnthropicModel(api_key="test-key")
        messages = [Message(role="user", content="Hello")]
        response = model.generate(messages, temperature=0.5)

        # Verify response
        assert isinstance(response, ModelResponse)
        assert response.content == "Hello, how can I help?"
        assert len(response.tool_calls) == 0
        assert response.usage == {"input_tokens": 10, "output_tokens": 20}
        assert response.model_name == "claude-sonnet-4-5@20250929"

        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args.kwargs
        assert call_args["temperature"] == 0.5
        assert call_args["model"] == "claude-sonnet-4-5@20250929"

    @patch("anthropic.Anthropic")
    def test_generate_with_tool_calls(self, mock_anthropic: MagicMock) -> None:
        """Test generation with tool calls in response."""
        # Setup mock response with tool call - use spec to ensure attributes are set correctly
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "get_weather"
        mock_tool_use.input = {"location": "San Francisco"}

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Let me check the weather."

        mock_response = MagicMock()
        mock_response.content = [mock_text, mock_tool_use]
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 25
        mock_response.model = "claude-sonnet-4-5@20250929"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Create model and generate with tools
        model = AnthropicModel(api_key="test-key")
        messages = [Message(role="user", content="What's the weather?")]
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {}},
            )
        ]
        response = model.generate(messages, tools=tools)

        # Verify response
        assert response.content == "Let me check the weather."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "San Francisco"}

        # Verify tools were passed to API
        call_args = mock_client.messages.create.call_args.kwargs
        assert "tools" in call_args
        assert len(call_args["tools"]) == 1

    @patch("anthropic.Anthropic")
    def test_generate_with_system_message(self, mock_anthropic: MagicMock) -> None:
        """Test that system messages are handled correctly."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=10)
        mock_response.model = "claude-sonnet-4-5@20250929"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        model = AnthropicModel(api_key="test-key")
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]
        model.generate(messages)

        # Verify system message was passed separately
        call_args = mock_client.messages.create.call_args.kwargs
        assert call_args["system"] == "You are helpful."
        assert len(call_args["messages"]) == 1

    @patch("anthropic.Anthropic")
    def test_generate_api_error_handling(self, mock_anthropic: MagicMock) -> None:
        """Test that API errors are properly handled and wrapped."""
        import anthropic

        mock_client = MagicMock()
        # Create a proper APIError with required request parameter
        mock_request = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIError(
            "API error", request=mock_request, body=None
        )
        mock_anthropic.return_value = mock_client

        model = AnthropicModel(api_key="test-key")
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(RuntimeError, match="Anthropic API error"):
            model.generate(messages)

    @patch("anthropic.Anthropic")
    def test_generate_connection_error_handling(self, mock_anthropic: MagicMock) -> None:
        """Test that connection errors are properly handled."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            request=MagicMock()
        )
        mock_anthropic.return_value = mock_client

        model = AnthropicModel(api_key="test-key")
        messages = [Message(role="user", content="Hello")]

        # APIConnectionError is a subclass of APIError, so it will match the API error handler
        with pytest.raises(RuntimeError, match="Anthropic API error"):
            model.generate(messages)

    def test_multiple_tool_calls_in_response(self) -> None:
        """Test parsing response with multiple tool calls."""
        model = AnthropicModel(api_key="test-key")

        # Create mock response with multiple tool calls - properly set attributes
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "I'll help with that."

        mock_tool1 = MagicMock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "tool1"
        mock_tool1.input = {"arg1": "value1"}

        mock_tool2 = MagicMock()
        mock_tool2.type = "tool_use"
        mock_tool2.name = "tool2"
        mock_tool2.input = {"arg2": "value2"}

        mock_response = MagicMock()
        mock_response.content = [mock_text, mock_tool1, mock_tool2]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 30
        mock_response.model = "test-model"

        response = model._parse_response(mock_response)

        assert response.content == "I'll help with that."
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "tool1"
        assert response.tool_calls[1].name == "tool2"
