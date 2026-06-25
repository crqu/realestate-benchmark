"""Gemini model implementation via Google Cloud Vertex AI.

Uses the google-genai SDK to access Gemini models hosted on Vertex AI.
"""

import logging

from google import genai
from google.genai import types

from realestate_benchmark.models.interface import (
    Message,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class GeminiModel:
    """Vertex AI implementation using Gemini models via google-genai SDK.

    Example:
        >>> model = GeminiModel(
        ...     project_id="your-project-id",
        ...     region="us-east5",
        ...     model="gemini-2.5-pro",
        ... )
        >>> response = model.generate([
        ...     Message(role="user", content="Hello!")
        ... ])
    """

    def __init__(
        self,
        project_id: str,
        region: str = "us-east5",
        model: str = "gemini-2.5-pro",
        json_mode: bool = False,
        thinking_budget: int | None = None,
    ):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=region,
        )
        self._model = model
        self._json_mode = json_mode
        self._thinking_budget = thinking_budget

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_tool_use(self) -> bool:
        return True

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[types.Content]]:
        system_instruction = None
        contents: list[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                if system_instruction is not None:
                    system_instruction += "\n\n" + msg.content
                else:
                    system_instruction = msg.content
            elif msg.role in ("user", "assistant"):
                role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    types.Content(
                        parts=[types.Part(text=msg.content)],
                        role=role,
                    )
                )
            else:
                raise ValueError(f"Invalid message role: {msg.role}")

        if contents and contents[0].role != "user":
            raise ValueError("First message must be from user")

        return system_instruction, contents

    def _convert_tools(self, tools: list[ToolDefinition]) -> types.Tool:
        declarations = []
        for tool in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
        return types.Tool(function_declarations=declarations)

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        try:
            system_instruction, contents = self._convert_messages(messages)

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=16384,
            )

            if system_instruction:
                config.system_instruction = system_instruction

            if self._json_mode and not tools:
                config.response_mime_type = "application/json"

            if self._thinking_budget is not None:
                config.thinking_config = types.ThinkingConfig(
                    thinking_budget=self._thinking_budget
                )

            if tools:
                config.tools = [self._convert_tools(tools)]

            response = self.client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            content = response.text or ""

            tool_calls = []
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        tool_calls.append(
                            ToolCall(
                                name=part.function_call.name,
                                arguments=dict(part.function_call.args)
                                if part.function_call.args
                                else {},
                            )
                        )

            usage = {"input_tokens": 0, "output_tokens": 0}
            if response.usage_metadata:
                usage["input_tokens"] = response.usage_metadata.prompt_token_count or 0
                usage["output_tokens"] = (
                    response.usage_metadata.candidates_token_count or 0
                )
                if response.usage_metadata.thoughts_token_count:
                    logger.debug(
                        "Thinking tokens: %d",
                        response.usage_metadata.thoughts_token_count,
                    )

            return ModelResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                model_name=self._model,
            )

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e
