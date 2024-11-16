import anthropic
import json
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096

# Links:
# Tool calling docs - https://docs.anthropic.com/en/docs/build-with-claude/tool-use


class AnthropicProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Anthropic provider with the given configuration.
        Pass the entire configuration dictionary to the Anthropic client constructor.
        """

        self.client = anthropic.Anthropic(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        # Check if the fist message is a system message
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = []

        # kwargs.setdefault('max_tokens', DEFAULT_MAX_TOKENS)
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        # Handle tool calls. Convert from OpenAI tool calls to Anthropic tool calls.
        if "tools" in kwargs:
            kwargs["tools"] = convert_openai_tools_to_anthropic(kwargs["tools"])

        # Convert tool results from OpenAI format to Anthropic format
        converted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg["role"] == "tool":
                    # Convert tool result message
                    converted_msg = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg["tool_call_id"],
                                "content": msg["content"],
                            }
                        ],
                    }
                    converted_messages.append(converted_msg)
                elif msg["role"] == "assistant" and "tool_calls" in msg:
                    # Handle assistant messages with tool calls
                    content = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    for tool_call in msg["tool_calls"]:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "input": json.loads(tool_call["function"]["arguments"]),
                            }
                        )
                    converted_messages.append({"role": "assistant", "content": content})
                else:
                    # Keep other messages as is
                    converted_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )
            else:
                # Handle Message objects
                if msg.role == "tool":
                    converted_msg = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                    converted_messages.append(converted_msg)
                elif msg.role == "assistant" and msg.tool_calls:
                    # Handle Message objects with tool calls
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tool_call in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": json.loads(tool_call.function.arguments),
                            }
                        )
                    converted_messages.append({"role": "assistant", "content": content})
                else:
                    converted_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )

        print(converted_messages)
        response = self.client.messages.create(
            model=model, system=system_message, messages=converted_messages, **kwargs
        )
        print(response)
        return self.normalize_response(response)

    def normalize_response(self, response):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()

        # Map Anthropic stop_reason to OpenAI finish_reason
        finish_reason_mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            # Add more mappings as needed
        }
        normalized_response.choices[0].finish_reason = finish_reason_mapping.get(
            response.stop_reason, "stop"
        )

        # Add usage information
        normalized_response.usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        # Check if the response contains tool usage
        if response.stop_reason == "tool_use":
            # Find the tool_use content
            tool_call = next(
                (content for content in response.content if content.type == "tool_use"),
                None,
            )

            if tool_call:
                function = Function(
                    name=tool_call.name, arguments=json.dumps(tool_call.input)
                )
                tool_call_obj = ChatCompletionMessageToolCall(
                    id=tool_call.id, function=function, type="function"
                )
                # Get the text content if any
                text_content = next(
                    (
                        content.text
                        for content in response.content
                        if content.type == "text"
                    ),
                    "",
                )

                message = Message(
                    content=text_content or None,
                    tool_calls=[tool_call_obj] if tool_call else None,
                    role="assistant",
                    refusal=None,
                )
                normalized_response.choices[0].message = message
                return normalized_response

        # Handle regular text response
        message = Message(
            content=response.content[0].text,
            role="assistant",
            tool_calls=None,
            refusal=None,
        )
        normalized_response.choices[0].message = message
        return normalized_response


def convert_openai_tools_to_anthropic(openai_tools):
    anthropic_tools = []

    for tool in openai_tools:
        # Only handle function-type tools from OpenAI
        if tool.get("type") != "function":
            continue

        function = tool["function"]

        anthropic_tool = {
            "name": function["name"],
            "description": function["description"],
            "input_schema": {
                "type": "object",
                "properties": function["parameters"]["properties"],
                "required": function["parameters"].get("required", []),
            },
        }

        anthropic_tools.append(anthropic_tool)

    return anthropic_tools
