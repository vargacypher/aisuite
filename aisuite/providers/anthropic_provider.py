import anthropic
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096


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

        # TODO: Handle tool calls. Convert from OpenAI tool calls to Anthropic tool calls.
        if "tools" in kwargs:
            kwargs["tools"] = convert_openai_tools_to_anthropic(kwargs["tools"])

        response = self.client.messages.create(
            model=model, system=system_message, messages=messages, **kwargs
        )
        print(response)
        return self.normalize_response(response)

    def normalize_response(self, response):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response.content[0].text
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
