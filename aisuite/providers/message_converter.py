from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall


class OpenAICompliantMessageConverter:
    """
    Base class for message converters that are compatible with OpenAI's API.
    """

    # Class variable that derived classes can override
    tool_results_as_strings = False

    @staticmethod
    def convert_request(messages):
        """Convert messages to OpenAI-compatible format."""
        print("Converting messages to OpenAI-compatible format")
        print(messages)
        transformed_messages = []
        for message in messages:
            tmsg = None
            if isinstance(message, Message):
                message_dict = message.model_dump(mode="json")
                message_dict.pop("refusal", None)  # Remove refusal field if present
                tmsg = message_dict
            else:
                tmsg = message
            # Check if tmsg is a dict, otherwise get role attribute
            role = tmsg["role"] if isinstance(tmsg, dict) else tmsg.role
            if role == "tool":
                if OpenAICompliantMessageConverter.tool_results_as_strings:
                    # Handle both dict and object cases for content
                    if isinstance(tmsg, dict):
                        tmsg["content"] = str(tmsg["content"])
                    else:
                        tmsg.content = str(tmsg.content)

            transformed_messages.append(tmsg)
        print("Transformed messages")
        print(transformed_messages)
        return transformed_messages

    @staticmethod
    def convert_response(response_data) -> ChatCompletionResponse:
        """Normalize the response to match OpenAI's response format."""
        completion_response = ChatCompletionResponse()
        choice = response_data["choices"][0]
        message = choice["message"]

        # Set basic message content
        completion_response.choices[0].message.content = message["content"]
        completion_response.choices[0].message.role = message.get("role", "assistant")

        # Handle tool calls if present
        if "tool_calls" in message and message["tool_calls"] is not None:
            tool_calls = []
            for tool_call in message["tool_calls"]:
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tool_call.get("id"),
                        type="function",  # Always set to "function" as it's the only valid value
                        function=tool_call.get("function"),
                    )
                )
            completion_response.choices[0].message.tool_calls = tool_calls

        return completion_response
