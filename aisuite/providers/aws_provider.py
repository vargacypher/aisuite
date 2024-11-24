import os
import json

import boto3
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message

# Implementation notes:
# OpenAI's tool calling format:
# {
#     "role": "assistant",
#     "content": null,
#     "tool_calls": [{
#         "id": "call_abc123",
#         "type": "function",
#         "function": {
#             "name": "top_song",
#             "arguments": "{\"sign\": \"WZPZ\"}"
#         }
#     }]
# }
# AWS Bedrock's tool calling format:
# https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#tool-use
# {
#     "role": "tool",
#     "content": "{\"song\": \"Yesterday\", \"artist\": \"The Beatles\"}",
#     "tool_call_id": "call_abc123"
# }


class AwsProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the AWS Bedrock provider with the given configuration.

        This class uses the AWS Bedrock converse API, which provides a consistent interface
        for all Amazon Bedrock models that support messages. Examples include:
        - anthropic.claude-v2
        - meta.llama3-70b-instruct-v1:0
        - mistral.mixtral-8x7b-instruct-v0:1

        The model value can be a baseModelId for on-demand throughput or a provisionedModelArn
        for higher throughput. To obtain a provisionedModelArn, use the CreateProvisionedModelThroughput API.

        For more information on model IDs, see:
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html

        Note:
        - The Anthropic Bedrock client uses default AWS credential providers, such as
          ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
        - If the region is not set, it defaults to us-west-1, which may lead to a
          "Could not connect to the endpoint URL" error.
        - The client constructor does not accept additional parameters.

        Args:
            **config: Configuration options for the provider.

        """
        self.region_name = config.get(
            "region_name", os.getenv("AWS_REGION_NAME", "us-west-2")
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)
        self.inference_parameters = [
            "maxTokens",
            "temperature",
            "topP",
            "stopSequences",
        ]

    def normalize_response(self, response):
        """Normalize the response from the Bedrock API to match OpenAI's response format."""
        print(f"Response: {response}")
        norm_response = ChatCompletionResponse()
        norm_response.choices[0].message.content = response["output"]["message"][
            "content"
        ][0]["text"]
        return norm_response

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by Anthropic will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        # https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        system_message = []
        if messages[0]["role"] == "system":
            system_message = [{"text": messages[0]["content"]}]
            messages = messages[1:]

        formatted_messages = []
        for message in messages:
            # Convert Message object to dict if necessary
            message_dict = (
                message.model_dump() if hasattr(message, "model_dump") else message
            )

            # QUIETLY Ignore any "system" messages except the first system message.
            if message_dict["role"] != "system":
                if message_dict["role"] == "tool":
                    # Transform tool result message to Bedrock format
                    bedrock_message = self.transform_tool_result_to_bedrock(
                        message_dict
                    )
                    if bedrock_message:
                        formatted_messages.append(bedrock_message)
                elif message_dict["role"] == "assistant":
                    # Convert assistant message to Bedrock format
                    bedrock_message = self.transform_assistant_to_bedrock(message_dict)
                    if bedrock_message:
                        formatted_messages.append(bedrock_message)
                else:
                    formatted_messages.append(
                        {
                            "role": message_dict["role"],
                            "content": [{"text": message_dict["content"]}],
                        }
                    )

        # Maintain a list of Inference Parameters which Bedrock supports.
        # These fields need to be passed using inferenceConfig.
        # Rest all other fields are passed as additionalModelRequestFields.
        inference_config = {}
        additional_model_request_fields = {}

        # Handle tools if present in kwargs
        tool_config = None
        if "tools" in kwargs:
            tool_config = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", " "),
                            "inputSchema": {"json": tool["function"]["parameters"]},
                        }
                    }
                    for tool in kwargs["tools"]
                ]
            }
            print(f"Tool config: {tool_config}")
            print(f"Received tools specification: {kwargs['tools']}")
            # Remove tools from kwargs since we're handling it separately
            del kwargs["tools"]

        # Iterate over the kwargs and separate the inference parameters and additional model request fields.
        for key, value in kwargs.items():
            if key in self.inference_parameters:
                inference_config[key] = value
            else:
                additional_model_request_fields[key] = value

        print(f"Formatted messages for Bedrock: {formatted_messages}")

        # Call the Bedrock Converse API with tool_config
        response = self.client.converse(
            modelId=model,  # baseModelId or provisionedModelArn
            messages=formatted_messages,
            system=system_message,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_request_fields,
            toolConfig=tool_config,
        )

        # Check if the model is requesting tool use
        if response.get("stopReason") == "tool_use":
            print(f"Bedrock response for tool use: {response}")
            tool_message = self.transform_tool_call_to_openai(response)
            if tool_message:
                norm_response = ChatCompletionResponse()
                norm_response.choices[0].message = self.transform_to_message(
                    tool_message
                )
                norm_response.choices[0].finish_reason = "tool_calls"
                return norm_response

        return self.normalize_response(response)

    # Transform framework Message to a format that AWS understands.
    def transform_from_messages(self, message: Message):
        return message.model_dump(mode="json")

    # Transform AWS message to a format that the framework Message understands.
    def transform_to_message(self, message_dict: dict):
        if message_dict.get("content") is None:
            message_dict["content"] = ""
        return Message(**message_dict)

    def transform_tool_call_to_openai(self, response):
        """Transform AWS Bedrock tool call response to OpenAI format."""
        if response.get("stopReason") != "tool_use":
            return None

        tool_calls = []
        for content in response["output"]["message"]["content"]:
            if "toolUse" in content:
                tool = content["toolUse"]
                tool_calls.append(
                    {
                        "type": "function",
                        "id": tool["toolUseId"],
                        "function": {
                            "name": tool["name"],
                            "arguments": json.dumps(tool["input"]),
                        },
                    }
                )

        if not tool_calls:
            return None

        return {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
            "refusal": None,
        }

    def transform_tool_result_to_bedrock(self, message):
        """Transform OpenAI tool result format to AWS Bedrock format."""
        if message["role"] != "tool" or "content" not in message:
            return None

        # Extract tool call ID and result from OpenAI format
        tool_call_id = message.get("tool_call_id")
        if not tool_call_id:
            raise LLMError("Tool result message must include tool_call_id")

        # Try to parse content as JSON, fall back to text if it fails
        try:
            content_json = json.loads(message["content"])
            content = [{"json": content_json}]
        except json.JSONDecodeError:
            content = [{"text": message["content"]}]

        return {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": tool_call_id, "content": content}}
            ],
        }

    def transform_assistant_to_bedrock(self, message):
        """Transform OpenAI assistant format to AWS Bedrock format."""
        if message["role"] != "assistant":
            return None

        content = []

        # Handle regular text content
        if message.get("content"):
            content.append({"text": message["content"]})

        # Handle tool calls if present
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                if tool_call["type"] == "function":
                    try:
                        input_json = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        input_json = tool_call["function"]["arguments"]

                    content.append(
                        {
                            "toolUse": {
                                "toolUseId": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "input": input_json,
                            }
                        }
                    )

        return {"role": "assistant", "content": content} if content else None
