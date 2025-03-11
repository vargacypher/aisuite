"""The interface to Google's Vertex AI."""

import os, json
import uuid

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    GenerationResponse,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)

from domain.models.framework import ProviderInterface, ChatCompletionResponse
from domain.models.framework.message import (
    ChatCompletionMessageToolCall,
    Message,
    Function,
)
from typing import Any

DEFAULT_TEMPERATURE = 0.7


class GoogleProvider(ProviderInterface):
    """Implements the ProviderInterface for interacting with Google's Vertex AI."""

    def __init__(self, **config):
        """Set up the Google AI client with a project ID."""
        self.project_id = config.get("project_id") or os.getenv("GOOGLE_PROJECT_ID")
        self.location = config.get("region") or os.getenv("GOOGLE_REGION")
        # self.app_creds_path = config.get("application_credentials") or os.getenv(
        #     "GOOGLE_APPLICATION_CREDENTIALS"
        # ) #TODO TEST IF IT`ll work inside GCP without this PRODUT

        if not self.project_id or not self.location:
            raise EnvironmentError(
                "Missing one or more required Google environment variables: "
                "GOOGLE_PROJECT_ID, GOOGLE_REGION. "
                "Please refer to the setup guide: /guides/google.md."
            )

        vertexai.init(project=self.project_id, location=self.location)

    def chat_completions_create(self, model, messages, **kwargs):
        """Request chat completions from the Google AI API.

        Args:
        ----
            model (str): Identifies the specific provider/model to use.
            messages (list of dict): A list of message objects in chat history.
            kwargs (dict): Optional arguments for the Google AI API.

        Returns:
        -------
            The ChatCompletionResponse with the completion result.

        """

        # Set the temperature if provided, otherwise use the default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)

        tools = self._convert_tool_spec(kwargs.get("tools", None))

        # Transform the roles in the messages
        transformed_messages = self.transform_roles(messages)

        # Convert the messages to the format expected Google
        final_message_history = self.convert_openai_to_vertex_ai(transformed_messages)

        model_kwargs = {
            "model_name": model,
            "generation_config": GenerationConfig(temperature=temperature),
        }

        if tools:
            model_kwargs["tools"] = [tools]

        model = GenerativeModel(**model_kwargs)

        # Start a chat with the GenerativeModel and send the last message
        response = model.generate_content(final_message_history)

        # Convert the response to the format expected by the OpenAI API
        return self.normalize_response(response)

    def convert_openai_to_vertex_ai(self, messages):
        """Convert OpenAI messages to Google AI messages."""
        from vertexai.generative_models import Content, Part

        function_calls = {}
        history = []

        for message in messages:
            tool_calls = message.get("tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    function_calls[function_name] = Part.from_dict(
                        {
                            "function_call": {
                                "name": function_name,
                                "args": function_args,
                            }
                        }
                    )
                continue

            if message["role"] == "tool":
                history.append(
                    Content(role="model", parts=[function_calls.get(message["name"])])
                )
                parts = [
                    Part.from_function_response(
                        name=message["name"],
                        response=json.loads(message["content"]),
                    )
                ]
            else:
                parts = [Part.from_text(message["content"])]

            role = message["role"]
            history.append(Content(role=role, parts=parts))

        return history

    def transform_roles(self, messages):
        """Transform the roles in the messages based on the provided transformations."""
        openai_roles_to_google_roles = {
            "system": "user",  # TODO test remove this or change to model
            "user": "user",
            "assistant": "model",
            "tool": "tool",
        }
        transformed_messages = []

        for message in messages:
            if isinstance(message, Message):
                message = message.model_dump()
            if role := openai_roles_to_google_roles.get(message["role"], None):
                message["role"] = role
                transformed_messages.append(message)
        return transformed_messages

    def normalize_response(self, response: GenerationResponse):
        """Normalize the response from Google AI to match OpenAI's response format."""
        openai_response = ChatCompletionResponse()

        # Extract the first candidate
        candidate = response.candidates[0]

        # Check if the candidate contains a function call
        function_calls = self._extract_tool_calls(response)
        if function_calls:
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=str(uuid.uuid4()),  # Generate a unique ID
                    function=Function(
                        name=fc["function"]["name"],
                        arguments=json.dumps(fc["function"]["arguments"]),
                    ),
                    type="function",
                )
                for fc in function_calls
            ]
            openai_response.choices[0].finish_reason = "tool_calls"
            openai_response.choices[0].message.role = "assistant"
            openai_response.choices[0].message.tool_calls = tool_calls

        try:
            openai_response.choices[0].message.content = candidate.content.parts[0].text
        except AttributeError:
            openai_response.choices[0].message.content = ""

        return openai_response

    def _extract_tool_calls(self, response: GenerationResponse) -> list[dict]:
        """
        Extracts tool calls from a GenerationResponse object.

        Args:
            response (GenerationResponse): The response object containing candidates with tool calls.

        Returns:
            list[dict]: A list of dictionaries, each representing a tool call with its function name and arguments.
        """
        toll_calls: list[dict] = []
        if response.candidates[0].function_calls:
            for function_call in response.candidates[0].function_calls:
                function_call_dict: dict[str, dict[str, Any]] = {
                    "function": {"arguments": {}}
                }
                function_call_dict["function"]["name"] = function_call.name
                for key, value in function_call.args.items():
                    function_call_dict["function"]["arguments"].update({key: value})
                toll_calls.append(function_call_dict)
        return toll_calls

    def _convert_tool_spec(self, tools):
        """Prepare the tool specification for the Google AI API."""
        if not tools:
            return None
        tool_spec = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name=function["function"]["name"],
                    parameters=(
                        function["function"].get(
                            "parameters", {"type": "object"}
                        )  # If no params are provided, default to an empty object, Need this because the API verifies the parameters type
                    ),
                    description=function["function"]["description"],
                )
                for function in tools
            ],
        )
        return tool_spec
