import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse

# Define a constant for the default temperature value
DEFAULT_TEMPERATURE = 0.7


class GoogleProvider(Provider):
    """Implements the Provider interface for interacting with Google's Vertex AI."""

    def __init__(self, **config):
        """Set up the Google AI client with a project ID."""
        self.project_id = config.get("project_id") or os.getenv("GOOGLE_PROJECT_ID")
        self.location = config.get("region") or os.getenv("GOOGLE_REGION")
        self.app_creds_path = config.get("application_credentials") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        if not self.project_id or not self.location or not self.app_creds_path:
            raise EnvironmentError(
                "Missing one or more required Google environment variables: "
                "GOOGLE_PROJECT_ID, GOOGLE_REGION, GOOGLE_APPLICATION_CREDENTIALS. "
                "Please refer to the setup guide."
            )

        # Initialize the Vertex AI client
        vertexai.init(project=self.project_id, location=self.location)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Request chat completions from the Google AI API.

        Args:
        ----
            model (str): Identifies the specific provider/model to use.
            messages (list of dict): A list of message objects in chat history.
            kwargs (dict): Optional parameters such as temperature, max_tokens, etc.

        Returns:
        -------
            The ChatCompletionResponse with the completion result.
        """
        # Check if temperature is provided in kwargs, otherwise use default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)

        # Transform OpenAI roles to Google roles
        transformed_messages = self.transform_roles(messages)

        # Prepare the chat history excluding the last message
        final_message_history = self.convert_openai_to_vertex_ai(
            transformed_messages[:-1]
        )
        last_message = transformed_messages[-1]["content"]

        # Initialize the model with generation configuration
        model = GenerativeModel(
            model=model,
            generation_config=GenerationConfig(temperature=temperature),
        )

        # Start the chat session and send the last message
        chat = model.start_chat(history=final_message_history)
        response = chat.send_message(last_message)

        # Convert the Google AI response to OpenAI format
        return self.convert_response_to_openai_format(response)

    def convert_openai_to_vertex_ai(self, messages):
        """Convert OpenAI messages to Google AI messages."""
        history = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            parts = [Part.from_text(content)]
            history.append(Content(role=role, parts=parts))
        return history

    def transform_roles(self, messages):
        """Transform the roles in the messages based on the provided transformations."""
        openai_roles_to_google_roles = {
            "system": "user",
            "assistant": "model",
        }

        for message in messages:
            if role := openai_roles_to_google_roles.get(message["role"], None):
                message["role"] = role
        return messages

    def convert_response_to_openai_format(self, response):
        """Convert Google AI response to OpenAI's ChatCompletionResponse format."""
        openai_response = ChatCompletionResponse()
        openai_response.choices[0].message.content = (
            response.candidates[0].content.parts[0].text
        )
        return openai_response
