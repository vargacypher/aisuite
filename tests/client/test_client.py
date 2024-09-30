import unittest
from unittest.mock import patch
from aisuite import Client
from aisuite import ProviderNames


class TestClient(unittest.TestCase):
    @patch("aisuite.providers.mistral_provider.MistralProvider.chat_completions_create")
    @patch("aisuite.providers.groq_provider.GroqProvider.chat_completions_create")
    @patch("aisuite.providers.openai_provider.OpenAIProvider.chat_completions_create")
    @patch(
        "aisuite.providers.aws_bedrock_provider.AWSBedrockProvider.chat_completions_create"
    )
    @patch("aisuite.providers.azure_provider.AzureProvider.chat_completions_create")
    @patch(
        "aisuite.providers.anthropic_provider.AnthropicProvider.chat_completions_create"
    )
    @patch("aisuite.providers.google_provider.GoogleProvider.chat_completions_create")
    def test_client_chat_completions(
        self,
        mock_google,
        mock_anthropic,
        mock_azure,
        mock_bedrock,
        mock_openai,
        mock_groq,
        mock_mistral,
    ):
        # Mock responses from providers
        mock_openai.return_value = "OpenAI Response"
        mock_bedrock.return_value = "AWS Bedrock Response"
        mock_azure.return_value = "Azure Response"
        mock_anthropic.return_value = "Anthropic Response"
        mock_groq.return_value = "Groq Response"
        mock_mistral.return_value = "Mistral Response"
        mock_google.return_value = "Google Response"

        # Provider configurations
        provider_configs = {
            ProviderNames.OPENAI: {"api_key": "test_openai_api_key"},
            ProviderNames.AWS: {
                "aws_access_key": "test_aws_access_key",
                "aws_secret_key": "test_aws_secret_key",
                "aws_session_token": "test_aws_session_token",
                "aws_region": "us-west-2",
            },
            ProviderNames.AZURE: {
                "api_key": "azure-api-key",
                "base_url": "https://model.ai.azure.com",
            },
            ProviderNames.GROQ: {
                "api_key": "groq-api-key",
            },
            ProviderNames.MISTRAL: {
                "api_key": "mistral-api-key",
            },
            ProviderNames.GOOGLE: {
                "project_id": "test_google_project_id",
                "region": "us-west4",
                "application_credentials": "test_google_application_credentials",
            },
        }

        # Initialize the client
        client = Client()
        client.configure(provider_configs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]

        # Test OpenAI model
        open_ai_model = ProviderNames.OPENAI + ":" + "gpt-4o"
        openai_response = client.chat.completions.create(
            open_ai_model, messages=messages
        )
        self.assertEqual(openai_response, "OpenAI Response")
        mock_openai.assert_called_once()

        # Test AWS Bedrock model
        bedrock_model = ProviderNames.AWS + ":" + "claude-v3"
        bedrock_response = client.chat.completions.create(
            bedrock_model, messages=messages
        )
        self.assertEqual(bedrock_response, "AWS Bedrock Response")
        mock_bedrock.assert_called_once()

        # Test Azure model
        azure_model = ProviderNames.AZURE + ":" + "azure-model"
        azure_response = client.chat.completions.create(azure_model, messages=messages)
        self.assertEqual(azure_response, "Azure Response")
        mock_azure.assert_called_once()

        # Test Anthropic model
        anthropic_model = ProviderNames.ANTHROPIC + ":" + "anthropic-model"
        anthropic_response = client.chat.completions.create(
            anthropic_model, messages=messages
        )
        self.assertEqual(anthropic_response, "Anthropic Response")
        mock_anthropic.assert_called_once()

        # Test Groq model
        groq_model = ProviderNames.GROQ + ":" + "groq-model"
        groq_response = client.chat.completions.create(groq_model, messages=messages)
        self.assertEqual(groq_response, "Groq Response")
        mock_groq.assert_called_once()

        # Test Mistral model
        mistral_model = ProviderNames.MISTRAL + ":" + "mistral-model"
        mistral_response = client.chat.completions.create(
            mistral_model, messages=messages
        )
        self.assertEqual(mistral_response, "Mistral Response")
        mock_mistral.assert_called_once()

        # Test Google model
        google_model = ProviderNames.GOOGLE + ":" + "google-model"
        google_response = client.chat.completions.create(
            google_model, messages=messages
        )
        self.assertEqual(google_response, "Google Response")
        mock_google.assert_called_once()

        # Test that new instances of Completion are not created each time we make an inference call.
        compl_instance = client.chat.completions
        next_compl_instance = client.chat.completions
        assert compl_instance is next_compl_instance

    @patch("aisuite.providers.openai_provider.OpenAIProvider.chat_completions_create")
    def test_invalid_provider_in_client_config(self, mock_openai):
        # Testing an invalid provider name in the configuration
        invalid_provider_configs = {
            "INVALID_PROVIDER": {"api_key": "invalid_api_key"},
        }

        # Expect ValueError when initializing Client with invalid provider
        with self.assertRaises(ValueError) as context:
            client = Client(invalid_provider_configs)

        # Verify the error message
        self.assertIn(
            "Provider INVALID_PROVIDER is not a valid provider",
            str(context.exception),
        )

    @patch("aisuite.providers.openai_provider.OpenAIProvider.chat_completions_create")
    def test_invalid_model_format_in_create(self, mock_openai):
        # Valid provider configurations
        provider_configs = {
            ProviderNames.OPENAI: {"api_key": "test_openai_api_key"},
        }

        # Initialize the client with valid provider
        client = Client(provider_configs)
        client.configure(provider_configs)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]

        # Invalid model format
        invalid_model = "invalidmodel"

        # Expect ValueError when calling create with invalid model format
        with self.assertRaises(ValueError) as context:
            client.chat.completions.create(invalid_model, messages=messages)

        # Verify the error message
        self.assertIn(
            "Invalid model format. Expected 'provider:model'", str(context.exception)
        )

    def create_mock_provider_response(self, expected_response):
        """
        Helper method to create a mock provider response.
        """
        mock_response = unittest.mock.Mock()
        mock_response.choices = [unittest.mock.Mock()]
        mock_response.choices[0].message.content = expected_response
        return mock_response

    @patch("aisuite.providers.openai_provider.OpenAIProvider.chat_completions_create")
    def test_client_call_method(self, mock_openai):
        expected_response = "expected-text-response"
        provider_response = self.create_mock_provider_response(expected_response)
        mock_openai.return_value = provider_response

        config = {
            ProviderNames.OPENAI: {"api_key": "test_openai_api_key"},
        }

        new_client = Client(config)

        # Test __call__ method (without system message)
        response = new_client("test-user-prompt", "openai:gpt-3.5-turbo")
        self.assertEqual(response, expected_response)
        mock_openai.assert_called_once()

        mock_openai.reset_mock()

        # Test __call__ method (with system message)
        response = new_client(
            "test-user-prompt",
            "openai:gpt-3.5-turbo",
            system_message="You are a helpful assistant.",
        )
        self.assertEqual(response, expected_response)
        mock_openai.assert_called_once()

        # Ensure the correct messages were passed to the provider
        called_messages = mock_openai.call_args[0][1]
        self.assertEqual(len(called_messages), 2)
        self.assertEqual(called_messages[0]["role"], "system")
        self.assertEqual(called_messages[0]["content"], "You are a helpful assistant.")
        self.assertEqual(called_messages[1]["role"], "user")
        self.assertEqual(called_messages[1]["content"], "test-user-prompt")

    @patch("aisuite.providers.openai_provider.OpenAIProvider.chat_completions_create")
    def test_client_call_method_with_kwargs(self, mock_openai):
        expected_response = "expected-text-response"
        provider_response = self.create_mock_provider_response(expected_response)
        mock_openai.return_value = provider_response

        config = {
            ProviderNames.OPENAI: {"api_key": "test_openai_api_key"},
        }

        new_client = Client(config)

        # Test __call__ method with additional kwargs
        response = new_client(
            "test-user-prompt", "openai:gpt-3.5-turbo", max_tokens=100, temperature=0.7
        )
        self.assertEqual(response, expected_response)
        mock_openai.assert_called_once()

        # Ensure the additional kwargs were passed to the provider
        _, kwargs = mock_openai.call_args
        self.assertEqual(kwargs.get("max_tokens"), 100)
        self.assertEqual(kwargs.get("temperature"), 0.7)


if __name__ == "__main__":
    unittest.main()
