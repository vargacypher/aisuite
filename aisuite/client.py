from .provider import ProviderFactory, ProviderNames


class Client:
    def __init__(self, provider_configs: dict = {}):
        """
        Initialize the client with provider configurations.
        Use the ProviderFactory to create provider instances.

        Args:
            provider_configs (dict): A dictionary containing provider configurations.
                Each key should be a ProviderNames enum or its string representation,
                and the value should be a dictionary of configuration options for that provider.
                For example:
                {
                    ProviderNames.OPENAI: {"api_key": "your_openai_api_key"},
                    "aws-bedrock": {
                        "aws_access_key": "your_aws_access_key",
                        "aws_secret_key": "your_aws_secret_key",
                        "aws_region": "us-west-2"
                    }
                }
        """
        self.providers = {}
        self.provider_configs = provider_configs
        self._chat = None
        self._initialize_providers()

    def __call__(self, prompt: str, model: str, system_message: str = None, **kwargs):
        """
        Call the client directly with a prompt and model.

        Args:
            prompt (str): The user's prompt or question.
            model (str): The model to use in the format "provider:model".
            system_message (str, optional): A system message to set the context.
            **kwargs: Additional arguments to pass to the chat completion.

        Returns:
            The response from the AI model.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = self.chat.completions.create(model, messages, **kwargs)
        return response.choices[0].message.content

    def _initialize_providers(self):
        """Helper method to initialize or update providers."""
        for provider_key, config in self.provider_configs.items():
            provider_key = self._validate_provider_key(provider_key)
            self.providers[provider_key.value] = ProviderFactory.create_provider(
                provider_key, config
            )

    def _validate_provider_key(self, provider_key):
        """
        Validate if the provider key is part of ProviderNames enum.
        Allow strings as well and convert them to ProviderNames.
        """
        if isinstance(provider_key, str):
            if provider_key not in ProviderNames._value2member_map_:
                raise ValueError(f"Provider {provider_key} is not a valid provider")
            return ProviderNames(provider_key)

        if isinstance(provider_key, ProviderNames):
            return provider_key

        raise ValueError(
            f"Provider {provider_key} should either be a string or enum ProviderNames"
        )

    def configure(self, provider_configs: dict = None):
        """
        Configure the client with provider configurations.
        """
        if provider_configs is None:
            return

        self.provider_configs.update(provider_configs)
        self._initialize_providers()  # NOTE: This will override existing provider instances.

    @property
    def chat(self):
        """Return the chat API interface."""
        if not self._chat:
            self._chat = Chat(self)
        return self._chat


class Chat:
    def __init__(self, client: "Client"):
        self.client = client
        self._completions = Completions(self.client)

    @property
    def completions(self):
        """Return the completions interface."""
        return self._completions


class Completions:
    def __init__(self, client: "Client"):
        self.client = client

    def create(self, model: str, messages: list, **kwargs):
        """
        Create chat completion based on the model, messages, and any extra arguments.
        """
        # Check that correct format is used
        if ":" not in model:
            raise ValueError(
                f"Invalid model format. Expected 'provider:model', got '{model}'"
            )

        # Extract the provider key from the model identifier, e.g., "aws-bedrock:model-name"
        provider_key, model_name = model.split(":", 1)

        if provider_key not in ProviderNames._value2member_map_:
            # If the provider key does not match, give a clearer message to guide the user
            valid_providers = ", ".join([p.value for p in ProviderNames])
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Expected one of: {valid_providers}. "
                "Make sure the model string is formatted correctly as 'provider:model'."
            )

        if provider_key not in self.client.providers:
            config = {}
            if provider_key in self.client.provider_configs:
                config = self.client.provider_configs[provider_key]
            self.client.providers[provider_key] = ProviderFactory.create_provider(
                ProviderNames(provider_key), config
            )

        provider = self.client.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Could not load provider for {provider_key}.")

        # Delegate the chat completion to the correct provider's implementation
        # Any additional arguments will be passed to the provider's implementation.
        # Eg: max_tokens, temperature, etc.
        return provider.chat_completions_create(model_name, messages, **kwargs)
