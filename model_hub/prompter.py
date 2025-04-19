# prompter.py
from typing import Dict, List, Optional, Tuple, Type, cast

# from model_hub.models import openai as openai_prompter
from model_hub.config import ModelConfig, ProviderConfigs

# Fix relative imports to use absolute imports
from model_hub.models.gemini import Gemini
from model_hub.models.model_abc import ModelName, ModelProviderABC
from model_hub.models.openai import OpenAi

# Define type for provider map
ProviderMap = Dict[str, Type[ModelProviderABC]]


class Prompter:

    _provider_map: ProviderMap = {
        ModelName.OPENAI.value: OpenAi,
        ModelName.GEMINI.value: Gemini,
        # Add more providers here as needed
    }

    def __init__(
        self,
        default_model: Optional[str] = None,
        provider_configs: Optional[ProviderConfigs] = None,
    ):
        """
        Initialize with a dictionary of provider configurations.

        Args:
            provider_configs: Dictionary mapping provider names to their configurations
                e.g., {"openai": openai_config, "gemini": gemini_config}
        Raises:
            ValueError: If no valid provider configurations are supplied
        """

        self._default_model = default_model

        self._model_providers: List[ModelProviderABC] = []

        # No valid provider configs supplied
        if not provider_configs:
            raise ValueError(
                "You must provide atleast one model provider configuration."
            )

        # Initialize only providers with valid configs
        self._append_to_model_provider_list(provider_configs)

    # Added so mypy can pick up types.
    def _get_provider_items(
        self, provider_configs: ProviderConfigs
    ) -> List[Tuple[str, Optional[ModelConfig]]]:
        return [
            (key, cast(Optional[ModelConfig], value))
            for key, value in provider_configs.items()
            if value is not None
        ]

    def _append_to_model_provider_list(self, provider_configs: ProviderConfigs) -> None:
        for provider_name, config in self._get_provider_items(provider_configs):
            if provider_name in self._provider_map and config is not None:
                provider_class = self._provider_map[provider_name]
                new_provider: ModelProviderABC = provider_class(config)
                self._model_providers.append(new_provider)

    def set_providers(self, provider_configs: Optional[ProviderConfigs] = None) -> None:
        """
        Clears any existing providers and sets only those contained in new config.
        """
        self._model_providers = []

        # No valid provider configs supplied
        if not provider_configs:
            raise ValueError(
                "You must provide atleast one model provider configuration."
            )

        # Initialize only providers with valid configs
        self._append_to_model_provider_list(provider_configs)

    def update_providers(
        self, provider_configs: Optional[ProviderConfigs] = None
    ) -> None:
        """
        Adds any new providers to the list of provider models.
        """

        # No valid provider configs supplied
        if not provider_configs:
            raise ValueError(
                "You must provide atleast one model provider configuration."
            )

        # Remove any items which will be duplicated
        self._model_providers = [
            provider
            for provider in self._model_providers
            if provider.get_name().value not in provider_configs.keys()
        ]

        # Initialize only providers with valid configs
        self._append_to_model_provider_list(provider_configs)

    def set_default_model(self, model: str) -> None:
        self._default_model = model

    def send(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Send a prompt to the appropriate model provider based on the requested model.

        This method attempts to find a provider that:
        1. Is properly initialized (not None)
        2. Has the requested model in its list of supported models
        3. Has the requested model in its list of available models

        Args:
            prompt: The text prompt to send to the model
            model: The specific model name to use

        Returns:
            The model's response as a string

        Raises:
            ValueError: If no provider supports the requested model
                    or if a provider is improperly initialized
        """
        if self._default_model is None and model is None:
            raise ValueError(
                "Prompter default_model and model arg both None. Atleast one must be defined."
            )

        if model is None:
            model = self._default_model

        for provider in self._model_providers:
            if provider is None:
                raise ValueError("Provider should never be None")
            if model not in provider.get_supported_models():
                continue
            if model not in provider.get_all_models():
                continue
            return provider.request(prompt, model)
        raise ValueError(f"Model - {model} - not supported by any providers")
