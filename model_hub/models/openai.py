from typing import List, Optional

import openai
from openai.types.responses.response import Response

from model_hub.config import ModelConfig
from model_hub.models.model_abc import ModelName, ModelProviderABC


class OpenAi(ModelProviderABC):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = openai.OpenAI(api_key=self._config.api_key)
        self._models: Optional[List[str]] = None

    def get_name(self) -> ModelName:
        return ModelName.OPENAI

    def request(self, prompt: str, model: str) -> str:
        response: Response = self._client.responses.create(
            model=model,
            input=prompt,
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_response_tokens,
        )
        return response.output_text

    def _get_models(self) -> List[str]:
        return [item.id for item in self._client.models.list()]

    def get_all_models(self) -> List[str]:
        if self._models is None:
            self._models = self._get_models()
        return self._models

    def get_supported_models(self) -> List[str]:
        return self._config.supported_models
