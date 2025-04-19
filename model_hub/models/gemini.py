import os
from typing import List, Optional

from google import genai
from google.genai import types
from model_hub.config import ModelConfig
from model_hub.models.model_abc import ModelProviderABC, ModelName


class Gemini(ModelProviderABC):
    def __init__(self, config: ModelConfig):
        self._config = config
        self._client = genai.Client(api_key=self._config.api_key)
        self._models: Optional[List[str]] = None 
    
    def get_name() -> ModelName:
        return ModelName.GEMINI
    
    def request(self, prompt: str, model: str) -> str:
        config: types.GenerateContentConfig = types.GenerateContentConfig(
            max_output_tokens=self._config.max_response_tokens,
            temperature=self._config.temperature,
        )
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return response.text

    def _get_models(self):
        models_list: List[str] = []
        for model in self._client.models.list():
            for action in model.supported_actions:
                if action == "generateContent":
                    model_name = model.name.replace("models/", "")
                    models_list.append(model_name)
        return models_list
    
    def get_all_models(self):
        if self._models is None:
            self._models = self._get_models()
        return self._models
    
    def get_supported_models(self):
        return self._config.supported_models
