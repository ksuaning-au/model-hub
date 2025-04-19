import enum
from abc import ABC, abstractmethod
from typing import List

from model_hub.config import ModelConfig


class ModelName(enum.Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class ModelProviderABC(ABC):
    def __init__(self, config: ModelConfig):
        self._config = config

    @abstractmethod
    def get_name(self) -> ModelName: ...

    @abstractmethod
    def request(self, prompt: str, model: str) -> str: ...

    @abstractmethod
    def get_all_models(self) -> List[str]: ...

    @abstractmethod
    def get_supported_models(self) -> List[str]: ...
