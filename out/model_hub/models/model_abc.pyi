import abc
import enum
from abc import ABC, abstractmethod
from model_hub.config import ModelConfig as ModelConfig

class ModelName(enum.Enum):
    OPENAI = 'openai'
    GEMINI = 'gemini'

class ModelProviderABC(ABC, metaclass=abc.ABCMeta):
    def __init__(self, config: ModelConfig) -> None: ...
    @abstractmethod
    def get_name(self) -> ModelName: ...
    @abstractmethod
    def request(self, prompt: str, model: str) -> str: ...
    @abstractmethod
    def get_all_models(self) -> list[str]: ...
    @abstractmethod
    def get_supported_models(self) -> list[str]: ...
