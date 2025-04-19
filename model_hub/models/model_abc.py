from abc import ABC, abstractmethod
import enum

class ModelName(enum.Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    

class ModelProviderABC(ABC):
    @abstractmethod
    def get_name() -> ModelName:
        raise NotImplemented("Model Provider Get Name Not Implemented")
    
    @abstractmethod
    def request(prompt: str, model: str, temperature: float, max_tokens: int):
        raise NotImplemented("Model Provider Request Not Implemented")
    
    @abstractmethod
    def get_all_models():
        raise NotImplemented("Model Provider Get All Models Not Implemented")
    
    @abstractmethod
    def get_supported_models():
        raise NotImplemented("Model Provider Get Supported Not Implemented")