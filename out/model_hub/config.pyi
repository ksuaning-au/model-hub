from dataclasses import dataclass, field
from typing import TypedDict

@dataclass
class ModelConfig:
    api_key: str | None = ...
    supported_models: list[str] = field(default_factory=list)
    max_response_tokens: int = ...
    temperature: float = ...

class ProviderConfigs(TypedDict, total=False):
    openai: ModelConfig | None
    gemini: ModelConfig | None
