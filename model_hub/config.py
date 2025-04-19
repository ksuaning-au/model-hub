from dataclasses import dataclass, field
from typing import Optional, List, TypedDict

@dataclass
class ModelConfig:
    api_key: Optional[str] = None
    supported_models: List[str] = field(default_factory=list)
    max_response_tokens: int = 4096
    temperature: float = 0.5

class ProviderConfigs(TypedDict, total = False):
    openai: Optional[ModelConfig] = None
    gemini: Optional[ModelConfig] = None
    