# Model Hub

A flexible and unified interface for working with various LLM providers including OpenAI and Google's Gemini.

## Features

- 🤝 **Single API** for multiple LLM providers
- 🔄 **Runtime configuration** of API keys and model parameters
- 📋 **Provider management** - add, update, or replace providers on the fly
- 🔎 **Model discovery** - automatic detection of available models
- ⚙️ **Type safety** - fully typed with mypy support

## Installation

```bash
pip install model-hub
```

## Quick Start

```python
from model_hub.prompter import Prompter
from model_hub.config import ModelConfig

# Load API keys from environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Create model configurations
openai_config = ModelConfig(
    api_key=os.getenv("OPENAI_API_KEY"),
    supported_models=["gpt-4o-mini"],
    max_response_tokens=4096,
    temperature=0.5
)

gemini_config = ModelConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    supported_models=["gemini-2.0-flash"]
)

# Initialize prompter with configurations and default model
prompter = Prompter(
    default_model="gemini-2.0-flash",
    provider_configs={
        "openai": openai_config,
        "gemini": gemini_config
    }
)

# Send a prompt using the default model (Gemini)
response = prompter.send("Tell me a fun fact about space")
print(f"Gemini response: {response}")

# Switch to using OpenAI
prompter.set_default_model("gpt-4o-mini")
response = prompter.send("Tell me a fun fact about the ocean")
print(f"OpenAI response: {response}")

# Or explicitly specify the model for a specific request
response = prompter.send("What's your favorite color?", model="gemini-2.0-flash")
print(f"Gemini response: {response}")
```

## Environment Setup

Create a .env file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## Advanced Usage

### Adding/Updating Providers

```python
# Add a new provider configuration or update an existing one
prompter.update_providers({
    "gemini": ModelConfig(
        api_key=new_gemini_api_key,
        supported_models=["gemini-2.0-flash", "gemini-2.0-pro"]
    )
})
```

### Replacing All Providers

```python
# Remove all existing providers and set new ones
prompter.set_providers({
    "openai": openai_config
})
```

## Development

### Testing

```bash
# Install test dependencies
pip install -r requirements.txt

# Run tests
./test.sh
```

### Type Checking

```bash
mypy model_hub
```

## License

MIT

## Contributors

- Original implementation by @ksuaning