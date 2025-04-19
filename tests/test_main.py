import unittest
from unittest.mock import MagicMock, patch
from typing import List, Optional
from model_hub.main import Prompter
from model_hub.config import ModelConfig, ProviderConfigs
from model_hub.models.model_abc import ModelName, ModelProviderABC

class MockProvider(ModelProviderABC):
    def __init__(self, config: ModelConfig, name: ModelName, all_models: List[str]):
        self._config = config
        self._name = name
        self._all_models = all_models
        self.request_calls = []
    
    def get_name():
        return ModelName.GEMINI  # Default for testing
    
    def get_all_models(self):
        return self._all_models
    
    def get_supported_models(self):
        return self._config.supported_models
    
    def request(self, prompt: str, model: str):
        self.request_calls.append((prompt, model))
        return f"Response from {self._name.value} model {model}: {prompt}"

class TestPrompter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create mock provider classes
        self.mock_gemini_provider = MagicMock()
        self.mock_openai_provider = MagicMock()
        
        # Set up provider map patch
        self.provider_map_patch = patch.dict(
            'model_hub.main.Prompter._provider_map', 
            {
                ModelName.GEMINI.value: lambda config: self.mock_gemini_provider,
                ModelName.OPENAI.value: lambda config: self.mock_openai_provider
            },
            clear=True
        )
        self.provider_map_patch.start()
        
        # Configure mock providers
        self.mock_gemini_provider.get_name.return_value = ModelName.GEMINI
        self.mock_gemini_provider.get_all_models.return_value = ["gemini-2.0-flash", "gemini-1.5-pro"]
        self.mock_gemini_provider.get_supported_models.return_value = ["gemini-2.0-flash"]
        self.mock_gemini_provider.request.return_value = "Mock Gemini response"
        
        self.mock_openai_provider.get_name.return_value = ModelName.OPENAI
        self.mock_openai_provider.get_all_models.return_value = ["gpt-4o-mini", "gpt-4"]
        self.mock_openai_provider.get_supported_models.return_value = ["gpt-4o-mini"]
        self.mock_openai_provider.request.return_value = "Mock OpenAI response"
        
        # Provider configs for testing
        self.provider_configs = {
            ModelName.GEMINI.value: ModelConfig(
                api_key="test-gemini-key",
                supported_models=["gemini-2.0-flash"]
            ),
            ModelName.OPENAI.value: ModelConfig(
                api_key="test-openai-key",
                supported_models=["gpt-4o-mini"]
            )
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.provider_map_patch.stop()
    
    def test_initialization(self):
        """Test Prompter initializes with the correct configuration."""
        prompter = Prompter("gemini-2.0-flash", self.provider_configs)
        self.assertEqual(prompter._default_model, "gemini-2.0-flash")
        self.assertEqual(len(prompter._model_providers), 2)
    
    def test_initialization_requires_configs(self):
        """Test Prompter initialization requires provider configs."""
        with self.assertRaises(ValueError):
            Prompter("gemini-2.0-flash", None)
    
    def test_send_with_default_model(self):
        """Test send uses the default model when no model is specified."""
        prompter = Prompter("gemini-2.0-flash", self.provider_configs)
        response = prompter.send("Hello, world!")
        
        self.mock_gemini_provider.request.assert_called_once_with("Hello, world!", "gemini-2.0-flash")
        self.assertEqual(response, "Mock Gemini response")
    
    def test_send_with_specific_model(self):
        """Test send uses the specified model."""
        prompter = Prompter("gemini-2.0-flash", self.provider_configs)
        response = prompter.send("Hello, world!", "gpt-4o-mini")
        
        self.mock_openai_provider.request.assert_called_once_with("Hello, world!", "gpt-4o-mini")
        self.assertEqual(response, "Mock OpenAI response")
    
    def test_send_requires_model(self):
        """Test send requires either default_model or model parameter."""
        prompter = Prompter(None, self.provider_configs)
        with self.assertRaises(ValueError):
            prompter.send("Hello, world!")
    
    def test_send_unsupported_model(self):
        """Test send raises error for unsupported model."""
        prompter = Prompter("gemini-2.0-flash", self.provider_configs)
        with self.assertRaises(ValueError):
            prompter.send("Hello, world!", "unsupported-model")
    
    def test_set_default_model(self):
        """Test setting a new default model."""
        prompter = Prompter("gemini-2.0-flash", self.provider_configs)
        prompter.set_default_model("gpt-4o-mini")
        
        response = prompter.send("Hello, world!")
        self.mock_openai_provider.request.assert_called_once_with("Hello, world!", "gpt-4o-mini")
    
    def test_set_providers(self):
        """Test setting new providers clears existing ones."""
        prompter = Prompter("gemini-2.0-flash", self.provider_configs)
        
        # Reset mocks to check calls
        self.mock_gemini_provider.reset_mock()
        self.mock_openai_provider.reset_mock()
        
        # Set only OpenAI provider
        new_configs = {
            ModelName.OPENAI.value: ModelConfig(
                api_key="new-openai-key",
                supported_models=["gpt-4o-mini"]
            )
        }
        
        prompter.set_providers(new_configs)
        
        # Should have only one provider now
        self.assertEqual(len(prompter._model_providers), 1)
        
        # Try to use Gemini model
        with self.assertRaises(ValueError):
            prompter.send("Hello, world!", "gemini-2.0-flash")
    
    def test_update_providers(self):
        """Test updating providers maintains non-duplicated ones."""
        # Create prompter with only Gemini
        gemini_only = {
            ModelName.GEMINI.value: ModelConfig(
                api_key="test-gemini-key",
                supported_models=["gemini-2.0-flash"]
            )
        }
        prompter = Prompter("gemini-2.0-flash", gemini_only)
        
        # Initially has only Gemini provider
        self.assertEqual(len(prompter._model_providers), 1)
        
        # Add OpenAI provider
        openai_only = {
            ModelName.OPENAI.value: ModelConfig(
                api_key="test-openai-key",
                supported_models=["gpt-4o-mini"]
            )
        }
        
        prompter.update_providers(openai_only)
        
        # Should now have both providers
        self.assertEqual(len(prompter._model_providers), 2)
        
        # Both models should work
        prompter.send("Hello, Gemini!", "gemini-2.0-flash")
        prompter.send("Hello, OpenAI!", "gpt-4o-mini")
        
        self.mock_gemini_provider.request.assert_called_once_with("Hello, Gemini!", "gemini-2.0-flash")
        self.mock_openai_provider.request.assert_called_once_with("Hello, OpenAI!", "gpt-4o-mini")

if __name__ == "__main__":
    unittest.main()