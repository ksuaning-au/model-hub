import unittest
from model_hub.config import ModelConfig, ProviderConfigs
from dataclasses import field
from typing import List

class TestModelConfig(unittest.TestCase):
    def test_default_values(self):
        """Test that ModelConfig initializes with correct default values."""
        config = ModelConfig()
        self.assertIsNone(config.api_key)
        self.assertEqual(config.supported_models, [])
        self.assertEqual(config.max_response_tokens, 4096)
        self.assertEqual(config.temperature, 0.5)
    
    def test_custom_values(self):
        """Test that ModelConfig accepts custom values."""
        config = ModelConfig(
            api_key="test-api-key",
            supported_models=["model1", "model2"],
            max_response_tokens=2048,
            temperature=0.7
        )
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.supported_models, ["model1", "model2"])
        self.assertEqual(config.max_response_tokens, 2048)
        self.assertEqual(config.temperature, 0.7)
    
    def test_mutable_default(self):
        """Test that each instance has its own supported_models list."""
        config1 = ModelConfig()
        config2 = ModelConfig()
        
        config1.supported_models.append("model1")
        
        self.assertEqual(config1.supported_models, ["model1"])
        self.assertEqual(config2.supported_models, [])

class TestProviderConfigs(unittest.TestCase):
    def test_typed_dict(self):
        """Test that ProviderConfigs works as a TypedDict."""
        config = ProviderConfigs(
            openai=ModelConfig(api_key="openai-key"),
            gemini=ModelConfig(api_key="gemini-key")
        )
        
        self.assertEqual(config["openai"].api_key, "openai-key")
        self.assertEqual(config["gemini"].api_key, "gemini-key")
    
    def test_optional_fields(self):
        """Test that ProviderConfigs fields are optional."""
        # Only openai
        config1 = ProviderConfigs(openai=ModelConfig(api_key="openai-key"))
        self.assertEqual(config1["openai"].api_key, "openai-key")
        self.assertNotIn("gemini", config1)
        
        # Only gemini
        config2 = ProviderConfigs(gemini=ModelConfig(api_key="gemini-key"))
        self.assertEqual(config2["gemini"].api_key, "gemini-key")
        self.assertNotIn("openai", config2)
        
        # Empty is valid too
        config3 = ProviderConfigs()
        self.assertEqual(len(config3), 0)

if __name__ == "__main__":
    unittest.main()