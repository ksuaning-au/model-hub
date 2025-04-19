import unittest
from unittest.mock import MagicMock, patch
from model_hub.models.openai import OpenAi
from model_hub.config import ModelConfig
from model_hub.models.model_abc import ModelName

class TestOpenAI(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            api_key="test-openai-key",
            supported_models=["gpt-4o-mini", "gpt-4"]
        )
        
        # Create patch for openai.OpenAI
        self.client_patch = patch('model_hub.models.openai.openai.OpenAI')
        self.mock_client_class = self.client_patch.start()
        
        # Setup mock client and models
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client
        
        # Mock models list
        self.mock_model1 = MagicMock()
        self.mock_model1.id = "gpt-4o-mini"
        
        self.mock_model2 = MagicMock()
        self.mock_model2.id = "gpt-4"
        
        self.mock_client.models.list.return_value = [self.mock_model1, self.mock_model2]
        
        # Mock response
        self.mock_response = MagicMock()
        self.mock_response.output_text = "This is a mock OpenAI response"
        self.mock_client.responses.create.return_value = self.mock_response
        
        # Initialize the OpenAI provider
        self.openai = OpenAi(self.config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patch.stop()
    
    def test_initialization(self):
        """Test that OpenAI provider initializes correctly."""
        self.assertEqual(self.openai._config, self.config)
        self.mock_client_class.assert_called_once_with(api_key="test-openai-key")
    
    def test_get_name(self):
        """Test get_name returns the correct model name."""
        self.assertEqual(OpenAi.get_name(), ModelName.OPENAI)
    
    def test_get_all_models(self):
        """Test get_all_models returns the correct models."""
        models = self.openai.get_all_models()
        self.assertEqual(models, ["gpt-4o-mini", "gpt-4"])
        
        # Call again to test caching
        self.mock_client.models.list.reset_mock()
        models = self.openai.get_all_models()
        self.assertEqual(models, ["gpt-4o-mini", "gpt-4"])
        # Should not be called again because of caching
        self.mock_client.models.list.assert_not_called()
    
    def test_get_supported_models(self):
        """Test get_supported_models returns the configured supported models."""
        models = self.openai.get_supported_models()
        self.assertEqual(models, ["gpt-4o-mini", "gpt-4"])
    
    def test_request(self):
        """Test request sends the correct parameters and returns the response."""
        response = self.openai.request("What is the meaning of life?", "gpt-4o-mini")
        
        self.mock_client.responses.create.assert_called_once()
        call_args = self.mock_client.responses.create.call_args[1]
        
        self.assertEqual(call_args["model"], "gpt-4o-mini")
        self.assertEqual(call_args["input"], "What is the meaning of life?")
        self.assertEqual(call_args["temperature"], 0.5)
        self.assertEqual(call_args["max_output_tokens"], 4096)
        
        self.assertEqual(response, "This is a mock OpenAI response")

if __name__ == "__main__":
    unittest.main()