import unittest
from unittest.mock import MagicMock, patch
from model_hub.models.gemini import Gemini
from model_hub.config import ModelConfig
from model_hub.models.model_abc import ModelName

class TestGemini(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            api_key="test-api-key",
            supported_models=["gemini-2.0-flash", "gemini-1.5-pro"]
        )
        
        # Create patch for genai.Client
        self.client_patch = patch('model_hub.models.gemini.genai.Client')
        self.mock_client_class = self.client_patch.start()
        
        # Setup mock client and models
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client
        
        # Mock models list
        self.mock_model = MagicMock()
        self.mock_model.name = "models/gemini-2.0-flash"
        self.mock_model.supported_actions = ["generateContent"]
        
        self.mock_model2 = MagicMock()
        self.mock_model2.name = "models/gemini-1.5-pro"
        self.mock_model2.supported_actions = ["generateContent"]
        
        self.mock_client.models.list.return_value = [self.mock_model, self.mock_model2]
        
        # Mock response
        self.mock_response = MagicMock()
        self.mock_response.text = "This is a mock response"
        self.mock_client.models.generate_content.return_value = self.mock_response
        
        # Initialize the Gemini provider
        self.gemini = Gemini(self.config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patch.stop()
    
    def test_initialization(self):
        """Test that Gemini provider initializes correctly."""
        self.assertEqual(self.gemini._config, self.config)
        self.mock_client_class.assert_called_once_with(api_key="test-api-key")
    
    def test_get_name(self):
        """Test get_name returns the correct model name."""
        self.assertEqual(Gemini.get_name(), ModelName.GEMINI)
    
    def test_get_all_models(self):
        """Test get_all_models returns the correct models."""
        models = self.gemini.get_all_models()
        self.assertEqual(models, ["gemini-2.0-flash", "gemini-1.5-pro"])
        
        # Call again to test caching
        self.mock_client.models.list.reset_mock()
        models = self.gemini.get_all_models()
        self.assertEqual(models, ["gemini-2.0-flash", "gemini-1.5-pro"])
        # Should not be called again because of caching
        self.mock_client.models.list.assert_not_called()
    
    def test_get_supported_models(self):
        """Test get_supported_models returns the configured supported models."""
        models = self.gemini.get_supported_models()
        self.assertEqual(models, ["gemini-2.0-flash", "gemini-1.5-pro"])
    
    def test_request(self):
        """Test request sends the correct parameters and returns the response."""
        response = self.gemini.request("Hello, world!", "gemini-2.0-flash")
        
        self.mock_client.models.generate_content.assert_called_once()
        call_args = self.mock_client.models.generate_content.call_args[1]
        
        self.assertEqual(call_args["model"], "gemini-2.0-flash")
        self.assertEqual(call_args["contents"], "Hello, world!")
        self.assertEqual(call_args["config"].max_output_tokens, 4096)
        self.assertEqual(call_args["config"].temperature, 0.5)
        
        self.assertEqual(response, "This is a mock response")

if __name__ == "__main__":
    unittest.main()