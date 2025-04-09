from unittest.mock import MagicMock, patch

import pytest

from src.models.ollama_client import OllamaClient

@pytest.fixture
def ollama_client():
    return OllamaClient()

def test_initialization(ollama_client):
    assert ollama_client.base_url == "http://localhost:11434"
    assert ollama_client.current_model == "llama2"
    assert isinstance(ollama_client._available_models, list)

@patch('requests.get')
def test_fetch_available_models(mock_get, ollama_client):
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "llama2"},
            {"name": "mistral"}
        ]
    }
    mock_get.return_value = mock_response
    
    models = ollama_client._fetch_available_models()
    assert len(models) == 2
    assert "llama2" in models
    assert "mistral" in models

@patch('requests.get')
def test_fetch_available_models_failure(mock_get):
    # Create a new client with the mocked request
    mock_get.side_effect = Exception("Connection error")
    client = OllamaClient()  # This will call _fetch_available_models
    
    assert len(client._available_models) == 1
    assert client._available_models[0] == "llama2"

def test_set_model(ollama_client):
    ollama_client._available_models = ["llama2", "mistral"]
    ollama_client.set_model("mistral")
    assert ollama_client.current_model == "mistral"
    
    # Test setting invalid model
    ollama_client.set_model("invalid_model")
    assert ollama_client.current_model == "mistral"  # Should not change

@patch('requests.post')
def test_generate_response(mock_post, ollama_client):
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test response"}
    mock_post.return_value = mock_response
    
    response = ollama_client.generate_response("Test prompt")
    assert response == "Test response"
    
    # Verify the API call
    mock_post.assert_called_once()
    call_args = mock_post.call_args[1]['json']
    assert call_args['model'] == "llama2"
    assert call_args['prompt'] == "Test prompt"
    assert not call_args['stream']

@patch('requests.post')
def test_generate_response_with_context(mock_post, ollama_client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Test response"}
    mock_post.return_value = mock_response
    
    response = ollama_client.generate_response("Test prompt", "Test context")
    assert response == "Test response"
    
    call_args = mock_post.call_args[1]['json']
    assert "Test context" in call_args['prompt']
    assert "Test prompt" in call_args['prompt'] 