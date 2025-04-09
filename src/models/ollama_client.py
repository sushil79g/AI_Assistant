import json
import typing
from typing import Dict, List, Optional

import requests

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = self._fetch_available_models()
        self.current_model = "gemma3:4b"  # Default model

    def _fetch_available_models(self) -> List[str]:
        """Fetch available models from Ollama API."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return ["gemma3:4b"]  # Fallback to default model
        except Exception:
            return ["gemma3:4b"]  # Fallback to default model

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        return model_name in self.available_models

    def _parse_response(self, response: requests.Response) -> Dict[str, typing.Any]:
        """Parse the response from Ollama API."""
        try:
            # Try to parse the entire response as JSON
            return response.json()
        except json.JSONDecodeError as e:
            # If that fails, try to parse line by line
            try:
                # Split the response into lines and parse each line
                lines = response.text.strip().split('\n')
                parsed_responses = []
                for line in lines:
                    if line.strip():  # Skip empty lines
                        parsed_responses.append(json.loads(line))
                
                # If we have multiple responses, return the last one
                if parsed_responses:
                    return parsed_responses[-1]
                return {"error": "No valid response found in the stream"}
            except json.JSONDecodeError:
                return {"error": f"Failed to parse response: {str(e)}"}

    def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, typing.Any]:
        """Send a chat message to the Ollama API."""
        if model is None:
            model = self.current_model

        if not self.is_model_available(model):
            return {
                "error": f"Model '{model}' is not available locally. Please run 'ollama pull {model}' to download it."
            }

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False  # Ensure we get a single response
        }

        if system:
            payload["system"] = system

        if context:
            payload["context"] = context

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return self._parse_response(response)
            elif response.status_code == 404:
                return {
                    "error": f"Model '{model}' is not available locally. Please run 'ollama pull {model}' to download it."
                }
            else:
                return {"error": f"Ollama call failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": f"Error calling Ollama API: {str(e)}"}

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, typing.Any]:
        """Generate text using the Ollama API."""
        if model is None:
            model = self.current_model

        if not self.is_model_available(model):
            return {
                "error": f"Model '{model}' is not available locally. Please run 'ollama pull {model}' to download it."
            }

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False  # Ensure we get a single response
        }

        if system:
            payload["system"] = system

        if context:
            payload["context"] = context

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return self._parse_response(response)
            elif response.status_code == 404:
                return {
                    "error": f"Model '{model}' is not available locally. Please run 'ollama pull {model}' to download it."
                }
            else:
                return {"error": f"Ollama call failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": f"Error calling Ollama API: {str(e)}"}

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.available_models

    def set_model(self, model_name: str) -> bool:
        """Set the current model to use."""
        if model_name in self.available_models:
            self.current_model = model_name
            return True
        return False 