import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from core.config import MODELS
from core.errors import LLMError

# Load environment variables from .env file
load_dotenv()

class LLMClient:
    def __init__(self, model_key: str):
        self.model = MODELS[model_key]
        self.api_key = self._get_api_key()
        self.temperature = self._get_temperature()
        
    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")
        return api_key
    
    def _get_temperature(self) -> float:
        """Get temperature setting based on model."""
        if "GPT 4.1" in self.model.name:
            return 0.1
        elif "GPT O4 Mini" in self.model.name:
            return 1.0
        else:
            return 0.1  # default temperature

    def complete(self, prompt: str) -> str:
        """Complete a text prompt using OpenAI."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.completions.create(
                model=self.model.id,
                prompt=prompt,
                max_tokens=2000,
                temperature=self.temperature
            )
            return response.choices[0].text.strip()
        except ImportError:
            raise LLMError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat completion using OpenAI."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model.id,
                messages=messages,
                max_tokens=2000,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            raise LLMError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            raise LLMError(f"OpenAI Chat API error: {str(e)}")