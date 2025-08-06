import openai
import anthropic
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import time
import streamlit as st

load_dotenv()

class LLMInterface:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        self.model_configs = {
            "gpt-4.1-2025-04-14": {
                "provider": "openai",
                "supports_temperature": True,
                "max_tokens": 4096,
                "temperature": 0.1
            },
            "o4-mini-2025-04-16": {
                "provider": "openai", 
                "supports_temperature": True,
                "max_tokens": 16384,
                "temperature": 0.1
            },
            "claude-sonnet-4-20250514": {
                "provider": "anthropic",
                "supports_temperature": True,
                "max_tokens": 4096,
                "temperature": 0.1
            }
        }
    
    def call_llm(self, prompt: str, model: str, temperature: Optional[float] = None, max_retries: int = 3) -> str:
        """
        Universal LLM calling function with retry logic
        """
        config = self.model_configs.get(model)
        if not config:
            raise ValueError(f"Unsupported model: {model}")
        
        if temperature is None:
            temperature = config["temperature"]
        
        for attempt in range(max_retries):
            try:
                if config["provider"] == "openai":
                    response = self._call_openai(prompt, model, temperature, config["max_tokens"])
                elif config["provider"] == "anthropic":
                    response = self._call_anthropic(prompt, model, temperature, config["max_tokens"])
                else:
                    raise ValueError(f"Unknown provider: {config['provider']}")
                
                return response.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
    
    def _call_openai(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        response = self.anthropic_client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text

# Global instance
llm_interface = LLMInterface()

def call_llm(prompt: str, model: str, temperature: Optional[float] = None) -> str:
    """Convenience function for LLM calls"""
    return llm_interface.call_llm(prompt, model, temperature)
