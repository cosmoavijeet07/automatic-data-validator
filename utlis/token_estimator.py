import tiktoken
from typing import Dict, Any

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for given text and model
    """
    try:
        # Map model names to tiktoken encodings
        model_mapping = {
            "gpt-4.1-2025-04-14": "cl100k_base",
            "o4-mini-2025-04-16": "cl100k_base",
            "claude-sonnet-4-20250514": "cl100k_base"  # Approximation
        }
        
        encoding_name = model_mapping.get(model, "cl100k_base")
        enc = tiktoken.get_encoding(encoding_name)
        
        return len(enc.encode(text))
        
    except Exception:
        # Fallback: rough estimation (4 chars per token)
        return len(text) // 4

def estimate_cost(token_count: int, model: str) -> float:
    """
    Estimate cost based on token count and model
    """
    # Approximate pricing (as of knowledge cutoff)
    pricing = {
        "gpt-4.1-2025-04-14": 0.03,  # per 1k tokens
        "o4-mini-2025-04-16": 0.0015,  # per 1k tokens
        "claude-sonnet-4-20250514": 0.015  # per 1k tokens
    }
    
    rate = pricing.get(model, 0.02)
    return (token_count / 1000) * rate

def check_token_limits(text: str, model: str) -> Dict[str, Any]:
    """
    Check if text exceeds model token limits
    """
    limits = {
        "gpt-4.1-2025-04-14": 8192,
        "o4-mini-2025-04-16": 128000,
        "claude-sonnet-4-20250514": 200000
    }
    
    token_count = estimate_tokens(text, model)
    limit = limits.get(model, 4096)
    
    return {
        'token_count': token_count,
        'limit': limit,
        'within_limit': token_count <= limit,
        'usage_percentage': (token_count / limit) * 100
    }
