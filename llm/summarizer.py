from llm.client import LLMClient
from llm.prompts import build_summary_prompt
from core.config import SUMMARY_WORD_LIMIT

def summarize_strategy(text: str, model_key: str) -> str:
    client = LLMClient(model_key)
    prompt = build_summary_prompt(text, SUMMARY_WORD_LIMIT)
    return client.complete(prompt)
