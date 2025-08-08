from typing import Dict, Any
from llm.client import LLMClient
from llm.prompts import build_code_prompt
from core.errors import LLMError

def generate_code(task: str, schema: dict, issues: dict | None, user_instructions: str | None, prev_errors: str | None, model_key: str) -> str:
    client = LLMClient(model_key)
    prompt = build_code_prompt(task, schema, issues, user_instructions, prev_errors)
    code = client.complete(prompt)
    if not isinstance(code, str) or len(code.strip()) == 0:
        raise LLMError("Empty code from LLM")
    return code
