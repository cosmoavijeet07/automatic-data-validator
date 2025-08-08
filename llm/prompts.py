import json

def build_code_prompt(task: str, schema: dict, issues: dict | None, user_instructions: str | None, prev_errors: str | None) -> str:
    ctx = {
        "task": task,
        "schema": schema,
        "issues": issues,
        "user_instructions": user_instructions,
        "previous_errors": prev_errors,
        "requirements": [
            "Output only Python code.",
            "Use pandas operations.",
            "Avoid file IO; operate on provided df variable.",
        ],
    }
    return "Generate Python code for data operations based on context:\n" + json.dumps(ctx, indent=2)

def build_summary_prompt(strategy_desc: str, word_limit: int) -> str:
    return f"Summarize the following strategy in under {word_limit} words, plain language:\n{strategy_desc}"
