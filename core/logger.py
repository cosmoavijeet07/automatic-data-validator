import time, os, json
from typing import Dict, Any, List
from core.config import LOG_DIR
from core.models import StepLog, SessionState

def start_step(state: SessionState, step_name: str) -> StepLog:
    step = StepLog(step=step_name, timestamp=time.time())
    state.logs.append(step)
    return step

def log_prompt(step: StepLog, prompt: Dict[str, Any]):
    step.prompts.append(prompt)

def log_response(step: StepLog, response: Dict[str, Any]):
    step.responses.append(response)

def log_code(step: StepLog, code: str):
    step.code.append(code)

def log_error(step: StepLog, err: str):
    step.errors.append(err)

def log_summary(step: StepLog, summary: str):
    step.result_summary = summary

def flush_logs(state: SessionState):
    path = os.path.join(LOG_DIR, f"{state.session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in state.logs], f, indent=2)
