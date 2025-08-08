import json, os
from typing import Optional
from core.models import SessionState
from core.config import STATE_DIR

def get_session_path(session_id: str) -> str:
    return os.path.join(STATE_DIR, f"{session_id}.json")

def save_session(state: SessionState):
    path = get_session_path(state.session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, default=lambda o: o.__dict__, indent=2)

def load_session(session_id: str) -> Optional[SessionState]:
    path = get_session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    state = SessionState(**d)
    return state
