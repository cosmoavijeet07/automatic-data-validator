import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

def log(session_id: str, stage: str, content: str):
    """
    Log content to session-specific log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/{session_id}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        'timestamp': timestamp,
        'stage': stage,
        'content': content
    }
    
    log_file = log_dir / f"{stage}_{timestamp}.log"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2, default=str)
        else:
            f.write(str(content))

def get_recent_logs(session_id: str, max_entries: int = 10) -> str:
    """
    Get recent log entries for display
    """
    log_dir = Path(f"logs/{session_id}")
    
    if not log_dir.exists():
        return "No logs available for this session."
    
    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    recent_logs = []
    for log_file in log_files[:max_entries]:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                timestamp = log_file.stem.split('_', 1)[1] if '_' in log_file.stem else 'unknown'
                stage = log_file.stem.split('_')[0]
                recent_logs.append(f"[{timestamp}] {stage}: {content[:200]}...")
        except Exception:
            continue
    
    return "\n".join(recent_logs)

def get_session_logs(session_id: str) -> str:
    """
    Get all logs for a session as a single string
    """
    log_dir = Path(f"logs/{session_id}")
    
    if not log_dir.exists():
        return "No logs available for this session."
    
    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime)
    
    all_logs = []
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                timestamp = log_file.stem.split('_', 1)[1] if '_' in log_file.stem else 'unknown'
                stage = log_file.stem.split('_')[0]
                all_logs.append(f"=== {stage.upper()} - {timestamp} ===\n{content}\n")
        except Exception:
            continue
    
    return "\n".join(all_logs)

def clear_session_logs(session_id: str):
    """
    Clear all logs for a session
    """
    import shutil
    log_dir = Path(f"logs/{session_id}")
    
    if log_dir.exists():
        shutil.rmtree(log_dir)
