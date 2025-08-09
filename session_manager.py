"""
Session Management Module for handling user sessions and state persistence
"""
import json
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from config import TEMP_DIR, SESSION_TIMEOUT
from utils import generate_session_id, save_json, load_json

class SessionManager:
    def __init__(self):
        """Initialize session manager"""
        self.sessions = {}
        self.session_dir = TEMP_DIR / "sessions"
        self.session_dir.mkdir(exist_ok=True)
        self._load_existing_sessions()
    
    def create_session(self, user_id: str = None) -> str:
        """Create a new session"""
        session_id = generate_session_id()
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'state': {
                'current_step': 'file_upload',
                'processing_mode': 'manual',
                'model_selected': 'gpt-4-turbo-preview',
                'file_uploaded': False,
                'schema_detected': False,
                'quality_analyzed': False,
                'corrections_applied': False,
                'pipeline_generated': False
            },
            'data': {
                'file_path': None,
                'file_type': None,
                'original_shape': None,
                'current_shape': None
            },
            'logs': [],
            'code_history': [],
            'execution_logs': []
        }
        
        self.sessions[session_id] = session_data
        self._save_session(session_id)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if session_id in self.sessions:
            # Update last accessed time
            self.sessions[session_id]['last_accessed'] = datetime.now().isoformat()
            self._save_session(session_id)
            return self.sessions[session_id]
        
        # Try to load from disk
        session_file = self.session_dir / f"{session_id}.json"
        if session_file.exists():
            session_data = load_json(session_file)
            self.sessions[session_id] = session_data
            return session_data
        
        return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Deep update
        for key, value in updates.items():
            if isinstance(value, dict) and key in session:
                session[key].update(value)
            else:
                session[key] = value
        
        session['last_accessed'] = datetime.now().isoformat()
        self.sessions[session_id] = session
        self._save_session(session_id)
        
        return True
    
    def update_session_state(self, session_id: str, state_updates: Dict[str, Any]) -> bool:
        """Update session state"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session['state'].update(state_updates)
        return self.update_session(session_id, {'state': session['state']})
    
    def add_log(self, session_id: str, log_entry: Dict[str, Any]) -> bool:
        """Add log entry to session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session['logs'].append(log_entry)
        return self.update_session(session_id, {'logs': session['logs']})
    
    def add_code_to_history(self, session_id: str, code: str) -> bool:
        """Add code to session history"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session['code_history'].append({
            'timestamp': datetime.now().isoformat(),
            'code': code
        })
        return self.update_session(session_id, {'code_history': session['code_history']})
    
    def save_dataframe(self, session_id: str, df: pd.DataFrame, name: str = 'current') -> bool:
        """Save DataFrame to session storage"""
        try:
            df_file = self.session_dir / f"{session_id}_{name}.pkl"
            df.to_pickle(df_file)
            
            # Update session data
            session = self.get_session(session_id)
            if session:
                if 'dataframes' not in session['data']:
                    session['data']['dataframes'] = {}
                session['data']['dataframes'][name] = str(df_file)
                self.update_session(session_id, {'data': session['data']})
            
            return True
        except Exception as e:
            logging.error(f"Error saving dataframe: {e}")
            return False
    
    def load_dataframe(self, session_id: str, name: str = 'current') -> Optional[pd.DataFrame]:
        """Load DataFrame from session storage"""
        try:
            session = self.get_session(session_id)
            if not session:
                return None
            
            if 'dataframes' in session['data'] and name in session['data']['dataframes']:
                df_file = Path(session['data']['dataframes'][name])
                if df_file.exists():
                    return pd.read_pickle(df_file)
            
            # Try direct file
            df_file = self.session_dir / f"{session_id}_{name}.pkl"
            if df_file.exists():
                return pd.read_pickle(df_file)
            
            return None
        except Exception as e:
            logging.error(f"Error loading dataframe: {e}")
            return None
    
    def cleanup_old_sessions(self) -> int:
        """Clean up old sessions"""
        cleaned = 0
        current_time = datetime.now()
        
        for session_id in list(self.sessions.keys()):
            session = self.sessions[session_id]
            last_accessed = datetime.fromisoformat(session['last_accessed'])
            
            if (current_time - last_accessed).total_seconds() > SESSION_TIMEOUT:
                self._delete_session(session_id)
                cleaned += 1
        
        return cleaned
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self._delete_session(session_id)
    
    def _save_session(self, session_id: str) -> bool:
        """Save session to disk"""
        try:
            if session_id not in self.sessions:
                return False
            
            session_file = self.session_dir / f"{session_id}.json"
            save_json(self.sessions[session_id], session_file)
            return True
        except Exception as e:
            logging.error(f"Error saving session: {e}")
            return False
    
    def _load_existing_sessions(self) -> None:
        """Load existing sessions from disk"""
        try:
            for session_file in self.session_dir.glob("*.json"):
                try:
                    session_data = load_json(session_file)
                    session_id = session_data.get('session_id')
                    if session_id:
                        self.sessions[session_id] = session_data
                except Exception as e:
                    logging.error(f"Error loading session file {session_file}: {e}")
        except Exception as e:
            logging.error(f"Error loading sessions: {e}")
    
    def _delete_session(self, session_id: str) -> bool:
        """Delete session and associated files"""
        try:
            # Remove from memory
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            # Delete session file
            session_file = self.session_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Delete associated dataframes
            for df_file in self.session_dir.glob(f"{session_id}_*.pkl"):
                df_file.unlink()
            
            return True
        except Exception as e:
            logging.error(f"Error deleting session: {e}")
            return False
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session summary"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session_id,
            'created_at': session['created_at'],
            'last_accessed': session['last_accessed'],
            'current_step': session['state']['current_step'],
            'processing_mode': session['state']['processing_mode'],
            'file_uploaded': session['state']['file_uploaded'],
            'num_logs': len(session['logs']),
            'num_code_history': len(session['code_history'])
        }
    
    def export_session_logs(self, session_id: str, file_path: str) -> bool:
        """Export session logs to file"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            logs_data = {
                'session_id': session_id,
                'created_at': session['created_at'],
                'exported_at': datetime.now().isoformat(),
                'logs': session['logs'],
                'code_history': session['code_history'],
                'execution_logs': session.get('execution_logs', [])
            }
            
            save_json(logs_data, file_path)
            return True
        except Exception as e:
            logging.error(f"Error exporting logs: {e}")
            return False