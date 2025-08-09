"""
Session manager module for handling user sessions and state persistence
"""

import json
import pickle
import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from config import TEMP_DIR, SESSION_TIMEOUT_HOURS, MAX_SESSIONS

class SessionManager:
    """Manages user sessions and data persistence"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = TEMP_DIR / f"session_{session_id}"
        self.session_file = self.session_dir / "session_data.json"
        self.data_file = self.session_dir / "session_data.pkl"
        
        # Create session directory
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize session data
        self.session_data = {
            "session_id": session_id,
            "created_at": datetime.datetime.now().isoformat(),
            "last_accessed": datetime.datetime.now().isoformat(),
            "current_step": "upload",
            "processing_mode": "manual",
            "selected_model": "GPT 4.1",
            "file_info": None,
            "schema": None,
            "quality_report": None,
            "correction_code": None,
            "pipeline_code": None,
            "metadata": {}
        }
        
        # Try to load existing session
        self.load_session()
    
    def save_session(self, data: Dict[str, Any] = None):
        """
        Save session data to file
        
        Args:
            data: Additional data to save
        """
        try:
            # Update last accessed time
            self.session_data["last_accessed"] = datetime.datetime.now().isoformat()
            
            # Update with provided data
            if data:
                self.session_data.update(data)
            
            # Save session metadata
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
            
        except Exception as e:
            print(f"Error saving session: {str(e)}")
    
    def load_session(self) -> bool:
        """
        Load session data from file
        
        Returns:
            True if session was loaded successfully, False otherwise
        """
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    loaded_data = json.load(f)
                
                # Check if session is not expired
                last_accessed = datetime.datetime.fromisoformat(loaded_data.get("last_accessed"))
                hours_since_access = (datetime.datetime.now() - last_accessed).total_seconds() / 3600
                
                if hours_since_access < SESSION_TIMEOUT_HOURS:
                    self.session_data.update(loaded_data)
                    return True
                else:
                    # Session expired, clean up
                    self.cleanup_session()
                    return False
            
            return False
            
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            return False
    
    def save_dataframe(self, df: pd.DataFrame, name: str = "main_data"):
        """
        Save DataFrame to session
        
        Args:
            df: DataFrame to save
            name: Name identifier for the data
        """
        try:
            data_path = self.session_dir / f"{name}.pkl"
            df.to_pickle(data_path)
            
            # Update session metadata
            self.session_data["metadata"][name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "saved_at": datetime.datetime.now().isoformat()
            }
            
            self.save_session()
            
        except Exception as e:
            print(f"Error saving DataFrame {name}: {str(e)}")
    
    def load_dataframe(self, name: str = "main_data") -> Optional[pd.DataFrame]:
        """
        Load DataFrame from session
        
        Args:
            name: Name identifier for the data
            
        Returns:
            DataFrame if found, None otherwise
        """
        try:
            data_path = self.session_dir / f"{name}.pkl"
            if data_path.exists():
                return pd.read_pickle(data_path)
            return None
            
        except Exception as e:
            print(f"Error loading DataFrame {name}: {str(e)}")
            return None
    
    def save_object(self, obj: Any, name: str):
        """
        Save any Python object to session
        
        Args:
            obj: Object to save
            name: Name identifier for the object
        """
        try:
            object_path = self.session_dir / f"{name}.pkl"
            with open(object_path, 'wb') as f:
                pickle.dump(obj, f)
            
            # Update metadata
            self.session_data["metadata"][name] = {
                "type": str(type(obj)),
                "saved_at": datetime.datetime.now().isoformat()
            }
            
            self.save_session()
            
        except Exception as e:
            print(f"Error saving object {name}: {str(e)}")
    
    def load_object(self, name: str) -> Any:
        """
        Load Python object from session
        
        Args:
            name: Name identifier for the object
            
        Returns:
            Object if found, None otherwise
        """
        try:
            object_path = self.session_dir / f"{name}.pkl"
            if object_path.exists():
                with open(object_path, 'rb') as f:
                    return pickle.load(f)
            return None
            
        except Exception as e:
            print(f"Error loading object {name}: {str(e)}")
            return None
    
    def update_session_state(self, **kwargs):
        """Update session state with key-value pairs"""
        for key, value in kwargs.items():
            self.session_data[key] = value
        self.save_session()
    
    def get_session_state(self, key: str = None):
        """Get session state value or entire state"""
        if key:
            return self.session_data.get(key)
        return self.session_data.copy()
    
    def cleanup_session(self):
        """Clean up session files and data"""
        try:
            import shutil
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir)
                
        except Exception as e:
            print(f"Error cleaning up session: {str(e)}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information and statistics"""
        info = {
            "session_id": self.session_id,
            "created_at": self.session_data["created_at"],
            "last_accessed": self.session_data["last_accessed"],
            "current_step": self.session_data["current_step"],
            "processing_mode": self.session_data["processing_mode"],
            "selected_model": self.session_data["selected_model"],
            "session_age_hours": self._get_session_age(),
            "data_files": self._get_data_files_info(),
            "session_size_mb": self._get_session_size()
        }
        
        return info
    
    def _get_session_age(self) -> float:
        """Calculate session age in hours"""
        created_at = datetime.datetime.fromisoformat(self.session_data["created_at"])
        return (datetime.datetime.now() - created_at).total_seconds() / 3600
    
    def _get_data_files_info(self) -> List[Dict[str, Any]]:
        """Get information about saved data files"""
        files_info = []
        
        if self.session_dir.exists():
            for file_path in self.session_dir.glob("*.pkl"):
                try:
                    file_stats = file_path.stat()
                    files_info.append({
                        "name": file_path.stem,
                        "size_mb": file_stats.st_size / (1024 * 1024),
                        "modified": datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    })
                except:
                    continue
        
        return files_info
    
    def _get_session_size(self) -> float:
        """Calculate total session size in MB"""
        total_size = 0
        
        if self.session_dir.exists():
            for file_path in self.session_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                    except:
                        continue
        
        return total_size / (1024 * 1024)
    
    def backup_session(self) -> str:
        """Create a backup of the current session"""
        try:
            import shutil
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"session_{self.session_id}_backup_{timestamp}"
            backup_path = TEMP_DIR / backup_name
            
            if self.session_dir.exists():
                shutil.copytree(self.session_dir, backup_path)
                return str(backup_path)
            
            return ""
            
        except Exception as e:
            print(f"Error creating session backup: {str(e)}")
            return ""
    
    def restore_session(self, backup_path: str) -> bool:
        """Restore session from backup"""
        try:
            import shutil
            backup_dir = Path(backup_path)
            
            if backup_dir.exists():
                # Clean current session
                self.cleanup_session()
                
                # Restore from backup
                shutil.copytree(backup_dir, self.session_dir)
                
                # Reload session data
                self.load_session()
                return True
            
            return False
            
        except Exception as e:
            print(f"Error restoring session: {str(e)}")
            return False
    
    @staticmethod
    def list_active_sessions() -> List[Dict[str, Any]]:
        """List all active sessions"""
        sessions = []
        
        try:
            for session_dir in TEMP_DIR.glob("session_*"):
                if session_dir.is_dir():
                    session_file = session_dir / "session_data.json"
                    
                    if session_file.exists():
                        try:
                            with open(session_file, 'r') as f:
                                session_data = json.load(f)
                            
                            # Check if session is not expired
                            last_accessed = datetime.datetime.fromisoformat(session_data.get("last_accessed"))
                            hours_since_access = (datetime.datetime.now() - last_accessed).total_seconds() / 3600
                            
                            if hours_since_access < SESSION_TIMEOUT_HOURS:
                                sessions.append({
                                    "session_id": session_data["session_id"],
                                    "created_at": session_data["created_at"],
                                    "last_accessed": session_data["last_accessed"],
                                    "current_step": session_data["current_step"],
                                    "age_hours": hours_since_access
                                })
                            
                        except:
                            continue
            
        except Exception as e:
            print(f"Error listing sessions: {str(e)}")
        
        return sessions
    
    @staticmethod
    def cleanup_expired_sessions():
        """Clean up expired sessions"""
        try:
            active_sessions = SessionManager.list_active_sessions()
            all_session_dirs = list(TEMP_DIR.glob("session_*"))
            
            # Get active session IDs
            active_ids = {session["session_id"] for session in active_sessions}
            
            # Clean up inactive sessions
            for session_dir in all_session_dirs:
                session_id = session_dir.name.replace("session_", "")
                
                if session_id not in active_ids:
                    try:
                        import shutil
                        shutil.rmtree(session_dir)
                        print(f"Cleaned up expired session: {session_id}")
                    except:
                        continue
                        
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    @staticmethod
    def get_session_statistics() -> Dict[str, Any]:
        """Get overall session statistics"""
        stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_size_mb": 0,
            "oldest_session": None,
            "newest_session": None
        }
        
        try:
            active_sessions = SessionManager.list_active_sessions()
            all_session_dirs = list(TEMP_DIR.glob("session_*"))
            
            stats["total_sessions"] = len(all_session_dirs)
            stats["active_sessions"] = len(active_sessions)
            
            # Calculate total size
            for session_dir in all_session_dirs:
                try:
                    session_size = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())
                    stats["total_size_mb"] += session_size / (1024 * 1024)
                except:
                    continue
            
            # Find oldest and newest active sessions
            if active_sessions:
                sorted_sessions = sorted(active_sessions, key=lambda x: x["created_at"])
                stats["oldest_session"] = sorted_sessions[0]
                stats["newest_session"] = sorted_sessions[-1]
            
        except Exception as e:
            print(f"Error getting session statistics: {str(e)}")
        
        return stats
    
    def export_session_data(self, format: str = "json") -> str:
        """Export session data in specified format"""
        try:
            export_data = {
                "session_info": self.get_session_info(),
                "session_data": self.session_data,
                "metadata": self.session_data.get("metadata", {})
            }
            
            if format == "json":
                return json.dumps(export_data, indent=2, default=str)
            
            elif format == "summary":
                summary = f"""
Session Export Summary
=====================
Session ID: {self.session_id}
Created: {self.session_data['created_at']}
Last Accessed: {self.session_data['last_accessed']}
Current Step: {self.session_data['current_step']}
Processing Mode: {self.session_data['processing_mode']}
Selected Model: {self.session_data['selected_model']}
Session Size: {self._get_session_size():.2f} MB

Data Files:
{chr(10).join([f"- {f['name']}: {f['size_mb']:.2f} MB" for f in self._get_data_files_info()])}
"""
                return summary
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            return f"Error exporting session data: {str(e)}"