import streamlit as st
import uuid
import os
from datetime import datetime
from pathlib import Path

def initialize_session():
    """Initialize a new session with unique ID and default state"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    # Initialize session state variables
    session_defaults = {
        'current_stage': 'upload',
        'file_info': None,
        'original_data': None,
        'current_data': None,
        'detected_schema': None,
        'edited_schema': None,
        'final_schema': None,
        'analysis_results': None,
        'cleaned_data': None,
        'validation_results': None,
        'error_message': None,
        'session_start_time': datetime.now(),
        'processing_history': []
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Create session directories
    create_session_directories(st.session_state.session_id)

def create_session_directories(session_id: str):
    """Create necessary directories for the session"""
    base_path = Path(f"logs/{session_id}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    downloads_path = Path(f"downloads/{session_id}")
    downloads_path.mkdir(parents=True, exist_ok=True)

def get_session_info() -> dict:
    """Get current session information"""
    return {
        'session_id': st.session_state.session_id,
        'current_stage': st.session_state.current_stage,
        'start_time': st.session_state.session_start_time,
        'duration': datetime.now() - st.session_state.session_start_time
    }

def update_stage(new_stage: str):
    """Update current processing stage"""
    st.session_state.current_stage = new_stage
    st.session_state.processing_history.append({
        'stage': new_stage,
        'timestamp': datetime.now()
    })

def cleanup_session(session_id: str):
    """Clean up session files (optional, for storage management)"""
    import shutil
    
    logs_path = Path(f"logs/{session_id}")
    downloads_path = Path(f"downloads/{session_id}")
    
    if logs_path.exists():
        shutil.rmtree(logs_path)
    
    if downloads_path.exists():
        shutil.rmtree(downloads_path)
