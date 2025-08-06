import sys
import io
import contextlib
import traceback
from typing import Dict, Any

def safe_execute_code(code: str, namespace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Python code safely and capture output
    """
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Redirect output
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        # Execute code
        exec(code, namespace)
        
        # Capture any printed output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        # Add output to namespace
        namespace['_stdout'] = stdout_output
        namespace['_stderr'] = stderr_output
        
        return namespace
        
    except Exception as e:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue()
        }
        
        raise Exception(f"Code execution failed: {error_info}")
    
    finally:
        # Always restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
