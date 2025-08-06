import sys
import io
import contextlib
import traceback
import re
from typing import Dict, Any

def clean_code(code: str) -> str:
    """
    Clean code by removing markdown formatting and extracting pure Python
    """
    # Remove code block delimiters like ```python or ```
    code = re.sub(r'^```(?:python|py)?\s*', '', code, flags=re.IGNORECASE | re.MULTILINE)
    code = code.strip('`')

    # Remove markdown headers or bold text lines
    code = re.sub(r'^#+.*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\*\*.*\*\*\s*\n', '', code, flags=re.MULTILINE)

    # Split lines and remove standalone language identifiers
    lines = code.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip().lower()

        # Skip if line is a language identifier or markdown artifact
        if stripped in ['python', 'py', 'sql', 'json', 'bash', 'sh', 'yaml', 'javascript', 'js', '```']:
            continue

        # Skip empty lines at the beginning
        if not cleaned_lines and not stripped:
            continue

        cleaned_lines.append(line)

    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines).strip()

def validate_python_syntax(code: str) -> tuple[bool, str]:
    """
    Validate that the cleaned code has valid Python syntax
    """
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax Error: {str(e)} at line {e.lineno}"
    except Exception as e:
        return False, f"Compilation Error: {str(e)}"

def safe_execute_code(code: str, namespace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Python code safely and capture output with comprehensive error handling
    """
    clean_python_code = clean_code(code)

    is_valid, syntax_error = validate_python_syntax(clean_python_code)
    if not is_valid:
        raise Exception(f"Code syntax validation failed: {syntax_error}")

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    safe_namespace = namespace.copy()

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        exec(clean_python_code, safe_namespace)

        safe_namespace['_stdout'] = stdout_capture.getvalue()
        safe_namespace['_stderr'] = stderr_capture.getvalue()
        safe_namespace['_execution_success'] = True
        safe_namespace['_cleaned_code'] = clean_python_code

        return safe_namespace

    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        error_info = {
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue(),
            'cleaned_code': clean_python_code,
            'original_code': code[:500] + "..." if len(code) > 500 else code
        }

        raise Exception(f"Code execution failed: {error_info}")

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def execute_with_retry(code: str, namespace: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """
    Execute code with automatic retry and progressive cleaning
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            if attempt == 0:
                return safe_execute_code(code, namespace)
            else:
                cleaned_code = clean_code(code)
                return safe_execute_code(cleaned_code, namespace)
        except Exception as e:
            last_error = e
            continue

    raise last_error

def extract_variable(result: Dict[str, Any], variable_names: list) -> Any:
    """
    Extract a specific variable from execution results, trying multiple possible names
    """
    for var_name in variable_names:
        if var_name in result:
            return result[var_name]
    return None

def log_execution_details(code: str, result: Dict[str, Any], session_id: str = None):
    """
    Log execution details for debugging purposes
    """
    if session_id:
        try:
            from .logger_service import log
            log_data = {
                'original_code_length': len(code),
                'cleaned_code_length': len(result.get('_cleaned_code', '')),
                'execution_success': result.get('_execution_success', False),
                'stdout': result.get('_stdout', ''),
                'stderr': result.get('_stderr', ''),
                'variables_created': [k for k in result.keys() if not k.startswith('_')]
            }
            log(session_id, "code_execution", str(log_data))
        except ImportError:
            pass

# --- Context-specific execution helpers ---

def execute_schema_code(code: str, dataframes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute schema detection code with predefined namespace
    """
    import pandas as pd
    import json
    import numpy as np

    namespace = {
        'pd': pd,
        'json': json,
        'np': np,
        'df_dict': dataframes
    }

    result = execute_with_retry(code, namespace)
    schema_info = extract_variable(result, ['schema_info', 'schema', 'detected_schema'])

    if schema_info is None:
        raise Exception("Schema detection code did not produce expected output variable")

    return schema_info

def execute_analysis_code(code: str, dataframes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute data analysis code with predefined namespace
    """
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    namespace = {
        'pd': pd,
        'np': np,
        'px': px,
        'go': go,
        'df_dict': dataframes
    }

    result = execute_with_retry(code, namespace)
    analysis_results = extract_variable(result, ['analysis_results', 'report', 'analysis_report'])

    if analysis_results is None:
        raise Exception("Analysis code did not produce expected output variable")

    return analysis_results

def execute_cleaning_code(code: str, dataframes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute data cleaning code with predefined namespace
    """
    import pandas as pd
    import numpy as np

    namespace = {
        'pd': pd,
        'np': np,
        'df_dict': dataframes.copy()
    }

    result = execute_with_retry(code, namespace)
    cleaned_data = extract_variable(result, ['cleaned_dataframes', 'clean_df_dict', 'cleaned_data', 'df_dict'])

    if cleaned_data is None:
        raise Exception("Cleaning code did not produce expected output variable")

    if hasattr(cleaned_data, 'columns'):
        cleaned_data = {'Sheet1': cleaned_data}

    return cleaned_data
