"""
Logger module for tracking session activities and data processing steps
"""

import json
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT
import logging
import numpy as np
import pandas as pd

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy, Pandas, and other special types"""
    def default(self, obj):
        # Handle basic Python types first
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle NumPy data types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.dtype):
            return str(obj)
        
        # Handle Pandas data types
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, pd.Categorical):
                return obj.tolist()
            elif 'pandas.core.dtypes.dtypes.CategoricalDtype' in str(type(obj)):
                return str(obj)
            elif hasattr(obj, 'dtype') and 'CategoricalDtype' in str(type(obj.dtype)):
                return str(obj.dtype)
        except NameError:
            pass  # Pandas not available
        
        # Handle datetime objects
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle any object with a name attribute
        if hasattr(obj, '__name__'):
            return obj.__name__
        
        # Convert sets to lists
        if isinstance(obj, set):
            return list(obj)
        
        # For any other type, try to convert to string
        try:
            return str(obj)
        except:
            return super().default(obj)


def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely dump data to JSON string handling various data types"""
    return json.dumps(data, cls=NumpyJSONEncoder, **kwargs)

def safe_dict_convert(data: Any) -> Any:
    """Recursively convert a dictionary to ensure all values are JSON serializable"""
    # Handle basic types first
    if data is None or isinstance(data, (str, int, float, bool)):
        return data
    
    # Handle containers
    if isinstance(data, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool)) else k: 
                safe_dict_convert(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [safe_dict_convert(item) for item in data]
    
    # Handle NumPy types
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.dtype):
        return str(data)
    
    # Handle Pandas types
    try:
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif isinstance(data, pd.Categorical):
            return data.tolist()
        elif 'pandas.core.dtypes.dtypes.CategoricalDtype' in str(type(data)):
            return str(data)
        elif hasattr(data, 'dtype') and 'CategoricalDtype' in str(type(data.dtype)):
            return str(data.dtype)
    except NameError:
        pass  # Pandas not available
    
    # Handle other special types
    if isinstance(data, (datetime.datetime, datetime.date)):
        return data.isoformat()
    elif isinstance(data, Path):
        return str(data)
    elif hasattr(data, '__name__'):
        return data.__name__
    elif isinstance(data, set):
        return list(data)
    
    # Final fallback
    try:
        return str(data)
    except:
        return None

class Logger:
    """Session logger for tracking data cleaning activities"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logs = []
        self.log_file_path = LOGS_DIR / f"session_{session_id}.json"
        
        # Setup Python logging
        self.python_logger = logging.getLogger(f"session_{session_id}")
        self.python_logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Create file handler if not exists
        if not self.python_logger.handlers:
            handler = logging.FileHandler(LOGS_DIR / f"session_{session_id}.log")
            handler.setLevel(getattr(logging, LOG_LEVEL))
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            self.python_logger.addHandler(handler)
        
        # Initialize session
        self.log("Session started", {"session_id": session_id})
    
    def log(self, action: str, details: Dict[str, Any] = None, level: str = "INFO"):
        """
        Log an action with details
        
        Args:
            action: Description of the action
            details: Additional details about the action
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # Convert details to ensure JSON serialization
        safe_details = safe_dict_convert(details) if details else {}
        
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "action": action,
            "level": level,
            "details": safe_details
        }
        
        # Add to memory logs
        self.logs.append(log_entry)
        
        # Log to Python logger
        log_message = f"{action} - {safe_json_dumps(safe_details) if safe_details else ''}"
        getattr(self.python_logger, level.lower())(log_message)
        
        # Save to file
        self._save_to_file()
    
    def log_data_operation(self, operation: str, before_shape: tuple, after_shape: tuple, 
                          details: Dict[str, Any] = None):
        """Log a data operation with before/after shapes"""
        operation_details = {
            "operation": operation,
            "before_shape": list(before_shape) if before_shape else None,
            "after_shape": list(after_shape) if after_shape else None,
            "rows_changed": after_shape[0] - before_shape[0] if before_shape and after_shape else 0,
            "columns_changed": after_shape[1] - before_shape[1] if before_shape and after_shape else 0
        }
        
        if details:
            safe_details = safe_dict_convert(details)
            operation_details.update(safe_details)
        
        self.log(f"Data operation: {operation}", operation_details)
    
    def log_llm_interaction(self, prompt_type: str, prompt: str, response: str, 
                           execution_time: float = None, model: str = None):
        """Log LLM interactions"""
        interaction_details = {
            "prompt_type": prompt_type,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "execution_time": execution_time,
            "model": model,
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response_preview": response[:200] + "..." if len(response) > 200 else response
        }
        
        self.log(f"LLM interaction: {prompt_type}", interaction_details)
    
    def log_code_execution(self, code: str, success: bool, execution_time: float = None, 
                          errors: List[str] = None, output: List[str] = None):
        """Log code execution results"""
        execution_details = {
            "code_length": len(code),
            "success": success,
            "execution_time": execution_time,
            "errors": errors or [],
            "output": output or [],
            "code_preview": code[:500] + "..." if len(code) > 500 else code
        }
        
        level = "INFO" if success else "ERROR"
        self.log("Code execution", execution_details, level)
    
    def log_file_operation(self, operation: str, filename: str, file_size: int = None, 
                          file_type: str = None, success: bool = True):
        """Log file operations"""
        file_details = {
            "operation": operation,
            "filename": filename,
            "file_size": file_size,
            "file_type": file_type,
            "success": success
        }
        
        level = "INFO" if success else "ERROR"
        self.log(f"File operation: {operation}", file_details, level)
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None, 
                  traceback: str = None):
        """Log errors with context"""
        error_details = {
            "error_type": error_type,
            "error_message": error_message,
            "context": safe_dict_convert(context) if context else {},
            "traceback": traceback
        }
        
        self.log(f"Error: {error_type}", error_details, "ERROR")
    
    def log_quality_analysis(self, quality_score: float, issues_found: List[str], 
                           analysis_time: float = None):
        """Log data quality analysis results"""
        quality_details = {
            "quality_score": float(quality_score) if quality_score else 0,
            "issues_count": len(issues_found),
            "issues_found": issues_found,
            "analysis_time": analysis_time
        }
        
        self.log("Quality analysis completed", quality_details)
    
    def log_schema_change(self, column: str, old_type: str, new_type: str, 
                         change_reason = None):
        """Log schema changes"""
        schema_details = {
            "column": column,
            "old_type": str(old_type),
            "new_type": str(new_type),
            "change_reason": change_reason
        }
        
        self.log(f"Schema change: {column}", schema_details)
    
    def get_logs(self, level: str = None, action_filter: str = None, 
                limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve logs with optional filtering
        
        Args:
            level: Filter by log level
            action_filter: Filter by action (contains)
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        filtered_logs = self.logs.copy()
        
        # Apply filters
        if level:
            filtered_logs = [log for log in filtered_logs if log.get('level') == level]
        
        if action_filter:
            filtered_logs = [log for log in filtered_logs if action_filter.lower() in log.get('action', '').lower()]
        
        # Apply limit
        if limit:
            filtered_logs = filtered_logs[-limit:]
        
        return filtered_logs
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the session activities"""
        if not self.logs:
            return {"error": "No logs available"}
        
        first_log = self.logs[0]
        last_log = self.logs[-1]
        
        # Calculate session duration
        start_time = datetime.datetime.fromisoformat(first_log['timestamp'])
        end_time = datetime.datetime.fromisoformat(last_log['timestamp'])
        duration = (end_time - start_time).total_seconds()
        
        # Count log levels
        level_counts = {}
        action_counts = {}
        
        for log in self.logs:
            level = log.get('level', 'INFO')
            level_counts[level] = level_counts.get(level, 0) + 1
            
            action = log.get('action', 'Unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Find data operations
        data_operations = [log for log in self.logs if 'Data operation' in log.get('action', '')]
        
        # Find errors
        errors = [log for log in self.logs if log.get('level') == 'ERROR']
        
        summary = {
            "session_id": self.session_id,
            "start_time": first_log['timestamp'],
            "end_time": last_log['timestamp'],
            "duration_seconds": duration,
            "total_logs": len(self.logs),
            "level_counts": level_counts,
            "action_counts": action_counts,
            "data_operations_count": len(data_operations),
            "errors_count": len(errors),
            "has_errors": len(errors) > 0
        }
        
        return summary
    
    def export_logs(self, format: str = "json", include_details: bool = True) -> str:
        """
        Export logs in specified format
        
        Args:
            format: Export format ('json', 'csv', 'txt')
            include_details: Whether to include detailed information
            
        Returns:
            Exported data as string
        """
        if format == "json":
            return safe_json_dumps(self.logs, indent=2)
        
        elif format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            
            if self.logs:
                # Get all possible fields
                fields = set()
                for log in self.logs:
                    fields.update(log.keys())
                    if include_details and 'details' in log:
                        for detail_key in log['details'].keys():
                            fields.add(f"detail_{detail_key}")
                
                fields = sorted(list(fields))
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                
                for log in self.logs:
                    row = log.copy()
                    if include_details and 'details' in log:
                        for key, value in log['details'].items():
                            row[f"detail_{key}"] = str(value)
                        del row['details']
                    writer.writerow(row)
            
            return output.getvalue()
        
        elif format == "txt":
            output_lines = []
            output_lines.append(f"Session Log Report - {self.session_id}")
            output_lines.append("=" * 50)
            output_lines.append("")
            
            for log in self.logs:
                output_lines.append(f"[{log['timestamp']}] {log['level']}: {log['action']}")
                
                if include_details and log.get('details'):
                    for key, value in log['details'].items():
                        output_lines.append(f"  {key}: {value}")
                
                output_lines.append("")
            
            return "\n".join(output_lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _save_to_file(self):
        """Save logs to file"""
        try:
            with open(self.log_file_path, 'w') as f:
                safe_json_dumps(self.logs, indent=2)
                f.write(safe_json_dumps(self.logs, indent=2))
        except Exception as e:
            self.python_logger.error(f"Failed to save logs to file: {str(e)}")
    
    def load_from_file(self) -> bool:
        """Load logs from file if exists"""
        try:
            if self.log_file_path.exists():
                with open(self.log_file_path, 'r') as f:
                    self.logs = json.load(f)
                return True
            return False
        except Exception as e:
            self.python_logger.error(f"Failed to load logs from file: {str(e)}")
            return False
    
    def clear_logs(self, confirm: bool = False):
        """Clear all logs (use with caution)"""
        if confirm:
            self.logs = []
            self._save_to_file()
            self.log("Logs cleared", {"action": "clear_logs"})
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from logs"""
        metrics = {
            "total_operations": 0,
            "average_execution_time": 0,
            "total_execution_time": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "llm_interactions": 0,
            "data_operations": 0
        }
        
        execution_times = []
        
        for log in self.logs:
            details = log.get('details', {})
            
            # Count operations
            if 'execution_time' in details:
                metrics["total_operations"] += 1
                execution_time = details['execution_time']
                if execution_time is not None:
                    execution_times.append(float(execution_time))
                    metrics["total_execution_time"] += float(execution_time)
            
            # Count success/failure
            if 'success' in details:
                if details['success']:
                    metrics["successful_operations"] += 1
                else:
                    metrics["failed_operations"] += 1
            
            # Count specific operation types
            if 'LLM interaction' in log.get('action', ''):
                metrics["llm_interactions"] += 1
            
            if 'Data operation' in log.get('action', ''):
                metrics["data_operations"] += 1
        
        # Calculate averages
        if execution_times:
            metrics["average_execution_time"] = sum(execution_times) / len(execution_times)
        
        return metrics