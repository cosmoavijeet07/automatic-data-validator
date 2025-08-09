"""
Logger module for tracking session activities and data processing steps
"""

import json
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT
import logging

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
        
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "action": action,
            "level": level,
            "details": details or {}
        }
        
        # Add to memory logs
        self.logs.append(log_entry)
        
        # Log to Python logger
        log_message = f"{action} - {json.dumps(details, default=str) if details else ''}"
        getattr(self.python_logger, level.lower())(log_message)
        
        # Save to file
        self._save_to_file()
    
    def log_data_operation(self, operation: str, before_shape: tuple, after_shape: tuple, 
                          details: Dict[str, Any] = None):
        """Log a data operation with before/after shapes"""
        operation_details = {
            "operation": operation,
            "before_shape": before_shape,
            "after_shape": after_shape,
            "rows_changed": after_shape[0] - before_shape[0],
            "columns_changed": after_shape[1] - before_shape[1]
        }
        
        if details:
            operation_details.update(details)
        
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
            "context": context or {},
            "traceback": traceback
        }
        
        self.log(f"Error: {error_type}", error_details, "ERROR")
    
    def log_quality_analysis(self, quality_score: float, issues_found: List[str], 
                           analysis_time: float = None):
        """Log data quality analysis results"""
        quality_details = {
            "quality_score": quality_score,
            "issues_count": len(issues_found),
            "issues_found": issues_found,
            "analysis_time": analysis_time
        }
        
        self.log("Quality analysis completed", quality_details)
    
    def log_schema_change(self, column: str, old_type: str, new_type: str, 
                         change_reason: str = None):
        """Log schema changes"""
        schema_details = {
            "column": column,
            "old_type": old_type,
            "new_type": new_type,
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
            return json.dumps(self.logs, indent=2, default=str)
        
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
                json.dump(self.logs, f, indent=2, default=str)
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
                    execution_times.append(execution_time)
                    metrics["total_execution_time"] += execution_time
            
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