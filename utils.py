"""
Utility functions for the Data Cleaning Application
"""
import os
import json
import uuid
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def get_file_hash(file_path: str) -> str:
    """Calculate file hash for caching"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def validate_file_size(file_path: str, max_size_mb: int = 100) -> bool:
    """Validate file size"""
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    return file_size <= max_size_mb

def detect_delimiter(file_path: str) -> str:
    """Detect CSV delimiter"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {d: first_line.count(d) for d in delimiters}
        return max(delimiter_counts, key=delimiter_counts.get)

def detect_encoding(file_path: str) -> str:
    """Detect file encoding"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1000)
            return encoding
        except:
            continue
    return 'utf-8'

def format_timestamp() -> str:
    """Get formatted timestamp"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_log_entry(
    step: str,
    action: str,
    details: Dict[str, Any],
    status: str = "success"
) -> Dict[str, Any]:
    """Create a structured log entry"""
    return {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "action": action,
        "status": status,
        "details": details
    }

def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(file_path: str) -> Any:
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def safe_convert_dtype(series: pd.Series, target_dtype: str) -> pd.Series:
    """Safely convert series to target data type"""
    try:
        if target_dtype == 'datetime64':
            return pd.to_datetime(series, errors='coerce')
        elif target_dtype == 'int64':
            return pd.to_numeric(series, errors='coerce').fillna(0).astype('int64')
        elif target_dtype == 'float64':
            return pd.to_numeric(series, errors='coerce')
        elif target_dtype == 'bool':
            return series.astype('bool')
        elif target_dtype == 'category':
            return series.astype('category')
        else:
            return series.astype(target_dtype)
    except Exception as e:
        logging.error(f"Error converting dtype: {e}")
        return series

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Automatically detect column types"""
    type_mapping = {}
    
    for col in df.columns:
        series = df[col]
        
        # Check for datetime
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.dropna().head(100))
                type_mapping[col] = 'datetime64'
                continue
            except:
                pass
        
        # Check for numeric
        if series.dtype in ['int64', 'float64']:
            type_mapping[col] = str(series.dtype)
        elif series.dtype == 'object':
            try:
                pd.to_numeric(series.dropna())
                if '.' in str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else False:
                    type_mapping[col] = 'float64'
                else:
                    type_mapping[col] = 'int64'
            except:
                # Check for categorical
                unique_ratio = len(series.unique()) / len(series)
                if unique_ratio < 0.05:  # Less than 5% unique values
                    type_mapping[col] = 'category'
                else:
                    type_mapping[col] = 'object'
        else:
            type_mapping[col] = str(series.dtype)
    
    return type_mapping

def detect_missing_patterns(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """Detect patterns representing missing values"""
    missing_patterns = {}
    
    for col in df.columns:
        patterns = []
        series = df[col]
        
        # Check for common missing patterns in object columns
        if series.dtype == 'object':
            value_counts = series.value_counts()
            for pattern in ['?', 'NA', 'n/a', 'N/A', 'missing', 'Missing', 'MISSING', 
                          'unknown', 'Unknown', 'UNKNOWN', '-', '--', '']:
                if pattern in value_counts.index:
                    patterns.append(pattern)
        
        if patterns:
            missing_patterns[col] = patterns
    
    return missing_patterns

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data quality metrics"""
    metrics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "duplicate_rows": len(df[df.duplicated()]),
        "duplicate_percentage": (len(df[df.duplicated()]) / len(df)) * 100,
        "column_metrics": {}
    }
    
    for col in df.columns:
        col_metrics = {
            "dtype": str(df[col].dtype),
            "missing_count": df[col].isnull().sum(),
            "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
            "unique_values": df[col].nunique(),
            "unique_percentage": (df[col].nunique() / len(df)) * 100
        }
        
        # Additional metrics for numeric columns
        if df[col].dtype in ['int64', 'float64']:
            col_metrics.update({
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "q1": df[col].quantile(0.25),
                "q3": df[col].quantile(0.75)
            })
            
            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            col_metrics["outlier_count"] = len(outliers)
            col_metrics["outlier_percentage"] = (len(outliers) / len(df)) * 100
        
        metrics["column_metrics"][col] = col_metrics
        metrics["missing_values"][col] = col_metrics["missing_count"]
    
    return metrics

def format_code_for_display(code: str) -> str:
    """Format code for display with proper indentation"""
    lines = code.strip().split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip():
            formatted_lines.append(line)
    return '\n'.join(formatted_lines)

def create_pipeline_code(code_history: List[str]) -> str:
    """Combine all code into a single pipeline script"""
    pipeline = """#!/usr/bin/env python3
\"\"\"
Auto-generated Data Cleaning Pipeline
Generated on: {timestamp}
\"\"\"

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def clean_data(input_path, output_path=None):
    \"\"\"
    Clean data using the generated pipeline
    
    Args:
        input_path: Path to input data file
        output_path: Path to save cleaned data (optional)
    
    Returns:
        Cleaned DataFrame
    \"\"\"
    # Load data
    file_ext = Path(input_path).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(input_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(input_path)
    elif file_ext == '.json':
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {{file_ext}}")
    
    print(f"Loaded data with shape: {{df.shape}}")
    
    # Apply cleaning steps
{cleaning_steps}
    
    # Save cleaned data
    if output_path:
        output_ext = Path(output_path).suffix.lower()
        if output_ext == '.csv':
            df.to_csv(output_path, index=False)
        elif output_ext in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=False)
        elif output_ext == '.json':
            df.to_json(output_path, orient='records')
        print(f"Saved cleaned data to: {{output_path}}")
    
    print(f"Final data shape: {{df.shape}}")
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    cleaned_df = clean_data(input_file, output_file)
    print("Data cleaning completed successfully!")
"""
    
    # Combine and indent cleaning steps
    cleaning_steps = []
    for i, code in enumerate(code_history, 1):
        cleaning_steps.append(f"    # Step {i}")
        for line in code.strip().split('\n'):
            if line.strip() and not line.strip().startswith('#'):
                cleaning_steps.append(f"    {line}")
        cleaning_steps.append("")
    
    return pipeline.format(
        timestamp=format_timestamp(),
        cleaning_steps='\n'.join(cleaning_steps)
    )