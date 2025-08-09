"""
Data Processing Module with optional profiling libraries
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import duckdb
from io import StringIO
import sys
import traceback

# Try to import profiling libraries (optional)
try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False
    logging.warning("ydata-profiling not available - using basic profiling")

try:
    import sweetviz as sv
    SWEETVIZ_AVAILABLE = True
except ImportError:
    SWEETVIZ_AVAILABLE = False
    logging.warning("sweetviz not available - using basic profiling")

from utils import (
    detect_delimiter, detect_encoding, detect_column_types,
    safe_convert_dtype, detect_missing_patterns,
    calculate_data_quality_metrics
)
from config import MISSING_PATTERNS, QUALITY_THRESHOLDS

class DataProcessor:
    def __init__(self, session_id: str):
        """Initialize data processor"""
        self.session_id = session_id
        self.df = None
        self.original_df = None
        self.metadata = {}
        self.schema = {}
        self.code_history = []
        self.execution_logs = []
        
    def load_data(self, file_path: str, file_type: str) -> Tuple[bool, str]:
        """Load data from file"""
        try:
            if file_type == 'csv':
                delimiter = detect_delimiter(file_path)
                encoding = detect_encoding(file_path)
                self.df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                
            elif file_type in ['xlsx', 'xls']:
                # Load Excel file - could have multiple sheets
                excel_file = pd.ExcelFile(file_path)
                if len(excel_file.sheet_names) > 1:
                    # For now, load first sheet. Can be extended for multi-sheet
                    self.df = pd.read_excel(file_path, sheet_name=0)
                    self.metadata['sheets'] = excel_file.sheet_names
                else:
                    self.df = pd.read_excel(file_path)
                    
            elif file_type == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    self.df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to create DataFrame from dict
                    try:
                        self.df = pd.DataFrame(data)
                    except:
                        # If direct conversion fails, try json_normalize
                        self.df = pd.json_normalize(data)
                        
            elif file_type == 'txt':
                # For text files, create a DataFrame with text content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
                self.df = pd.DataFrame({'text': [text_content]})
                self.metadata['text_file'] = True
                
            else:
                return False, f"Unsupported file type: {file_type}"
            
            # Store original for comparison
            self.original_df = self.df.copy()
            
            # Detect initial schema
            self.schema = detect_column_types(self.df)
            
            # Store metadata
            self.metadata.update({
                'file_path': file_path,
                'file_type': file_type,
                'original_shape': self.df.shape,
                'columns': list(self.df.columns)
            })
            
            return True, "Data loaded successfully"
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False, str(e)
    
    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        """Get data preview"""
        if self.df is not None:
            return self.df.head(n_rows)
        return pd.DataFrame()
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get detailed schema information"""
        if self.df is None:
            return {}
        
        schema_info = {}
        for col in self.df.columns:
            schema_info[col] = {
                'dtype': str(self.df[col].dtype),
                'detected_type': self.schema.get(col, 'object'),
                'null_count': int(self.df[col].isnull().sum()),
                'null_percentage': float(self.df[col].isnull().sum() / len(self.df) * 100),
                'unique_count': int(self.df[col].nunique()),
                'unique_percentage': float(self.df[col].nunique() / len(self.df) * 100),
                'sample_values': self.df[col].dropna().head(5).tolist() if len(self.df[col].dropna()) > 0 else []
            }
            
            # Add statistics for numeric columns
            if self.df[col].dtype in ['int64', 'float64']:
                schema_info[col].update({
                    'mean': float(self.df[col].mean()) if not self.df[col].isnull().all() else None,
                    'median': float(self.df[col].median()) if not self.df[col].isnull().all() else None,
                    'std': float(self.df[col].std()) if not self.df[col].isnull().all() else None,
                    'min': float(self.df[col].min()) if not self.df[col].isnull().all() else None,
                    'max': float(self.df[col].max()) if not self.df[col].isnull().all() else None
                })
            
            # Check for potential date columns
            if self.df[col].dtype == 'object' and len(self.df[col].dropna()) > 0:
                try:
                    pd.to_datetime(self.df[col].dropna().head(100))
                    schema_info[col]['potential_date'] = True
                except:
                    schema_info[col]['potential_date'] = False
        
        return schema_info
    
    def update_schema(self, column_updates: Dict[str, Dict[str, str]]) -> Tuple[bool, str]:
        """Update schema based on user edits"""
        try:
            for col, updates in column_updates.items():
                if col not in self.df.columns:
                    continue
                
                # Update data type
                if 'dtype' in updates:
                    new_dtype = updates['dtype']
                    self.df[col] = safe_convert_dtype(self.df[col], new_dtype)
                    self.schema[col] = new_dtype
                
                # Handle date format conversion
                if 'date_format' in updates and updates.get('dtype') == 'datetime64':
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], format=updates['date_format'])
                    except:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
            return True, "Schema updated successfully"
            
        except Exception as e:
            logging.error(f"Error updating schema: {e}")
            return False, str(e)
    
    def generate_quality_report(self, use_profiling: bool = False) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = {
            'basic_metrics': calculate_data_quality_metrics(self.df),
            'missing_patterns': detect_missing_patterns(self.df),
            'quality_issues': []
        }
        
        # Identify quality issues
        for col in self.df.columns:
            issues = []
            
            # Check missing values
            missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            if missing_pct > QUALITY_THRESHOLDS['missing_threshold'] * 100:
                issues.append(f"High missing values: {missing_pct:.2f}%")
            
            # Check cardinality for categorical
            if self.df[col].dtype == 'object':
                cardinality = self.df[col].nunique() / len(self.df)
                if cardinality > QUALITY_THRESHOLDS['cardinality_threshold']:
                    issues.append(f"High cardinality: {cardinality:.2f}")
            
            # Check for outliers in numeric columns
            if self.df[col].dtype in ['int64', 'float64']:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_count = (z_scores > QUALITY_THRESHOLDS['outlier_zscore']).sum()
                if outlier_count > 0:
                    issues.append(f"Outliers detected: {outlier_count} values")
            
            if issues:
                report['quality_issues'].append({
                    'column': col,
                    'issues': issues
                })
        
        # Check for duplicate rows
        duplicate_pct = (len(self.df[self.df.duplicated()]) / len(self.df)) * 100
        if duplicate_pct > QUALITY_THRESHOLDS['duplicate_threshold'] * 100:
            report['quality_issues'].append({
                'column': 'FULL_ROW',
                'issues': [f"Duplicate rows: {duplicate_pct:.2f}%"]
            })
        
        # Use advanced profiling if available and requested
        if use_profiling:
            if YDATA_AVAILABLE:
                try:
                    profile = ProfileReport(self.df, minimal=True)
                    report['ydata_profile'] = "Profile generated (view separately)"
                except Exception as e:
                    logging.warning(f"YData profiling failed: {e}")
            
            if SWEETVIZ_AVAILABLE:
                try:
                    sweet_report = sv.analyze(self.df)
                    report['sweetviz_report'] = "Report generated (view separately)"
                except Exception as e:
                    logging.warning(f"Sweetviz profiling failed: {e}")
        
        return report
    
    def execute_code(self, code: str, description: str = "") -> Tuple[bool, str, Any]:
        """Execute cleaning code safely"""
        try:
            # Create a copy to work with
            df = self.df.copy()
            
            # Prepare execution environment
            local_vars = {'df': df, 'pd': pd, 'np': np}
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            # Execute code
            exec(code, {'__builtins__': __builtins__, 'pd': pd, 'np': np}, local_vars)
            
            # Get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Update dataframe if successful
            if 'df' in local_vars:
                self.df = local_vars['df']
                
            # Store in history
            self.code_history.append(code)
            self.execution_logs.append({
                'code': code,
                'description': description,
                'success': True,
                'output': output
            })
            
            # Get any additional results
            results = {}
            for key in local_vars:
                if key not in ['df', 'pd', 'np', '__builtins__']:
                    results[key] = local_vars[key]
            
            return True, output, results
            
        except Exception as e:
            sys.stdout = old_stdout
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            
            self.execution_logs.append({
                'code': code,
                'description': description,
                'success': False,
                'error': error_msg
            })
            
            return False, error_msg, None
    
    def apply_missing_value_treatment(self, strategy: Dict[str, str]) -> Tuple[bool, str]:
        """Apply missing value treatment strategies"""
        try:
            for col, method in strategy.items():
                if col not in self.df.columns:
                    continue
                
                if method == 'drop':
                    self.df = self.df.dropna(subset=[col])
                elif method == 'mean':
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif method == 'median':
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                elif method == 'mode':
                    mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else None
                    if mode_val is not None:
                        self.df[col].fillna(mode_val, inplace=True)
                elif method == 'forward_fill':
                    self.df[col].fillna(method='ffill', inplace=True)
                elif method == 'backward_fill':
                    self.df[col].fillna(method='bfill', inplace=True)
                elif method == 'interpolate':
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].interpolate(inplace=True)
                else:
                    # Custom value
                    self.df[col].fillna(method, inplace=True)
            
            return True, "Missing values treated successfully"
            
        except Exception as e:
            logging.error(f"Error treating missing values: {e}")
            return False, str(e)
    
    def remove_duplicates(self, subset: List[str] = None, keep: str = 'first') -> Tuple[bool, str]:
        """Remove duplicate rows"""
        try:
            original_len = len(self.df)
            self.df = self.df.drop_duplicates(subset=subset, keep=keep)
            removed = original_len - len(self.df)
            
            return True, f"Removed {removed} duplicate rows"
            
        except Exception as e:
            logging.error(f"Error removing duplicates: {e}")
            return False, str(e)
    
    def handle_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> Tuple[bool, str]:
        """Handle outliers in numeric columns"""
        try:
            if column not in self.df.columns:
                return False, f"Column {column} not found"
            
            if self.df[column].dtype not in ['int64', 'float64']:
                return False, f"Column {column} is not numeric"
            
            if method == 'iqr':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers
                self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
                self.df = self.df[z_scores <= threshold]
            
            return True, f"Outliers handled for column {column}"
            
        except Exception as e:
            logging.error(f"Error handling outliers: {e}")
            return False, str(e)
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """Compare current dataframe with original"""
        if self.original_df is None or self.df is None:
            return {}
        
        return {
            'original_shape': self.original_df.shape,
            'current_shape': self.df.shape,
            'rows_removed': len(self.original_df) - len(self.df),
            'columns_added': list(set(self.df.columns) - set(self.original_df.columns)),
            'columns_removed': list(set(self.original_df.columns) - set(self.df.columns)),
            'memory_usage_original': self.original_df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'memory_usage_current': self.df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
    
    def export_data(self, file_path: str, file_format: str = 'csv') -> Tuple[bool, str]:
        """Export cleaned data"""
        try:
            if file_format == 'csv':
                self.df.to_csv(file_path, index=False)
            elif file_format == 'xlsx':
                self.df.to_excel(file_path, index=False)
            elif file_format == 'json':
                self.df.to_json(file_path, orient='records', indent=2)
            else:
                return False, f"Unsupported format: {file_format}"
            
            return True, f"Data exported to {file_path}"
            
        except Exception as e:
            logging.error(f"Error exporting data: {e}")
            return False, str(e)