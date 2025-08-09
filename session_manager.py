"""
Schema manager module for detecting and managing data schemas
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, Any, List, Union
import json
from config import CATEGORICAL_THRESHOLD, DATE_FORMATS, SCHEMA_DETECTION_PROMPT

class SchemaManager:
    """Manages schema detection and editing operations"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.categorical_threshold = CATEGORICAL_THRESHOLD
        self.date_formats = DATE_FORMATS
    
    def detect_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect schema for the provided DataFrame
        
        Args:
            data: pandas DataFrame
            
        Returns:
            Dictionary with column schema information
        """
        schema = {}
        
        for column in data.columns:
            col_schema = self._analyze_column(data[column])
            schema[column] = col_schema
        
        return schema
    
    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze individual column to determine schema"""
        col_info = {
            'dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100
        }
        
        # Determine if categorical
        if col_info['unique_percentage'] < self.categorical_threshold * 100:
            col_info['is_categorical'] = True
            col_info['categories'] = list(series.dropna().unique())
        else:
            col_info['is_categorical'] = False
        
        # Check for date/time patterns
        if series.dtype == 'object':
            date_format = self._detect_date_format(series)
            if date_format:
                col_info['is_datetime'] = True
                col_info['date_format'] = date_format
                col_info['suggested_dtype'] = 'datetime64[ns]'
            else:
                col_info['is_datetime'] = False
        
        # Suggest appropriate data type
        col_info['suggested_dtype'] = self._suggest_dtype(series, col_info)
        
        # Additional analysis for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            col_info.update(self._analyze_numeric_column(series))
        
        return col_info
    
    def _detect_date_format(self, series: pd.Series) -> Union[str, None]:
        """Detect date format in string column"""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return None
        
        # Sample some values for testing
        sample_size = min(100, len(non_null_series))
        sample = non_null_series.sample(sample_size) if len(non_null_series) > sample_size else non_null_series
        
        for date_format in self.date_formats:
            try:
                # Try to parse with this format
                parsed_count = 0
                for value in sample:
                    try:
                        datetime.strptime(str(value), date_format)
                        parsed_count += 1
                    except (ValueError, TypeError):
                        continue
                
                # If more than 80% can be parsed, consider this format valid
                if parsed_count / len(sample) > 0.8:
                    return date_format
            except:
                continue
        
        return None
    
    def _suggest_dtype(self, series: pd.Series, col_info: Dict[str, Any]) -> str:
        """Suggest appropriate data type for column"""
        if col_info.get('is_datetime', False):
            return 'datetime64[ns]'
        
        if col_info.get('is_categorical', False):
            return 'category'
        
        # Try to convert to numeric
        if series.dtype == 'object':
            numeric_series = pd.to_numeric(series, errors='coerce')
            non_null_count = numeric_series.notna().sum()
            
            if non_null_count / len(series) > 0.8:  # 80% can be converted to numeric
                if numeric_series.dropna().apply(lambda x: x.is_integer()).all():
                    return 'int64'
                else:
                    return 'float64'
        
        return str(series.dtype)
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Additional analysis for numeric columns"""
        numeric_info = {}
        
        try:
            numeric_info.update({
                'min_value': series.min(),
                'max_value': series.max(),
                'mean_value': series.mean(),
                'median_value': series.median(),
                'std_value': series.std(),
                'has_negative': (series < 0).any(),
                'has_zero': (series == 0).any(),
                'is_integer_like': series.dropna().apply(lambda x: x.is_integer()).all() if series.dtype == 'float64' else None
            })
        except Exception:
            pass
        
        return numeric_info
    
    def process_nl_schema_change(self, current_schema: Dict[str, Any], 
                                nl_instruction: str) -> Dict[str, Any]:
        """Process natural language instruction to modify schema"""
        try:
            # Check if LLM client is available
            if not self.llm_client or not self.llm_client.client:
                raise ValueError("LLM client not available. Please check your OpenAI API key.")
            
            # Prepare prompt for LLM
            prompt = f"""
            Current schema:
            {json.dumps(current_schema, indent=2, default=str)}
            
            User instruction: {nl_instruction}
            
            {SCHEMA_DETECTION_PROMPT}
            
            Update the schema based on the user instruction. Return the complete updated schema as JSON.
            """
            
            response = self.llm_client.get_completion(prompt)
            
            # Parse response as JSON
            try:
                updated_schema = json.loads(response)
                return updated_schema
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    updated_schema = json.loads(json_match.group())
                    return updated_schema
                else:
                    raise ValueError("Could not parse LLM response as JSON")
                    
        except Exception as e:
            raise ValueError(f"Error processing natural language instruction: {str(e)}")
    
    def apply_schema_changes(self, data: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Apply schema changes to DataFrame"""
        modified_data = data.copy()
        
        for column, col_schema in schema.items():
            if column not in modified_data.columns:
                continue
            
            try:
                suggested_dtype = col_schema.get('suggested_dtype', col_schema.get('dtype'))
                
                if suggested_dtype == 'datetime64[ns]':
                    # Convert to datetime
                    date_format = col_schema.get('date_format')
                    if date_format:
                        modified_data[column] = pd.to_datetime(
                            modified_data[column], 
                            format=date_format, 
                            errors='coerce'
                        )
                    else:
                        modified_data[column] = pd.to_datetime(
                            modified_data[column], 
                            errors='coerce'
                        )
                
                elif suggested_dtype == 'category':
                    # Convert to categorical
                    modified_data[column] = modified_data[column].astype('category')
                
                elif suggested_dtype in ['int64', 'Int64']:
                    # Convert to integer (nullable if Int64)
                    numeric_col = pd.to_numeric(modified_data[column], errors='coerce')
                    if suggested_dtype == 'Int64':
                        modified_data[column] = numeric_col.astype('Int64')
                    else:
                        modified_data[column] = numeric_col.astype('int64')
                
                elif suggested_dtype in ['float64', 'Float64']:
                    # Convert to float
                    modified_data[column] = pd.to_numeric(modified_data[column], errors='coerce')
                
                elif suggested_dtype == 'bool':
                    # Convert to boolean
                    modified_data[column] = modified_data[column].astype('bool')
                
                elif suggested_dtype == 'object':
                    # Ensure string type
                    modified_data[column] = modified_data[column].astype('object')
                
            except Exception as e:
                print(f"Warning: Could not convert column '{column}' to {suggested_dtype}: {str(e)}")
                continue
        
        return modified_data
    
    def validate_schema_changes(self, data: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate proposed schema changes"""
        validation_results = {}
        
        for column, col_schema in schema.items():
            if column not in data.columns:
                validation_results[column] = {
                    'valid': False,
                    'error': 'Column not found in data'
                }
                continue
            
            suggested_dtype = col_schema.get('suggested_dtype', col_schema.get('dtype'))
            
            try:
                # Test conversion
                test_series = data[column].copy()
                
                if suggested_dtype == 'datetime64[ns]':
                    pd.to_datetime(test_series, errors='coerce')
                elif suggested_dtype in ['int64', 'Int64']:
                    pd.to_numeric(test_series, errors='coerce')
                elif suggested_dtype in ['float64', 'Float64']:
                    pd.to_numeric(test_series, errors='coerce')
                
                validation_results[column] = {'valid': True}
                
            except Exception as e:
                validation_results[column] = {
                    'valid': False,
                    'error': str(e)
                }
        
        return validation_results
    
    def get_schema_summary(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics of the schema"""
        summary = {
            'total_columns': len(schema),
            'data_types': {},
            'categorical_columns': 0,
            'datetime_columns': 0,
            'numeric_columns': 0,
            'text_columns': 0,
            'columns_with_nulls': 0,
            'high_null_columns': []  # > 50% nulls
        }
        
        for column, col_info in schema.items():
            dtype = col_info.get('suggested_dtype', col_info.get('dtype', 'unknown'))
            
            # Count data types
            summary['data_types'][dtype] = summary['data_types'].get(dtype, 0) + 1
            
            # Count special types
            if col_info.get('is_categorical', False):
                summary['categorical_columns'] += 1
            
            if col_info.get('is_datetime', False):
                summary['datetime_columns'] += 1
            
            if dtype in ['int64', 'float64', 'Int64', 'Float64']:
                summary['numeric_columns'] += 1
            
            if dtype == 'object':
                summary['text_columns'] += 1
            
            # Count null issues
            if col_info.get('null_count', 0) > 0:
                summary['columns_with_nulls'] += 1
            
            if col_info.get('null_percentage', 0) > 50:
                summary['high_null_columns'].append(column)
        
        return summary