"""
Data corrector module for generating and executing data cleaning code
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, Any, Tuple, List
from config import CORRECTION_CODE_PROMPT, STRATEGY_SUMMARY_PROMPT, ALLOWED_IMPORTS, BLOCKED_FUNCTIONS
import traceback
import ast

class DataCorrector:
    """Handles data correction code generation and execution"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.allowed_imports = ALLOWED_IMPORTS
        self.blocked_functions = BLOCKED_FUNCTIONS
    
    def generate_correction_code(self, data: pd.DataFrame, schema: Dict[str, Any],
                               quality_report: Dict[str, Any], user_instructions: str = "") -> Tuple[str, str]:
        """
        Generate data correction code using LLM
        
        Args:
            data: DataFrame to be cleaned
            schema: Data schema information
            quality_report: Quality analysis report
            user_instructions: Additional user instructions
            
        Returns:
            Tuple of (correction_code, strategy_summary)
        """
        try:
            # Prepare context for code generation
            context = self._prepare_correction_context(data, schema, quality_report, user_instructions)
            
            # Generate correction code
            correction_code = self._generate_code_with_llm(context)
            
            # Validate the generated code
            validation_result = self._validate_generated_code(correction_code)
            
            if not validation_result['is_safe']:
                raise ValueError(f"Generated code failed safety validation: {validation_result['issues']}")
            
            # Generate strategy summary
            strategy_summary = self._generate_strategy_summary(correction_code, context)
            
            return correction_code, strategy_summary
            
        except Exception as e:
            raise ValueError(f"Failed to generate correction code: {str(e)}")
    
    def _prepare_correction_context(self, data: pd.DataFrame, schema: Dict[str, Any],
                                  quality_report: Dict[str, Any], user_instructions: str) -> Dict[str, Any]:
        """Prepare context for code generation"""
        context = {
            'data_shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'sample_data': data.head(3).to_dict(),
            'schema': schema,
            'quality_issues': {
                'missing_values': quality_report['missing_values'],
                'duplicates': quality_report['duplicates'],
                'outliers': quality_report['outliers'],
                'inconsistencies': quality_report.get('categorical_analysis', {}).get('inconsistencies', {}),
                'data_consistency': quality_report.get('data_consistency', {})
            },
            'user_instructions': user_instructions,
            'quality_score': quality_report.get('quality_score', 0)
        }
        
        return context
    
    def _generate_code_with_llm(self, context: Dict[str, Any]) -> str:
        """Generate correction code using LLM"""
        prompt = f"""
        {CORRECTION_CODE_PROMPT}
        
        Data Context:
        - Shape: {context['data_shape']}
        - Columns: {context['columns']}
        - Data types: {context['dtypes']}
        
        Sample Data:
        {json.dumps(context['sample_data'], indent=2, default=str)}
        
        Quality Issues:
        {json.dumps(context['quality_issues'], indent=2, default=str)}
        
        User Instructions: {context['user_instructions']}
        
        Generate a complete Python function called 'clean_data' that:
        1. Takes a DataFrame as input parameter 'df'
        2. Returns the cleaned DataFrame
        3. Handles all identified issues systematically
        4. Includes proper error handling and logging
        5. Uses only pandas, numpy, datetime, and re libraries
        
        The function should be production-ready and well-documented.
        """
        
        code = self.llm_client.generate_code(prompt)
        
        # Ensure the code defines a clean_data function
        if 'def clean_data(' not in code:
            code = self._wrap_code_in_function(code)
        
        return code
    
    def _wrap_code_in_function(self, code: str) -> str:
        """Wrap loose code in a clean_data function"""
        wrapped_code = f"""
import pandas as pd
import numpy as np
import datetime
import re
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    \"\"\"
    Clean the input DataFrame
    
    Args:
        df: pandas DataFrame to clean
        
    Returns:
        pandas DataFrame: Cleaned data
    \"\"\"
    try:
        # Store original shape for logging
        original_shape = df.shape
        print(f"Starting data cleaning. Original shape: {{original_shape}}")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Generated cleaning code
{self._indent_code(code, 8)}
        
        # Log final results
        final_shape = cleaned_df.shape
        print(f"Data cleaning completed. Final shape: {{final_shape}}")
        print(f"Rows removed: {{original_shape[0] - final_shape[0]}}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error during data cleaning: {{str(e)}}")
        print("Returning original data")
        return df
"""
        return wrapped_code
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Add indentation to code lines"""
        indent = " " * spaces
        lines = code.split('\n')
        indented_lines = [indent + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)
    
    def _validate_generated_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code for safety"""
        validation_result = {
            'is_safe': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Parse the code to check for dangerous constructs
            tree = ast.parse(code)
            
            # Check for blocked functions
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.blocked_functions:
                            validation_result['is_safe'] = False
                            validation_result['issues'].append(f"Blocked function used: {node.func.id}")
                
                # Check imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] not in self.allowed_imports:
                            validation_result['warnings'].append(f"Unusual import: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] not in self.allowed_imports:
                        validation_result['warnings'].append(f"Unusual import from: {node.module}")
            
            # Check for basic syntax correctness
            compile(code, '<string>', 'exec')
            
        except SyntaxError as e:
            validation_result['is_safe'] = False
            validation_result['issues'].append(f"Syntax error: {str(e)}")
        except Exception as e:
            validation_result['warnings'].append(f"Validation warning: {str(e)}")
        
        return validation_result
    
    def _generate_strategy_summary(self, code: str, context: Dict[str, Any]) -> str:
        """Generate human-readable strategy summary"""
        try:
            context_str = f"Data has {context['data_shape'][0]} rows and {context['data_shape'][1]} columns with quality score {context['quality_score']}"
            summary = self.llm_client.summarize_strategy(code, context_str)
            return summary
        except Exception as e:
            # Fallback summary
            return f"Data cleaning strategy addresses missing values, duplicates, and data quality issues. Quality score: {context['quality_score']}/100"
    
    def execute_correction_code(self, data: pd.DataFrame, code: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute the correction code safely
        
        Args:
            data: DataFrame to clean
            code: Python code to execute
            
        Returns:
            Tuple of (cleaned_data, execution_log)
        """
        execution_log = {
            'success': False,
            'original_shape': data.shape,
            'final_shape': None,
            'execution_time': None,
            'errors': [],
            'warnings': [],
            'output': []
        }
        
        try:
            import time
            from io import StringIO
            import sys
            
            start_time = time.time()
            
            # Capture print statements
            captured_output = StringIO()
            sys.stdout = captured_output
            
            # Create execution environment
            exec_globals = {
                'pd': pd,
                'pandas': pd,
                'np': np,
                'numpy': np,
                'datetime': __import__('datetime'),
                're': re,
                'warnings': __import__('warnings'),
                '__builtins__': {}  # Restricted builtins for safety
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Get the clean_data function
            if 'clean_data' not in exec_globals:
                raise ValueError("Generated code must define a 'clean_data' function")
            
            clean_data_func = exec_globals['clean_data']
            
            # Execute the cleaning function
            cleaned_data = clean_data_func(data.copy())
            
            # Restore stdout
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            execution_log.update({
                'success': True,
                'final_shape': cleaned_data.shape,
                'execution_time': time.time() - start_time,
                'output': output.split('\n') if output else [],
                'rows_removed': data.shape[0] - cleaned_data.shape[0],
                'columns_removed': data.shape[1] - cleaned_data.shape[1]
            })
            
            return cleaned_data, execution_log
            
        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = sys.__stdout__
            
            execution_log['errors'].append({
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Return original data if cleaning fails
            return data, execution_log
    
    def modify_correction_code(self, original_code: str, modification_request: str) -> str:
        """
        Modify correction code based on user request
        
        Args:
            original_code: Original correction code
            modification_request: User's modification request
            
        Returns:
            Modified code
        """
        try:
            prompt = f"""
            Modify the following Python data cleaning code based on the user's request:
            
            Original Code:
            ```python
            {original_code}
            ```
            
            User's Modification Request: {modification_request}
            
            Return the complete modified code. Ensure the function signature remains 'clean_data(df)' 
            and follows the same structure and safety guidelines.
            """
            
            modified_code = self.llm_client.generate_code(prompt)
            
            # Validate the modified code
            validation_result = self._validate_generated_code(modified_code)
            
            if not validation_result['is_safe']:
                raise ValueError(f"Modified code failed safety validation: {validation_result['issues']}")
            
            return modified_code
            
        except Exception as e:
            raise ValueError(f"Failed to modify correction code: {str(e)}")
    
    def generate_correction_report(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame,
                                 execution_log: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive correction report"""
        report = {
            'summary': {
                'success': execution_log['success'],
                'original_shape': execution_log['original_shape'],
                'final_shape': execution_log['final_shape'],
                'execution_time': execution_log.get('execution_time', 0),
                'rows_removed': execution_log.get('rows_removed', 0),
                'columns_removed': execution_log.get('columns_removed', 0)
            },
            'data_changes': {},
            'quality_improvements': {},
            'issues_resolved': [],
            'execution_details': execution_log
        }
        
        if execution_log['success']:
            # Analyze data changes
            report['data_changes'] = self._analyze_data_changes(original_data, cleaned_data)
            
            # Calculate quality improvements
            report['quality_improvements'] = self._calculate_quality_improvements(original_data, cleaned_data)
            
            # Identify resolved issues
            report['issues_resolved'] = self._identify_resolved_issues(original_data, cleaned_data)
        
        return report
    
    def _analyze_data_changes(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict[str, Any]:
        """Analyze changes between original and cleaned data"""
        changes = {
            'shape_change': {
                'rows': cleaned.shape[0] - original.shape[0],
                'columns': cleaned.shape[1] - original.shape[1]
            },
            'column_changes': {},
            'data_type_changes': {},
            'missing_value_changes': {}
        }
        
        # Analyze column-level changes
        common_columns = set(original.columns) & set(cleaned.columns)
        
        for column in common_columns:
            original_col = original[column]
            cleaned_col = cleaned[column]
            
            # Data type changes
            if str(original_col.dtype) != str(cleaned_col.dtype):
                changes['data_type_changes'][column] = {
                    'from': str(original_col.dtype),
                    'to': str(cleaned_col.dtype)
                }
            
            # Missing value changes
            original_nulls = original_col.isnull().sum()
            cleaned_nulls = cleaned_col.isnull().sum()
            
            if original_nulls != cleaned_nulls:
                changes['missing_value_changes'][column] = {
                    'original_nulls': original_nulls,
                    'cleaned_nulls': cleaned_nulls,
                    'difference': cleaned_nulls - original_nulls
                }
        
        # Identify dropped columns
        dropped_columns = set(original.columns) - set(cleaned.columns)
        if dropped_columns:
            changes['dropped_columns'] = list(dropped_columns)
        
        # Identify new columns
        new_columns = set(cleaned.columns) - set(original.columns)
        if new_columns:
            changes['new_columns'] = list(new_columns)
        
        return changes
    
    def _calculate_quality_improvements(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality score improvements"""
        improvements = {
            'missing_values': {},
            'duplicates': {},
            'data_consistency': {}
        }
        
        # Missing values improvement
        original_missing = (original.isnull().sum().sum() / original.size) * 100
        cleaned_missing = (cleaned.isnull().sum().sum() / cleaned.size) * 100
        
        improvements['missing_values'] = {
            'original_percentage': original_missing,
            'cleaned_percentage': cleaned_missing,
            'improvement': original_missing - cleaned_missing
        }
        
        # Duplicates improvement
        original_duplicates = (original.duplicated().sum() / len(original)) * 100
        cleaned_duplicates = (cleaned.duplicated().sum() / len(cleaned)) * 100
        
        improvements['duplicates'] = {
            'original_percentage': original_duplicates,
            'cleaned_percentage': cleaned_duplicates,
            'improvement': original_duplicates - cleaned_duplicates
        }
        
        # Overall quality score improvement (simplified calculation)
        original_quality = 100 - (original_missing * 0.5) - (original_duplicates * 0.3)
        cleaned_quality = 100 - (cleaned_missing * 0.5) - (cleaned_duplicates * 0.3)
        
        improvements['overall_quality'] = {
            'original_score': max(0, original_quality),
            'cleaned_score': max(0, cleaned_quality),
            'improvement': cleaned_quality - original_quality
        }
        
        return improvements
    
    def _identify_resolved_issues(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> List[str]:
        """Identify what issues were resolved"""
        resolved_issues = []
        
        # Check for missing value resolution
        original_missing = original.isnull().sum().sum()
        cleaned_missing = cleaned.isnull().sum().sum()
        
        if cleaned_missing < original_missing:
            resolved_issues.append(f"Reduced missing values by {original_missing - cleaned_missing} cells")
        
        # Check for duplicate resolution
        original_duplicates = original.duplicated().sum()
        cleaned_duplicates = cleaned.duplicated().sum()
        
        if cleaned_duplicates < original_duplicates:
            resolved_issues.append(f"Removed {original_duplicates - cleaned_duplicates} duplicate rows")
        
        # Check for data type improvements
        common_columns = set(original.columns) & set(cleaned.columns)
        type_improvements = 0
        
        for column in common_columns:
            original_type = str(original[column].dtype)
            cleaned_type = str(cleaned[column].dtype)
            
            # Consider certain type changes as improvements
            if (original_type == 'object' and cleaned_type in ['datetime64[ns]', 'category', 'int64', 'float64']):
                type_improvements += 1
        
        if type_improvements > 0:
            resolved_issues.append(f"Improved data types for {type_improvements} columns")
        
        # Check for outlier handling (simplified)
        numeric_columns = original.select_dtypes(include=[np.number]).columns
        outliers_handled = 0
        
        for column in numeric_columns:
            if column in cleaned.columns:
                original_std = original[column].std()
                cleaned_std = cleaned[column].std()
                
                # If standard deviation decreased significantly, outliers might have been handled
                if cleaned_std < original_std * 0.8:
                    outliers_handled += 1
        
        if outliers_handled > 0:
            resolved_issues.append(f"Handled outliers in {outliers_handled} numeric columns")
        
        return resolved_issues
    
    def generate_correction_templates(self) -> Dict[str, str]:
        """Generate common correction code templates"""
        templates = {
            'handle_missing_values': '''
def handle_missing_values(df):
    """Handle missing values in the dataset"""
    cleaned_df = df.copy()
    
    # For numeric columns, fill with median
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if cleaned_df[col].isnull().sum() > 0:
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_columns = cleaned_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if cleaned_df[col].isnull().sum() > 0:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col].fillna(mode_value[0], inplace=True)
    
    return cleaned_df
            ''',
            
            'remove_duplicates': '''
def remove_duplicates(df):
    """Remove duplicate records"""
    cleaned_df = df.copy()
    
    # Remove exact duplicates
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    removed_count = initial_count - len(cleaned_df)
    
    print(f"Removed {removed_count} duplicate rows")
    return cleaned_df
            ''',
            
            'handle_outliers': '''
def handle_outliers(df, method='iqr'):
    """Handle outliers using IQR method"""
    cleaned_df = df.copy()
    
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        if method == 'iqr':
            Q1 = cleaned_df[column].quantile(0.25)
            Q3 = cleaned_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return cleaned_df
            ''',
            
            'standardize_formats': '''
def standardize_formats(df):
    """Standardize data formats"""
    cleaned_df = df.copy()
    
    # Standardize text columns
    text_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in text_columns:
        # Remove extra whitespace and standardize case
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.title()
    
    return cleaned_df
            '''
        }
        
        return templates
    
    def apply_template_correction(self, data: pd.DataFrame, template_name: str) -> pd.DataFrame:
        """Apply a predefined correction template"""
        templates = self.generate_correction_templates()
        
        if template_name not in templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        try:
            # Execute the template
            exec_globals = {
                'pd': pd,
                'np': np,
                'datetime': __import__('datetime'),
                're': re
            }
            
            exec(templates[template_name], exec_globals)
            
            # Get the function (same name as template)
            func_name = template_name
            if func_name not in exec_globals:
                # Try to find the function in the executed code
                for name, obj in exec_globals.items():
                    if callable(obj) and name.startswith('handle_') or name.startswith('remove_') or name.startswith('standardize_'):
                        func_name = name
                        break
            
            correction_func = exec_globals[func_name]
            return correction_func(data.copy())
            
        except Exception as e:
            raise ValueError(f"Failed to apply template '{template_name}': {str(e)}")
    
    def get_correction_suggestions(self, quality_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get automated correction suggestions based on quality report"""
        suggestions = []
        
        # Missing values suggestions
        missing_percentage = quality_report['missing_values']['missing_percentage']
        if missing_percentage > 5:
            suggestions.append({
                'type': 'missing_values',
                'priority': 'high' if missing_percentage > 20 else 'medium',
                'description': f'Handle {missing_percentage:.1f}% missing values',
                'template': 'handle_missing_values',
                'impact': 'Improves data completeness and analysis accuracy'
            })
        
        # Duplicates suggestions
        duplicate_percentage = quality_report['duplicates']['duplicate_percentage']
        if duplicate_percentage > 1:
            suggestions.append({
                'type': 'duplicates',
                'priority': 'high' if duplicate_percentage > 10 else 'medium',
                'description': f'Remove {duplicate_percentage:.1f}% duplicate records',
                'template': 'remove_duplicates',
                'impact': 'Eliminates redundant data and improves analysis accuracy'
            })
        
        # Outliers suggestions
        outlier_percentage = quality_report['outliers'].get('outlier_percentage', 0)
        if outlier_percentage > 2:
            suggestions.append({
                'type': 'outliers',
                'priority': 'medium',
                'description': f'Handle {outlier_percentage:.1f}% outlier values',
                'template': 'handle_outliers',
                'impact': 'Reduces skew and improves statistical analysis'
            })
        
        # Data consistency suggestions
        inconsistencies = quality_report.get('categorical_analysis', {}).get('inconsistencies', {})
        if inconsistencies:
            suggestions.append({
                'type': 'formatting',
                'priority': 'low',
                'description': f'Standardize format for {len(inconsistencies)} columns',
                'template': 'standardize_formats',
                'impact': 'Improves data consistency and reduces processing errors'
            })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        suggestions.sort(key=lambda x: priority_order[x['priority']], reverse=True)
        
        return suggestions