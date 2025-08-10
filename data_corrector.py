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
import sys
from io import StringIO

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
            
            # Ensure the code has proper imports at the beginning
            if not correction_code.startswith('import'):
                correction_code = self._add_standard_imports() + "\n\n" + correction_code
            
            # Validate the generated code
            validation_result = self._validate_generated_code(correction_code)
            
            if not validation_result['is_safe']:
                # Try to fix common issues
                correction_code = self._fix_common_code_issues(correction_code)
                validation_result = self._validate_generated_code(correction_code)
                
                if not validation_result['is_safe']:
                    raise ValueError(f"Generated code failed safety validation: {validation_result['issues']}")
            
            # Generate strategy summary
            strategy_summary = self._generate_strategy_summary(correction_code, context)
            
            return correction_code, strategy_summary
            
        except Exception as e:
            # Fallback to a basic correction code
            fallback_code = self._generate_fallback_correction_code(data, quality_report)
            fallback_summary = "Basic data cleaning: handling missing values, removing duplicates, and standardizing formats."
            return fallback_code, fallback_summary
    
    def _add_standard_imports(self) -> str:
        """Add standard imports to the code"""
        return """import pandas as pd
import numpy as np
import datetime
import re
import warnings
warnings.filterwarnings('ignore')"""
    
    def _fix_common_code_issues(self, code: str) -> str:
        """Fix common issues in generated code"""
        # Ensure imports are at the top
        imports = []
        other_lines = []
        
        for line in code.split('\n'):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                imports.append(line)
            else:
                other_lines.append(line)
        
        # Add missing standard imports
        standard_imports = self._add_standard_imports().split('\n')
        for imp in standard_imports:
            if imp and not any(imp in existing for existing in imports):
                imports.append(imp)
        
        # Reconstruct the code
        fixed_code = '\n'.join(imports) + '\n\n' + '\n'.join(other_lines)
        
        # Ensure the clean_data function exists
        if 'def clean_data(' not in fixed_code:
            fixed_code = self._wrap_code_in_function(fixed_code)
        
        return fixed_code
    
    def _prepare_correction_context(self, data: pd.DataFrame, schema: Dict[str, Any],
                                  quality_report: Dict[str, Any], user_instructions: str) -> Dict[str, Any]:
        """Prepare context for code generation"""
        context = {
            'data_shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},
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
        
        IMPORTANT: Include all necessary imports at the beginning of the code.
        Start with:
        import pandas as pd
        import numpy as np
        import datetime
        import re
        import warnings
        
        The function should be production-ready and well-documented.
        """
        
        code = self.llm_client.generate_code(prompt)
        
        # Clean up the code
        code = code.strip()
        
        # Remove markdown code blocks if present
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        return code
    
    def _wrap_code_in_function(self, code: str) -> str:
        """Wrap loose code in a clean_data function"""
        # Check if there are already imports in the code
        has_imports = any(line.strip().startswith('import') for line in code.split('\n'))
        
        if not has_imports:
            imports = self._add_standard_imports()
        else:
            imports = ""
        
        wrapped_code = f"""{imports}

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
        indented_lines = []
        
        for line in lines:
            # Don't indent imports or function definitions at root level
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                continue
            elif line.strip():
                indented_lines.append(indent + line)
            else:
                indented_lines.append(line)
        
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
                        base_module = alias.name.split('.')[0]
                        if base_module not in self.allowed_imports and base_module not in ['warnings']:
                            validation_result['warnings'].append(f"Unusual import: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base_module = node.module.split('.')[0]
                        if base_module not in self.allowed_imports and base_module not in ['warnings']:
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
            
            start_time = time.time()
            
            # Capture print statements
            captured_output = StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured_output
            
            # Create execution environment with all necessary modules pre-imported
            exec_globals = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
                'pd': pd,
                'pandas': pd,
                'np': np,
                'numpy': np,
                'datetime': __import__('datetime'),
                're': re,
                'warnings': __import__('warnings'),
                'print': print,  # Ensure print is available
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'callable': callable,
                'type': type,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'AttributeError': AttributeError,
                'IndexError': IndexError
            }
            
            # Add common pandas/numpy functions
            exec_globals.update({
                'isna': pd.isna,
                'isnull': pd.isnull,
                'notna': pd.notna,
                'notnull': pd.notnull,
                'to_datetime': pd.to_datetime,
                'to_numeric': pd.to_numeric,
                'concat': pd.concat,
                'merge': pd.merge,
                'DataFrame': pd.DataFrame,
                'Series': pd.Series
            })
            
            # Execute the code
            exec(code, exec_globals)
            
            # Get the clean_data function
            if 'clean_data' not in exec_globals:
                # Try to find any function that looks like a cleaning function
                for name, obj in exec_globals.items():
                    if callable(obj) and 'clean' in name.lower():
                        clean_data_func = obj
                        break
                else:
                    raise ValueError("No 'clean_data' function found in the generated code")
            else:
                clean_data_func = exec_globals['clean_data']
            
            # Execute the cleaning function
            cleaned_data = clean_data_func(data.copy())
            
            # Restore stdout
            sys.stdout = old_stdout
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
            sys.stdout = old_stdout if 'old_stdout' in locals() else sys.stdout
            
            execution_log['errors'].append({
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Return original data if cleaning fails
            return data, execution_log
    
    def _generate_fallback_correction_code(self, data: pd.DataFrame, quality_report: Dict[str, Any]) -> str:
        """Generate a basic fallback correction code when LLM fails"""
        missing_percentage = quality_report.get('missing_values', {}).get('missing_percentage', 0)
        has_duplicates = quality_report.get('duplicates', {}).get('total_duplicates', 0) > 0
        
        code = """import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    \"\"\"Basic data cleaning function\"\"\"
    try:
        print(f"Starting data cleaning. Original shape: {df.shape}")
        
        # Make a copy
        cleaned_df = df.copy()
        
        # Remove duplicates
        if cleaned_df.duplicated().sum() > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            print(f"Removed {df.shape[0] - cleaned_df.shape[0]} duplicate rows")
        
        # Handle missing values
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().sum() > 0:
                if cleaned_df[column].dtype in ['float64', 'int64']:
                    # Fill numeric columns with median
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                else:
                    # Fill categorical columns with mode or 'Unknown'
                    if not cleaned_df[column].mode().empty:
                        cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                    else:
                        cleaned_df[column].fillna('Unknown', inplace=True)
        
        # Remove columns with too many missing values (>90%)
        threshold = 0.9
        for column in cleaned_df.columns:
            if (df[column].isnull().sum() / len(df)) > threshold:
                cleaned_df = cleaned_df.drop(columns=[column])
                print(f"Dropped column '{column}' due to excessive missing values")
        
        print(f"Data cleaning completed. Final shape: {cleaned_df.shape}")
        print(f"Rows removed: {df.shape[0] - cleaned_df.shape[0]}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return df
"""
        
        return code