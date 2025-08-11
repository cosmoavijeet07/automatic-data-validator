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
            # Prepare comprehensive context for code generation
            context = self._prepare_correction_context(data, schema, quality_report, user_instructions)
            
            # Generate correction code with full context
            correction_code = self._generate_code_with_llm(context, data)
            
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
        """Prepare comprehensive context for code generation"""
        
        # Get complete data statistics
        data_stats = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},
            'sample_data': data.head(5).to_dict('records'),
            'null_counts': data.isnull().sum().to_dict(),
            'unique_counts': {col: data[col].nunique() for col in data.columns},
            'memory_usage': data.memory_usage(deep=True).to_dict()
        }
        
        # Extract all quality issues
        quality_issues = {
            'missing_values': quality_report.get('missing_values', {}),
            'duplicates': quality_report.get('duplicates', {}),
            'outliers': quality_report.get('outliers', {}),
            'categorical_analysis': quality_report.get('categorical_analysis', {}),
            'numeric_analysis': quality_report.get('numeric_analysis', {}),
            'data_consistency': quality_report.get('data_consistency', {}),
            'data_types': quality_report.get('data_types', {}),
            'quality_score': quality_report.get('quality_score', 0)
        }
        
        context = {
            'data_stats': data_stats,
            'schema': schema,
            'quality_issues': quality_issues,
            'user_instructions': user_instructions,
            'enhanced_analysis': quality_report.get('enhanced_analysis', {}),
            'critical_issues': self._identify_critical_issues(quality_report)
        }
        
        return context
    
    def _identify_critical_issues(self, quality_report: Dict[str, Any]) -> List[str]:
        """Identify critical issues that must be addressed"""
        critical_issues = []
        
        # Check missing values
        missing_info = quality_report.get('missing_values', {})
        if missing_info.get('problematic_columns'):
            critical_issues.append(f"Columns with >50% missing: {', '.join(missing_info['problematic_columns'])}")
        
        # Check duplicates
        dup_info = quality_report.get('duplicates', {})
        if dup_info.get('is_problematic'):
            critical_issues.append(f"High duplicate rate: {dup_info['duplicate_percentage']:.1f}%")
        
        # Check outliers
        outlier_info = quality_report.get('outliers', {})
        if outlier_info.get('outlier_percentage', 0) > 10:
            critical_issues.append(f"High outlier rate: {outlier_info['outlier_percentage']:.1f}%")
        
        # Check data type issues
        type_info = quality_report.get('data_types', {})
        if type_info.get('type_suggestions'):
            critical_issues.append(f"Data type optimization needed for {len(type_info['type_suggestions'])} columns")
        
        return critical_issues
    
    def _generate_code_with_llm(self, context: Dict[str, Any], data: pd.DataFrame) -> str:
        """Generate correction code using LLM with comprehensive context"""
        
        # Create detailed prompt with all context
        prompt = f"""
        {CORRECTION_CODE_PROMPT}
        
        COMPLETE DATA CONTEXT:
        ===================
        
        Data Statistics:
        - Shape: {context['data_stats']['shape']}
        - Columns: {context['data_stats']['columns']}
        - Data types: {json.dumps(context['data_stats']['dtypes'], indent=2)}
        - Null counts per column: {json.dumps(context['data_stats']['null_counts'], indent=2)}
        - Unique values per column: {json.dumps(context['data_stats']['unique_counts'], indent=2)}
        
        Sample Data (first 5 rows):
        {json.dumps(context['data_stats']['sample_data'], indent=2, default=str)}
        
        COMPLETE SCHEMA INFORMATION:
        ===========================
        {json.dumps(context['schema'], indent=2, default=str)}
        
        COMPLETE QUALITY ANALYSIS:
        =========================
        Missing Values:
        {json.dumps(context['quality_issues']['missing_values'], indent=2, default=str)}
        
        Duplicates:
        {json.dumps(context['quality_issues']['duplicates'], indent=2, default=str)}
        
        Outliers:
        {json.dumps(context['quality_issues']['outliers'], indent=2, default=str)}
        
        Categorical Analysis:
        {json.dumps(context['quality_issues']['categorical_analysis'], indent=2, default=str)}
        
        Numeric Analysis:
        {json.dumps(context['quality_issues']['numeric_analysis'], indent=2, default=str)}
        
        Data Type Issues:
        {json.dumps(context['quality_issues']['data_types'], indent=2, default=str)}
        
        CRITICAL ISSUES TO ADDRESS:
        ==========================
        {chr(10).join(['- ' + issue for issue in context['critical_issues']])}
        
        Quality Score: {context['quality_issues']['quality_score']}/100
        
        User Instructions: {context['user_instructions'] if context['user_instructions'] else 'No specific instructions provided'}
        
        REQUIREMENTS:
        ============
        Generate a complete Python function called 'clean_data' that:
        1. Takes a DataFrame as input parameter 'df'
        2. Returns the cleaned DataFrame
        3. Addresses ALL identified quality issues systematically
        4. Includes detailed logging of operations performed
        5. Uses only pandas, numpy, datetime, and re libraries
        6. Handles edge cases and errors gracefully
        7. Preserves data integrity while cleaning
        
        The function should:
        - Handle missing values based on column type and distribution
        - Remove or handle duplicates appropriately
        - Fix data type issues
        - Handle outliers using appropriate methods
        - Standardize categorical values
        - Log each operation with before/after statistics
        
        IMPORTANT: 
        - Include all necessary imports at the beginning
        - Add detailed comments explaining each cleaning step
        - Print progress messages for each major operation
        - The function must be production-ready and well-documented
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
        print(f"Columns removed: {{original_shape[1] - final_shape[1]}}")
        
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
            # Create comprehensive context string
            context_str = f"""
            Data has {context['data_stats']['shape'][0]} rows and {context['data_stats']['shape'][1]} columns.
            Quality score: {context['quality_issues']['quality_score']}/100.
            Critical issues: {', '.join(context['critical_issues']) if context['critical_issues'] else 'None identified'}.
            """
            
            summary = self.llm_client.summarize_strategy(code, context_str)
            return summary
        except Exception as e:
            # Fallback summary
            return f"Data cleaning strategy addresses {len(context['critical_issues'])} critical issues. Quality score: {context['quality_issues']['quality_score']}/100"
    
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
                'print': print,
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
        """Generate a comprehensive fallback correction code when LLM fails"""
        
        missing_info = quality_report.get('missing_values', {})
        dup_info = quality_report.get('duplicates', {})
        outlier_info = quality_report.get('outliers', {})
        cat_info = quality_report.get('categorical_analysis', {})
        
        code = """import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    \"\"\"Comprehensive data cleaning function\"\"\"
    try:
        print(f"Starting data cleaning. Original shape: {df.shape}")
        print("="*50)
        
        # Make a copy
        cleaned_df = df.copy()
        
        # 1. Remove duplicates
        print("\\nStep 1: Removing duplicates...")
        duplicates_before = cleaned_df.duplicated().sum()
        if duplicates_before > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            print(f"  Removed {duplicates_before} duplicate rows")
        else:
            print("  No duplicates found")
        
        # 2. Handle missing values
        print("\\nStep 2: Handling missing values...")
        for column in cleaned_df.columns:
            null_count = cleaned_df[column].isnull().sum()
            if null_count > 0:
                null_percentage = (null_count / len(cleaned_df)) * 100
                
                # Drop column if >90% missing
                if null_percentage > 90:
                    cleaned_df = cleaned_df.drop(columns=[column])
                    print(f"  Dropped column '{column}' (>{null_percentage:.1f}% missing)")
                    continue
                
                # Handle based on data type
                if cleaned_df[column].dtype in ['float64', 'int64']:
                    # Numeric columns - use median for <30% missing, mean otherwise
                    if null_percentage < 30:
                        fill_value = cleaned_df[column].median()
                        cleaned_df[column].fillna(fill_value, inplace=True)
                        print(f"  Filled {null_count} missing values in '{column}' with median: {fill_value:.2f}")
                    else:
                        fill_value = cleaned_df[column].mean()
                        cleaned_df[column].fillna(fill_value, inplace=True)
                        print(f"  Filled {null_count} missing values in '{column}' with mean: {fill_value:.2f}")
                elif cleaned_df[column].dtype == 'object':
                    # Categorical columns - use mode or 'Unknown'
                    if not cleaned_df[column].mode().empty:
                        fill_value = cleaned_df[column].mode()[0]
                        cleaned_df[column].fillna(fill_value, inplace=True)
                        print(f"  Filled {null_count} missing values in '{column}' with mode: '{fill_value}'")
                    else:
                        cleaned_df[column].fillna('Unknown', inplace=True)
                        print(f"  Filled {null_count} missing values in '{column}' with 'Unknown'")
        
        # 3. Handle outliers in numeric columns
        print("\\nStep 3: Handling outliers...")
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            Q1 = cleaned_df[column].quantile(0.25)
            Q3 = cleaned_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = cleaned_df[(cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)]
            if len(outliers) > 0:
                # Cap outliers instead of removing
                cleaned_df[column] = cleaned_df[column].clip(lower_bound, upper_bound)
                print(f"  Capped {len(outliers)} outliers in '{column}'")
        
        # 4. Standardize categorical values
        print("\\nStep 4: Standardizing categorical values...")
        object_columns = cleaned_df.select_dtypes(include=['object']).columns
        for column in object_columns:
            # Strip whitespace and convert to title case
            cleaned_df[column] = cleaned_df[column].astype(str).str.strip().str.title()
            print(f"  Standardized values in '{column}'")
        
        # 5. Convert data types where appropriate
        print("\\nStep 5: Optimizing data types...")
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype == 'object':
                # Try to convert to datetime
                try:
                    cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
                    if cleaned_df[column].notna().sum() > 0:
                        print(f"  Converted '{column}' to datetime")
                    else:
                        cleaned_df[column] = cleaned_df[column].astype('object')
                except:
                    pass
                
                # Check if should be categorical
                if cleaned_df[column].dtype == 'object':
                    unique_ratio = cleaned_df[column].nunique() / len(cleaned_df)
                    if unique_ratio < 0.05:  # Less than 5% unique values
                        cleaned_df[column] = cleaned_df[column].astype('category')
                        print(f"  Converted '{column}' to categorical")
        
        print("\\n" + "="*50)
        print(f"Data cleaning completed. Final shape: {cleaned_df.shape}")
        print(f"Rows removed: {df.shape[0] - cleaned_df.shape[0]}")
        print(f"Columns removed: {df.shape[1] - cleaned_df.shape[1]}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return df
"""
        
        return code