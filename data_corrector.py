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
        return self.generate_correction_code_with_full_context(data, schema, quality_report, user_instructions)
    
    def generate_correction_code_with_full_context(self, data: pd.DataFrame, schema: Dict[str, Any],
                                                  quality_report: Dict[str, Any], user_instructions: str = "") -> Tuple[str, str]:
        """
        Generate data correction code with comprehensive context
        
        Args:
            data: DataFrame to be cleaned
            schema: Complete updated schema information
            quality_report: Complete quality analysis report
            user_instructions: Additional user instructions (may include error context)
            
        Returns:
            Tuple of (correction_code, strategy_summary)
        """
        try:
            # Prepare comprehensive context
            context = self._prepare_comprehensive_context(data, schema, quality_report, user_instructions)
            
            # Generate correction code with full context
            correction_code = self._generate_code_with_comprehensive_llm(context)
            
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
                    # Generate safer fallback code
                    correction_code = self._generate_safe_fallback_code(data, quality_report, context)
            
            # Generate strategy summary
            strategy_summary = self._generate_strategy_summary(correction_code, context)
            
            return correction_code, strategy_summary
            
        except Exception as e:
            # Fallback to a basic correction code
            fallback_code = self._generate_safe_fallback_code(data, quality_report, {})
            fallback_summary = "Basic data cleaning: handling missing values, removing duplicates, and standardizing formats."
            return fallback_code, fallback_summary
    
    def _prepare_comprehensive_context(self, data: pd.DataFrame, schema: Dict[str, Any],
                                      quality_report: Dict[str, Any], user_instructions: str) -> Dict[str, Any]:
        """Prepare comprehensive context for code generation"""
        
        # Extract detailed column information
        column_details = {}
        for col in data.columns:
            column_details[col] = {
                'dtype': str(data[col].dtype),
                'null_count': data[col].isnull().sum(),
                'null_percentage': (data[col].isnull().sum() / len(data)) * 100,
                'unique_count': data[col].nunique(),
                'sample_values': data[col].dropna().head(5).tolist() if not data[col].dropna().empty else [],
                'has_outliers': col in quality_report.get('outliers', {}).get('columns_with_outliers', {}),
                'is_categorical': col in quality_report.get('categorical_analysis', {}).get('categorical_columns', []),
                'schema_info': schema.get(col, {})
            }
        
        context = {
            'data_shape': data.shape,
            'columns': list(data.columns),
            'column_details': column_details,
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},
            'sample_data': data.head(5).to_dict(),
            'schema': schema,
            'quality_score': quality_report.get('quality_score', 0),
            'quality_issues': {
                'missing_values': quality_report['missing_values'],
                'duplicates': quality_report['duplicates'],
                'outliers': quality_report['outliers'],
                'inconsistencies': quality_report.get('categorical_analysis', {}).get('inconsistencies', {}),
                'data_consistency': quality_report.get('data_consistency', {}),
                'type_suggestions': quality_report.get('data_types', {}).get('type_suggestions', {}),
                'problematic_columns': quality_report['missing_values'].get('problematic_columns', [])
            },
            'numeric_analysis': quality_report.get('numeric_analysis', {}),
            'categorical_analysis': quality_report.get('categorical_analysis', {}),
            'user_instructions': user_instructions
        }
        
        return context
    
    def _generate_code_with_comprehensive_llm(self, context: Dict[str, Any]) -> str:
        """Generate correction code using LLM with comprehensive context"""
        
        # Create detailed prompt with all context
        prompt = f"""
        Generate comprehensive Python code to clean the dataset based on the complete analysis.
        
        DATASET INFORMATION:
        - Shape: {context['data_shape']}
        - Quality Score: {context['quality_score']}/100
        - Columns: {context['columns']}
        
        DETAILED SCHEMA (UPDATED):
        {json.dumps(context['schema'], indent=2, default=str)}
        
        COLUMN DETAILS:
        {json.dumps(context['column_details'], indent=2, default=str)}
        
        QUALITY ISSUES FOUND:
        1. Missing Values: {context['quality_issues']['missing_values']['missing_percentage']:.1f}%
           - Problematic columns: {context['quality_issues']['missing_values'].get('problematic_columns', [])}
           - Columns with missing: {list(context['quality_issues']['missing_values'].get('columns_with_missing', {}).keys())}
        
        2. Duplicates: {context['quality_issues']['duplicates']['duplicate_percentage']:.1f}%
           - Total duplicate rows: {context['quality_issues']['duplicates']['total_duplicates']}
        
        3. Outliers: {context['quality_issues']['outliers'].get('outlier_percentage', 0):.1f}%
           - Columns with outliers: {list(context['quality_issues']['outliers'].get('columns_with_outliers', {}).keys())}
        
        4. Data Type Issues:
           - Suggested type changes: {json.dumps(context['quality_issues'].get('type_suggestions', {}), indent=2)}
        
        5. Categorical Inconsistencies:
           {json.dumps(context['quality_issues'].get('inconsistencies', {}), indent=2, default=str)}
        
        NUMERIC COLUMNS ANALYSIS:
        {json.dumps(context.get('numeric_analysis', {}), indent=2, default=str)}
        
        CATEGORICAL COLUMNS ANALYSIS:
        {json.dumps(context.get('categorical_analysis', {}), indent=2, default=str)}
        
        USER INSTRUCTIONS:
        {context['user_instructions']}
        
        REQUIREMENTS FOR THE CODE:
        1. Start with all necessary imports (pandas, numpy, datetime, re, warnings)
        2. Create a function called 'clean_data(df)' that takes a DataFrame and returns cleaned DataFrame
        3. Handle all identified issues systematically:
           - For missing values: Use appropriate strategies (median for numeric, mode for categorical, forward fill for time series)
           - For duplicates: Remove exact duplicates, keep first occurrence
           - For outliers: Use IQR or Z-score method as appropriate
           - For inconsistent categories: Standardize text (strip, lower/upper case, fix common typos)
           - For data types: Convert to appropriate types based on schema
        4. Include error handling for each operation
        5. Add informative print statements for each major operation
        6. Preserve data integrity - make a copy before modifying
        7. Return the cleaned DataFrame
        
        IMPORTANT NOTES:
        - Do NOT use any external libraries beyond pandas, numpy, datetime, re, warnings
        - Do NOT use file I/O operations (no open, read, write)
        - Do NOT use exec, eval, or similar functions
        - Handle edge cases (empty DataFrames, all null columns, etc.)
        - Ensure all column names are properly quoted if they contain spaces or special characters
        
        Generate production-ready, well-documented code that addresses all issues comprehensively.
        """
        
        code = self.llm_client.generate_code(prompt)
        
        # Clean up the code
        code = code.strip()
        
        # Remove markdown code blocks if present
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        return code
    
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
            context_str = f"""
            Data shape: {context['data_shape']}
            Quality score: {context['quality_score']}/100
            Missing values: {context['quality_issues']['missing_values']['missing_percentage']:.1f}%
            Duplicates: {context['quality_issues']['duplicates']['duplicate_percentage']:.1f}%
            Outliers: {context['quality_issues']['outliers'].get('outlier_percentage', 0):.1f}%
            """
            
            summary = self.llm_client.summarize_strategy(code, context_str)
            return summary
        except Exception as e:
            # Fallback summary
            return (f"Comprehensive data cleaning to address: "
                   f"{context['quality_issues']['missing_values']['missing_percentage']:.1f}% missing values, "
                   f"{context['quality_issues']['duplicates']['duplicate_percentage']:.1f}% duplicates, "
                   f"and improve quality score from {context['quality_score']:.0f}/100")
    
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
                'Series': pd.Series,
                'Index': pd.Index,
                'MultiIndex': pd.MultiIndex,
                'Categorical': pd.Categorical,
                'cut': pd.cut,
                'qcut': pd.qcut,
                'get_dummies': pd.get_dummies,
                'factorize': pd.factorize
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
    
    def _generate_safe_fallback_code(self, data: pd.DataFrame, quality_report: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a safe fallback correction code"""
        
        missing_percentage = quality_report.get('missing_values', {}).get('missing_percentage', 0)
        duplicate_percentage = quality_report.get('duplicates', {}).get('duplicate_percentage', 0)
        
        code = f"""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    \"\"\"
    Safe data cleaning function with comprehensive error handling
    \"\"\"
    try:
        print(f"Starting data cleaning. Original shape: {{df.shape}}")
        print(f"Missing values: {missing_percentage:.1f}%")
        print(f"Duplicates: {duplicate_percentage:.1f}%")
        
        # Make a copy
        cleaned_df = df.copy()
        
        # Step 1: Remove duplicates
        if cleaned_df.duplicated().sum() > 0:
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            print(f"Removed {{initial_rows - len(cleaned_df)}} duplicate rows")
        
        # Step 2: Handle missing values
        for column in cleaned_df.columns:
            null_count = cleaned_df[column].isnull().sum()
            if null_count > 0:
                null_pct = (null_count / len(cleaned_df)) * 100
                
                # Drop column if >90% missing
                if null_pct > 90:
                    cleaned_df = cleaned_df.drop(columns=[column])
                    print(f"Dropped column '{{column}}' ({{null_pct:.1f}}% missing)")
                    continue
                
                # Fill based on data type
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    # Use median for numeric
                    median_val = cleaned_df[column].median()
                    if pd.notna(median_val):
                        cleaned_df[column].fillna(median_val, inplace=True)
                        print(f"Filled {{null_count}} missing values in '{{column}}' with median")
                elif pd.api.types.is_datetime64_any_dtype(cleaned_df[column]):
                    # Forward fill for datetime
                    cleaned_df[column].fillna(method='ffill', inplace=True)
                    cleaned_df[column].fillna(method='bfill', inplace=True)
                    print(f"Filled missing dates in '{{column}}' using forward/backward fill")
                else:
                    # Use mode for categorical
                    mode_val = cleaned_df[column].mode()
                    if not mode_val.empty:
                        cleaned_df[column].fillna(mode_val[0], inplace=True)
                        print(f"Filled {{null_count}} missing values in '{{column}}' with mode")
                    else:
                        cleaned_df[column].fillna('Unknown', inplace=True)
                        print(f"Filled {{null_count}} missing values in '{{column}}' with 'Unknown'")
        
        # Step 3: Basic outlier handling for numeric columns
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            Q1 = cleaned_df[column].quantile(0.25)
            Q3 = cleaned_df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)).sum()
                if outliers > 0:
                    # Cap outliers instead of removing
                    cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
                    print(f"Capped {{outliers}} outliers in '{{column}}'")
        
        # Step 4: Standardize text columns
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for column in text_columns:
            try:
                # Strip whitespace and standardize
                cleaned_df[column] = cleaned_df[column].astype(str).str.strip()
                # Remove extra spaces
                cleaned_df[column] = cleaned_df[column].str.replace(r'\\s+', ' ', regex=True)
                print(f"Standardized text in '{{column}}'")
            except:
                pass  # Skip if conversion fails
        
        print(f"Data cleaning completed. Final shape: {{cleaned_df.shape}}")
        print(f"Rows removed: {{df.shape[0] - cleaned_df.shape[0]}}")
        print(f"Columns removed: {{df.shape[1] - cleaned_df.shape[1]}}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error during data cleaning: {{str(e)}}")
        print("Returning original data")
        return df
"""
        
        return code
    
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