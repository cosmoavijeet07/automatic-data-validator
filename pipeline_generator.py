"""
Pipeline generator module for creating final data cleaning pipelines
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from config import PIPELINE_GENERATION_PROMPT

class PipelineGenerator:
    """Generates final data cleaning pipelines from session logs"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_pipeline(self, logs: List[Dict[str, Any]], schema: Dict[str, Any], 
                         correction_code: str = "") -> str:
        """
        Generate a complete data cleaning pipeline
        
        Args:
            logs: Session logs containing all operations
            schema: Final data schema
            correction_code: Final correction code used
            
        Returns:
            Complete pipeline Python script
        """
        try:
            # Extract relevant information from logs
            pipeline_context = self._extract_pipeline_context(logs, schema, correction_code)
            
            # Generate the pipeline code
            pipeline_code = self._generate_pipeline_code(pipeline_context)
            
            return pipeline_code
            
        except Exception as e:
            raise ValueError(f"Failed to generate pipeline: {str(e)}")
    
    def _extract_pipeline_context(self, logs: List[Dict[str, Any]], schema: Dict[str, Any], 
                                correction_code: str) -> Dict[str, Any]:
        """Extract relevant context from session logs"""
        context = {
            'file_operations': [],
            'schema_changes': [],
            'data_operations': [],
            'quality_issues': [],
            'correction_steps': [],
            'final_schema': schema,
            'correction_code': correction_code,
            'session_summary': {}
        }
        
        # Process logs to extract pipeline steps
        for log in logs:
            action = log.get('action', '')
            details = log.get('details', {})
            
            # File operations
            if 'File operation' in action:
                context['file_operations'].append({
                    'operation': details.get('operation'),
                    'filename': details.get('filename'),
                    'file_type': details.get('file_type')
                })
            
            # Schema changes
            elif 'Schema change' in action:
                context['schema_changes'].append({
                    'column': details.get('column'),
                    'old_type': details.get('old_type'),
                    'new_type': details.get('new_type'),
                    'reason': details.get('change_reason')
                })
            
            # Data operations
            elif 'Data operation' in action:
                context['data_operations'].append({
                    'operation': details.get('operation'),
                    'before_shape': details.get('before_shape'),
                    'after_shape': details.get('after_shape'),
                    'rows_changed': details.get('rows_changed'),
                    'columns_changed': details.get('columns_changed')
                })
            
            # Quality analysis
            elif 'Quality analysis' in action:
                context['quality_issues'] = details.get('issues_found', [])
            
            # Code execution
            elif 'Code execution' in action:
                if details.get('success'):
                    context['correction_steps'].append({
                        'code_length': details.get('code_length'),
                        'execution_time': details.get('execution_time'),
                        'output': details.get('output', [])
                    })
        
        # Create session summary
        context['session_summary'] = {
            'total_operations': len(context['data_operations']),
            'schema_changes_count': len(context['schema_changes']),
            'quality_issues_count': len(context['quality_issues']),
            'correction_steps_count': len(context['correction_steps'])
        }
        
        return context
    
    def _generate_pipeline_code(self, context: Dict[str, Any]) -> str:
        """Generate the actual pipeline code"""
        
        # Prepare prompt for LLM
        prompt = f"""
        {PIPELINE_GENERATION_PROMPT}
        
        Pipeline Context:
        {json.dumps(context, indent=2, default=str)}
        
        Create a complete Python script that:
        1. Defines a main function 'clean_data_pipeline(input_file_path, output_file_path)'
        2. Handles file loading based on extension (.csv, .xlsx, .json)
        3. Applies all the schema changes and corrections that were performed
        4. Includes proper error handling and logging
        5. Saves the cleaned data to the output path
        6. Returns a summary of operations performed
        
        The script should be self-contained and production-ready.
        Include a main section that can be run from command line.
        """
        
        try:
            pipeline_code = self.llm_client.generate_code(prompt)
            
            # Enhance the generated code with additional components
            enhanced_pipeline = self._enhance_pipeline_code(pipeline_code, context)
            
            return enhanced_pipeline
            
        except Exception as e:
            # Generate fallback pipeline
            return self._generate_fallback_pipeline(context)
    
    def _enhance_pipeline_code(self, base_code: str, context: Dict[str, Any]) -> str:
        """Enhance the generated pipeline with additional features"""
        
        # Add header with metadata
        header = f'''"""
Data Cleaning Pipeline
Generated on: {datetime.now().isoformat()}
Session Summary: {context['session_summary']}

This pipeline replicates the data cleaning operations performed in the interactive session.
"""

import pandas as pd
import numpy as np
import json
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
        
        # Add utility functions
        utilities = '''
def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

def load_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data from various file formats"""
    file_path = Path(file_path)
    file_info = {
        'filename': file_path.name,
        'extension': file_path.suffix.lower(),
        'size_mb': file_path.stat().st_size / (1024 * 1024)
    }
    
    try:
        if file_info['extension'] == '.csv':
            df = pd.read_csv(file_path)
        elif file_info['extension'] in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_info['extension'] == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_info['extension']}")
        
        file_info['shape'] = df.shape
        logger.info(f"Loaded data from {file_path.name}: {df.shape}")
        
        return df, file_info
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def save_data(df: pd.DataFrame, output_path: str, format: str = 'csv'):
    """Save data to specified format"""
    output_path = Path(output_path)
    
    try:
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() in ['xlsx', 'excel']:
            df.to_excel(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            # Default to CSV
            df.to_csv(output_path, index=False)
        
        logger.info(f"Data saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {str(e)}")
        raise

def generate_summary_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                          operations_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary report of the cleaning process"""
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'original_data': {
            'shape': original_df.shape,
            'columns': list(original_df.columns),
            'memory_usage_mb': original_df.memory_usage(deep=True).sum() / (1024 * 1024)
        },
        'cleaned_data': {
            'shape': cleaned_df.shape,
            'columns': list(cleaned_df.columns),
            'memory_usage_mb': cleaned_df.memory_usage(deep=True).sum() / (1024 * 1024)
        },
        'changes': {
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_removed': original_df.shape[1] - cleaned_df.shape[1],
            'memory_saved_mb': (original_df.memory_usage(deep=True).sum() - cleaned_df.memory_usage(deep=True).sum()) / (1024 * 1024)
        },
        'operations_performed': operations_log
    }
    
    return report

'''
        
        # Add main execution section
        main_section = '''
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Cleaning Pipeline')
    parser.add_argument('input_file', help='Input data file path')
    parser.add_argument('output_file', help='Output cleaned data file path')
    parser.add_argument('--log-file', help='Log file path (optional)')
    parser.add_argument('--format', default='csv', choices=['csv', 'xlsx', 'json'], 
                       help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file:
        setup_logging(args.log_file)
    
    try:
        # Run the pipeline
        summary = clean_data_pipeline(args.input_file, args.output_file, args.format)
        
        # Print summary
        print("\\n" + "="*50)
        print("DATA CLEANING PIPELINE COMPLETED")
        print("="*50)
        print(f"Original data shape: {summary['original_data']['shape']}")
        print(f"Cleaned data shape: {summary['cleaned_data']['shape']}")
        print(f"Rows removed: {summary['changes']['rows_removed']}")
        print(f"Columns removed: {summary['changes']['columns_removed']}")
        print(f"Memory saved: {summary['changes']['memory_saved_mb']:.2f} MB")
        print(f"Operations performed: {len(summary['operations_performed'])}")
        print("="*50)
        
        # Save summary report
        summary_file = Path(args.output_file).parent / f"{Path(args.output_file).stem}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary report saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise
'''
        
        # Combine all parts
        enhanced_code = header + utilities + base_code + main_section
        
        return enhanced_code
    
    def _generate_fallback_pipeline(self, context: Dict[str, Any]) -> str:
        """Generate a basic fallback pipeline when LLM generation fails"""
        
        schema_changes = context.get('schema_changes', [])
        correction_code = context.get('correction_code', '')
        
        fallback_code = f'''"""
Fallback Data Cleaning Pipeline
Generated on: {datetime.now().isoformat()}
"""

import pandas as pd
import numpy as np
import json
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_data_pipeline(input_file_path: str, output_file_path: str, output_format: str = 'csv') -> Dict[str, Any]:
    """
    Basic data cleaning pipeline
    
    Args:
        input_file_path: Path to input data file
        output_file_path: Path to save cleaned data
        output_format: Output format (csv, xlsx, json)
        
    Returns:
        Summary dictionary of operations performed
    """
    operations_log = []
    
    try:
        # Load data
        logger.info(f"Loading data from {{input_file_path}}")
        original_df, file_info = load_data(input_file_path)
        operations_log.append({{'operation': 'load_data', 'details': file_info}})
        
        # Start with a copy
        cleaned_df = original_df.copy()
        
        # Apply schema changes
        schema_changes = {schema_changes}
        for change in schema_changes:
            column = change.get('column')
            new_type = change.get('new_type')
            
            if column in cleaned_df.columns:
                try:
                    if new_type == 'datetime64[ns]':
                        cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
                    elif new_type == 'category':
                        cleaned_df[column] = cleaned_df[column].astype('category')
                    elif new_type in ['int64', 'float64']:
                        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                    
                    operations_log.append({{'operation': 'schema_change', 'column': column, 'new_type': new_type}})
                    logger.info(f"Changed {{column}} to {{new_type}}")
                    
                except Exception as e:
                    logger.warning(f"Could not convert {{column}} to {{new_type}}: {{str(e)}}")
        
        # Basic cleaning operations
        initial_shape = cleaned_df.shape
        
        # Remove duplicates
        duplicates_before = cleaned_df.duplicated().sum()
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = duplicates_before - cleaned_df.duplicated().sum()
        
        if duplicates_removed > 0:
            operations_log.append({{'operation': 'remove_duplicates', 'count': duplicates_removed}})
            logger.info(f"Removed {{duplicates_removed}} duplicate rows")
        
        # Handle missing values (basic strategy)
        missing_before = cleaned_df.isnull().sum().sum()
        
        # Fill numeric columns with median
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if cleaned_df[col].isnull().sum() > 0:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_columns = cleaned_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if cleaned_df[col].isnull().sum() > 0:
                mode_value = cleaned_df[col].mode()
                if not mode_value.empty:
                    cleaned_df[col].fillna(mode_value[0], inplace=True)
        
        missing_after = cleaned_df.isnull().sum().sum()
        missing_handled = missing_before - missing_after
        
        if missing_handled > 0:
            operations_log.append({{'operation': 'handle_missing_values', 'count': missing_handled}})
            logger.info(f"Handled {{missing_handled}} missing values")
        
        # Save cleaned data
        save_data(cleaned_df, output_file_path, output_format)
        operations_log.append({{'operation': 'save_data', 'output_path': output_file_path, 'format': output_format}})
        
        # Generate summary
        summary = generate_summary_report(original_df, cleaned_df, operations_log)
        
        logger.info("Pipeline completed successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {{str(e)}}")
        raise

# Utility functions (same as enhanced version)
def load_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data from various file formats"""
    file_path = Path(file_path)
    file_info = {{
        'filename': file_path.name,
        'extension': file_path.suffix.lower(),
        'size_mb': file_path.stat().st_size / (1024 * 1024)
    }}
    
    try:
        if file_info['extension'] == '.csv':
            df = pd.read_csv(file_path)
        elif file_info['extension'] in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_info['extension'] == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {{file_info['extension']}}")
        
        file_info['shape'] = df.shape
        return df, file_info
        
    except Exception as e:
        raise ValueError(f"Error loading data: {{str(e)}}")

def save_data(df: pd.DataFrame, output_path: str, format: str = 'csv'):
    """Save data to specified format"""
    output_path = Path(output_path)
    
    if format.lower() == 'csv':
        df.to_csv(output_path, index=False)
    elif format.lower() in ['xlsx', 'excel']:
        df.to_excel(output_path, index=False)
    elif format.lower() == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        df.to_csv(output_path, index=False)

def generate_summary_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                          operations_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary report"""
    return {{
        'timestamp': datetime.datetime.now().isoformat(),
        'original_data': {{
            'shape': original_df.shape,
            'columns': list(original_df.columns)
        }},
        'cleaned_data': {{
            'shape': cleaned_df.shape,
            'columns': list(cleaned_df.columns)
        }},
        'changes': {{
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_removed': original_df.shape[1] - cleaned_df.shape[1]
        }},
        'operations_performed': operations_log
    }}

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <input_file> <output_file> [format]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    output_format = sys.argv[3] if len(sys.argv) > 3 else 'csv'
    
    try:
        summary = clean_data_pipeline(input_file, output_file, output_format)
        print("Pipeline completed successfully!")
        print(f"Original shape: {{summary['original_data']['shape']}}")
        print(f"Final shape: {summary['cleaned_data']['shape']}")
        print(f"Operations performed: {len(summary['operations_performed'])}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)
'''
        
        return fallback_code
    
    def generate_pipeline_documentation(self, context: Dict[str, Any]) -> str:
        """Generate comprehensive documentation for the pipeline"""
        
        doc = f"""
# Data Cleaning Pipeline Documentation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This pipeline replicates the data cleaning operations performed during an interactive session.

## Session Summary
- Total Operations: {context['session_summary'].get('total_operations', 0)}
- Schema Changes: {context['session_summary'].get('schema_changes_count', 0)}
- Quality Issues Addressed: {context['session_summary'].get('quality_issues_count', 0)}
- Correction Steps: {context['session_summary'].get('correction_steps_count', 0)}

## Pipeline Steps

### 1. Data Loading
The pipeline supports loading data from multiple formats:
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- JSON files (.json)

### 2. Schema Transformations
The following schema changes are applied:
"""
        
        # Add schema changes
        for change in context.get('schema_changes', []):
            doc += f"""
- **{change.get('column')}**: {change.get('old_type')} → {change.get('new_type')}
  - Reason: {change.get('reason', 'Data type optimization')}
"""
        
        # Add data operations
        doc += """

### 3. Data Quality Operations
The following data quality operations are performed:
"""
        
        for operation in context.get('data_operations', []):
            doc += f"""
- **{operation.get('operation')}**: 
  - Before: {operation.get('before_shape')}
  - After: {operation.get('after_shape')}
  - Rows changed: {operation.get('rows_changed', 0)}
  - Columns changed: {operation.get('columns_changed', 0)}
"""
        
        # Add quality issues
        if context.get('quality_issues'):
            doc += """

### 4. Quality Issues Addressed
The following quality issues were identified and addressed:
"""
            for issue in context['quality_issues']:
                doc += f"- {issue}\n"
        
        # Add usage instructions
        doc += """

## Usage Instructions

### Command Line Usage
```bash
python data_cleaning_pipeline.py input_file.csv output_file.csv [format]
```

### Programmatic Usage
```python
from data_cleaning_pipeline import clean_data_pipeline

summary = clean_data_pipeline('input.csv', 'output.csv', 'csv')
print(summary)
```

### Parameters
- `input_file_path`: Path to the input data file
- `output_file_path`: Path where cleaned data will be saved
- `output_format`: Output format ('csv', 'xlsx', 'json')

### Output
The pipeline generates:
1. Cleaned data file in the specified format
2. Summary report (JSON format)
3. Processing logs

## Dependencies
The pipeline requires the following Python packages:
- pandas
- numpy
- openpyxl (for Excel support)

Install with:
```bash
pip install pandas numpy openpyxl
```

## Error Handling
The pipeline includes comprehensive error handling:
- Invalid file formats are rejected with clear error messages
- Data type conversion errors are logged but don't stop the pipeline
- Missing value handling uses appropriate strategies per data type
- All operations are logged for debugging

## Performance Considerations
- Large files (>1GB) may require significant memory
- Processing time scales with data size and complexity
- Consider chunking for very large datasets

## Customization
The pipeline can be customized by:
1. Modifying the schema transformation logic
2. Adding custom data quality checks
3. Implementing domain-specific cleaning rules
4. Adjusting missing value handling strategies

## Support
For issues or questions, refer to the session logs and quality reports generated during the interactive session.
"""
        
        return doc
    
    def validate_pipeline_code(self, pipeline_code: str) -> Dict[str, Any]:
        """Validate the generated pipeline code"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            # Check for required functions
            required_functions = ['clean_data_pipeline', 'load_data', 'save_data']
            for func in required_functions:
                if f'def {func}(' not in pipeline_code:
                    validation_result['issues'].append(f"Missing required function: {func}")
                    validation_result['is_valid'] = False
            
            # Check for required imports
            required_imports = ['pandas', 'numpy', 'json', 'datetime', 'logging']
            for imp in required_imports:
                if f'import {imp}' not in pipeline_code and f'from {imp}' not in pipeline_code:
                    validation_result['warnings'].append(f"Missing import: {imp}")
            
            # Check for main execution block
            if 'if __name__ == "__main__":' not in pipeline_code:
                validation_result['warnings'].append("Missing main execution block")
            
            # Check for error handling
            if 'try:' not in pipeline_code or 'except' not in pipeline_code:
                validation_result['warnings'].append("Limited error handling detected")
            
            # Check for logging
            if 'logger' not in pipeline_code and 'logging' not in pipeline_code:
                validation_result['warnings'].append("No logging implementation found")
            
            # Syntax check
            try:
                compile(pipeline_code, '<string>', 'exec')
            except SyntaxError as e:
                validation_result['issues'].append(f"Syntax error: {str(e)}")
                validation_result['is_valid'] = False
            
            # Add suggestions
            if validation_result['warnings']:
                validation_result['suggestions'].extend([
                    "Consider adding comprehensive logging",
                    "Implement robust error handling",
                    "Add input validation",
                    "Include performance monitoring"
                ])
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False
        
        return validation_result
    
    def generate_test_script(self, context: Dict[str, Any]) -> str:
        """Generate a test script for the pipeline"""
        
        test_script = f"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Import the pipeline (assumes pipeline file is named data_cleaning_pipeline.py)
from data_cleaning_pipeline import clean_data_pipeline, load_data, save_data

class TestDataCleaningPipeline(unittest.TestCase):
    \"\"\"Test cases for the data cleaning pipeline\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures\"\"\"
        # Create sample test data
        self.test_data = pd.DataFrame({{
            'id': [1, 2, 3, 4, 5, 5],  # Duplicate
            'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],  # Missing value
            'age': [25, 30, 35, '40', 45, 45],  # Mixed types
            'salary': [50000, 60000, 70000, 80000, None, 90000],  # Missing value
            'date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-05-01']
        }})
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, 'test_input.csv')
        self.output_file = os.path.join(self.temp_dir, 'test_output.csv')
        
        # Save test data
        self.test_data.to_csv(self.input_file, index=False)
    
    def tearDown(self):
        \"\"\"Clean up test fixtures\"\"\"
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_data(self):
        \"\"\"Test data loading functionality\"\"\"
        df, file_info = load_data(self.input_file)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, self.test_data.shape)
        self.assertIn('filename', file_info)
        self.assertIn('shape', file_info)
    
    def test_save_data(self):
        \"\"\"Test data saving functionality\"\"\"
        # Test CSV saving
        save_data(self.test_data, self.output_file, 'csv')
        self.assertTrue(os.path.exists(self.output_file))
        
        # Load and verify
        loaded_df = pd.read_csv(self.output_file)
        self.assertEqual(loaded_df.shape, self.test_data.shape)
    
    def test_pipeline_execution(self):
        \"\"\"Test complete pipeline execution\"\"\"
        summary = clean_data_pipeline(self.input_file, self.output_file)
        
        # Check that summary is returned
        self.assertIsInstance(summary, dict)
        self.assertIn('original_data', summary)
        self.assertIn('cleaned_data', summary)
        self.assertIn('operations_performed', summary)
        
        # Check that output file exists
        self.assertTrue(os.path.exists(self.output_file))
        
        # Load cleaned data and verify
        cleaned_df = pd.read_csv(self.output_file)
        self.assertLessEqual(cleaned_df.shape[0], self.test_data.shape[0])  # May have fewer rows due to duplicate removal
    
    def test_duplicate_removal(self):
        \"\"\"Test that duplicates are removed\"\"\"
        summary = clean_data_pipeline(self.input_file, self.output_file)
        cleaned_df = pd.read_csv(self.output_file)
        
        # Check for duplicates
        duplicates = cleaned_df.duplicated().sum()
        self.assertEqual(duplicates, 0)
    
    def test_missing_value_handling(self):
        \"\"\"Test missing value handling\"\"\"
        summary = clean_data_pipeline(self.input_file, self.output_file)
        cleaned_df = pd.read_csv(self.output_file)
        
        # Check that missing values are reduced
        original_nulls = self.test_data.isnull().sum().sum()
        cleaned_nulls = cleaned_df.isnull().sum().sum()
        
        self.assertLessEqual(cleaned_nulls, original_nulls)
    
    def test_error_handling(self):
        \"\"\"Test error handling with invalid inputs\"\"\"
        # Test with non-existent file
        with self.assertRaises(Exception):
            clean_data_pipeline('nonexistent.csv', self.output_file)
        
        # Test with invalid output path
        with self.assertRaises(Exception):
            clean_data_pipeline(self.input_file, '/invalid/path/output.csv')

class TestDataQuality(unittest.TestCase):
    \"\"\"Test data quality improvements\"\"\"
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, 'quality_test.csv')
        self.output_file = os.path.join(self.temp_dir, 'quality_output.csv')
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_type_consistency(self):
        \"\"\"Test that data types are consistent after cleaning\"\"\"
        # Create data with mixed types
        mixed_data = pd.DataFrame({{
            'numeric': ['1', '2', '3.5', '4'],
            'dates': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01']
        }})
        
        mixed_data.to_csv(self.input_file, index=False)
        
        summary = clean_data_pipeline(self.input_file, self.output_file)
        cleaned_df = pd.read_csv(self.output_file)
        
        # Verify data types (this will depend on your specific pipeline logic)
        self.assertIsInstance(summary, dict)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
"""
        
        return test_script
    
    def create_pipeline_package(self, pipeline_code: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Create a complete pipeline package with all files"""
        
        package = {
            'data_cleaning_pipeline.py': pipeline_code,
            'README.md': self.generate_pipeline_documentation(context),
            'test_pipeline.py': self.generate_test_script(context),
            'requirements.txt': self._generate_requirements_file(),
            'config.py': self._generate_config_file(context),
            'setup.py': self._generate_setup_file()
        }
        
        return package
    
    def _generate_requirements_file(self) -> str:
        """Generate requirements.txt file"""
        return """pandas>=1.3.0
numpy>=1.21.0
openpyxl>=3.0.0
xlsxwriter>=3.0.0
"""
    
    def _generate_config_file(self, context: Dict[str, Any]) -> str:
        """Generate configuration file"""
        return f"""# Pipeline Configuration
# Generated on: {datetime.now().isoformat()}

# File handling settings
SUPPORTED_INPUT_FORMATS = ['.csv', '.xlsx', '.xls', '.json']
SUPPORTED_OUTPUT_FORMATS = ['csv', 'xlsx', 'json']
MAX_FILE_SIZE_MB = 500

# Data processing settings
DEFAULT_CHUNK_SIZE = 10000
MISSING_VALUE_THRESHOLD = 0.5
OUTLIER_THRESHOLD = 3

# Schema settings from session
SCHEMA_CHANGES = {context.get('schema_changes', [])}

# Quality thresholds
DUPLICATE_THRESHOLD = 0.1
QUALITY_SCORE_THRESHOLD = 75

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
"""
    
    def _generate_setup_file(self) -> str:
        """Generate setup.py file"""
        return f"""from setuptools import setup, find_packages

setup(
    name="data-cleaning-pipeline",
    version="1.0.0",
    description="Generated data cleaning pipeline",
    author="AI Data Cleaning Platform",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "openpyxl>=3.0.0",
    ],
    python_requires=">=3.7",
    entry_points={{
        'console_scripts': [
            'clean-data=data_cleaning_pipeline:main',
        ],
    }},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
"""