import pandas as pd
import json
import streamlit as st
from typing import Dict, Any, List
from .llm_interface import call_llm
from .code_executor import safe_execute_code
from .logger_service import log
from prompts.templates import get_prompt_template

def generate_schema_detection_code(dataframes: Dict[str, pd.DataFrame], model: str) -> str:
    """
    Generate Python code to detect schema from dataframes
    """
    # Prepare data samples
    data_samples = {}
    for sheet_name, df in dataframes.items():
        sample_data = df.head(5).to_dict('records')
        data_samples[sheet_name] = {
            'columns': list(df.columns),
            'sample_rows': sample_data,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'total_rows': len(df),
            'null_counts': df.isnull().sum().to_dict()
        }
    
    prompt = get_prompt_template('schema_detection').format(
        data_samples=json.dumps(data_samples, indent=2, default=str)
    )
    
    return call_llm(prompt, model)

def execute_schema_code(code: str, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Execute schema detection code safely
    """
    namespace = {
        'pd': pd,
        'json': json,
        'df_dict': dataframes
    }
    
    result = safe_execute_code(code, namespace)
    
    if 'schema_info' not in result:
        raise Exception("Schema detection code did not produce 'schema_info' variable")
    
    return result['schema_info']

def create_schema_editor(schema: Dict[str, Any], dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Create interactive schema editor UI with comprehensive error handling
    """
    # Validate inputs
    if schema is None:
        st.error("❌ Schema is None. Cannot create editor.")
        raise ValueError("Schema parameter cannot be None")
    
    if not isinstance(schema, dict):
        st.error("❌ Schema must be a dictionary.")
        raise ValueError("Schema parameter must be a dictionary")
    
    if not schema:
        st.warning("⚠️ Schema is empty. Please run schema detection first.")
        return {}
    
    edited_schema = {}
    
    for sheet_name, sheet_schema in schema.items():
        st.subheader(f"📊 {sheet_name}")
        
        # Validate sheet_schema
        if not isinstance(sheet_schema, dict):
            st.warning(f"⚠️ Invalid schema format for sheet '{sheet_name}'. Skipping.")
            continue
        
        edited_schema[sheet_name] = {}
        
        # Create columns for the editor
        col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
        
        with col1:
            st.write("**Column**")
        with col2:
            st.write("**Current Type**")
        with col3:
            st.write("**New Type**")
        with col4:
            st.write("**Format/Notes**")
        
        for col_name, col_info in sheet_schema.items():
            # Handle both dict and string formats
            if isinstance(col_info, dict):
                current_dtype = col_info.get('dtype', 'str')
                current_format = col_info.get('format', '')
                nullable = col_info.get('nullable', True)
                unique_values = col_info.get('unique_values', 0)
                recommendations = col_info.get('recommendations', '')
            else:
                # Handle simple string format
                current_dtype = str(col_info)
                current_format = ''
                nullable = True
                unique_values = 0
                recommendations = ''
            
            with col1:
                st.write(col_name)
            
            with col2:
                st.write(current_dtype)
            
            with col3:
                dtype_options = ['str', 'int', 'float', 'datetime', 'bool', 'categorical']
                try:
                    default_index = dtype_options.index(current_dtype) if current_dtype in dtype_options else 0
                except ValueError:
                    default_index = 0
                
                new_dtype = st.selectbox(
                    f"Type for {col_name}",
                    options=dtype_options,
                    index=default_index,
                    key=f"dtype_{sheet_name}_{col_name}",
                    label_visibility="collapsed"
                )
            
            with col4:
                format_info = st.text_input(
                    f"Format for {col_name}",
                    value=current_format,
                    placeholder="e.g., %Y-%m-%d for dates",
                    key=f"format_{sheet_name}_{col_name}",
                    label_visibility="collapsed"
                )
            
            # Update edited schema
            edited_schema[sheet_name][col_name] = {
                'dtype': new_dtype,
                'format': format_info,
                'nullable': nullable,
                'unique_values': unique_values,
                'recommendations': recommendations
            }
    
    return edited_schema

def apply_nl_instructions(schema: Dict[str, Any], instructions: str, model: str) -> Dict[str, Any]:
    """
    Apply natural language instructions to modify schema
    """
    try:
        prompt = get_prompt_template('schema_edit').format(
            current_schema=json.dumps(schema, indent=2),
            instructions=instructions
        )
        
        response = call_llm(prompt, model)
        
        # Extract JSON from response with multiple patterns
        import re
        
        # Try to find JSON block first
        json_patterns = [
            r'``````',  # JSON code block
            r'``````',      # Generic code block
            r'(\{[\s\S]*\})',              # Any JSON-like structure
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, response, re.DOTALL)
            if json_match:
                try:
                    updated_schema = json.loads(json_match.group(1))
                    return updated_schema
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, try to parse the entire response
        try:
            updated_schema = json.loads(response.strip())
            return updated_schema
        except json.JSONDecodeError:
            raise Exception("No valid JSON found in LLM response")
            
    except Exception as e:
        st.error(f"Failed to parse schema modifications: {str(e)}")
        st.write("**LLM Response:**")
        st.text(response if 'response' in locals() else "No response received")
        return schema

def validate_schema_structure(schema: Dict[str, Any]) -> bool:
    """
    Validate that the schema has the expected structure
    """
    if not isinstance(schema, dict):
        return False
    
    for sheet_name, sheet_schema in schema.items():
        if not isinstance(sheet_schema, dict):
            return False
        
        for col_name, col_info in sheet_schema.items():
            # Accept both dict and string formats
            if isinstance(col_info, dict):
                if 'dtype' not in col_info:
                    return False
            elif not isinstance(col_info, str):
                return False
    
    return True

def create_default_schema(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Create a default schema when detection fails
    """
    default_schema = {}
    
    for sheet_name, df in dataframes.items():
        default_schema[sheet_name] = {}
        
        for col_name in df.columns:
            # Simple type inference
            col_data = df[col_name]
            
            if pd.api.types.is_integer_dtype(col_data):
                dtype = 'int'
            elif pd.api.types.is_float_dtype(col_data):
                dtype = 'float'
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                dtype = 'datetime'
            elif pd.api.types.is_bool_dtype(col_data):
                dtype = 'bool'
            else:
                dtype = 'str'
            
            default_schema[sheet_name][col_name] = {
                'dtype': dtype,
                'format': '',
                'nullable': col_data.isnull().any(),
                'unique_values': col_data.nunique(),
                'recommendations': 'Auto-detected type'
            }
    
    return default_schema

def repair_schema(schema: Dict[str, Any], dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Repair a malformed schema by filling in missing information
    """
    if not validate_schema_structure(schema):
        return create_default_schema(dataframes)
    
    repaired_schema = {}
    
    for sheet_name, sheet_schema in schema.items():
        if sheet_name not in dataframes:
            continue
            
        repaired_schema[sheet_name] = {}
        df = dataframes[sheet_name]
        
        for col_name in df.columns:
            if col_name in sheet_schema:
                col_info = sheet_schema[col_name]
                
                if isinstance(col_info, dict):
                    repaired_schema[sheet_name][col_name] = col_info
                else:
                    # Convert string to dict format
                    repaired_schema[sheet_name][col_name] = {
                        'dtype': str(col_info),
                        'format': '',
                        'nullable': True,
                        'unique_values': df[col_name].nunique(),
                        'recommendations': ''
                    }
            else:
                # Add missing column with default info
                repaired_schema[sheet_name][col_name] = {
                    'dtype': 'str',
                    'format': '',
                    'nullable': True,
                    'unique_values': df[col_name].nunique(),
                    'recommendations': 'Added during repair'
                }
    
    return repaired_schema
