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
    Create interactive schema editor UI
    """
    edited_schema = {}
    
    for sheet_name, sheet_schema in schema.items():
        st.subheader(f"📊 {sheet_name}")
        
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
            current_dtype = col_info.get('dtype', 'str')
            
            with col1:
                st.write(col_name)
            
            with col2:
                st.write(current_dtype)
            
            with col3:
                new_dtype = st.selectbox(
                    f"Type for {col_name}",
                    options=['str', 'int', 'float', 'datetime', 'bool', 'categorical'],
                    index=['str', 'int', 'float', 'datetime', 'bool', 'categorical'].index(current_dtype) if current_dtype in ['str', 'int', 'float', 'datetime', 'bool', 'categorical'] else 0,
                    key=f"dtype_{sheet_name}_{col_name}",
                    label_visibility="collapsed"
                )
            
            with col4:
                format_info = st.text_input(
                    f"Format for {col_name}",
                    value=col_info.get('format', ''),
                    placeholder="e.g., %Y-%m-%d for dates",
                    key=f"format_{sheet_name}_{col_name}",
                    label_visibility="collapsed"
                )
            
            # Update edited schema
            edited_schema[sheet_name][col_name] = {
                'dtype': new_dtype,
                'format': format_info,
                'nullable': col_info.get('nullable', True),
                'unique_values': col_info.get('unique_values', 0),
                'recommendations': col_info.get('recommendations', '')
            }
    
    return edited_schema

def apply_nl_instructions(schema: Dict[str, Any], instructions: str, model: str) -> Dict[str, Any]:
    """
    Apply natural language instructions to modify schema
    """
    prompt = get_prompt_template('schema_edit').format(
        current_schema=json.dumps(schema, indent=2),
        instructions=instructions
    )
    
    response = call_llm(prompt, model)
    
    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            updated_schema = json.loads(json_match.group())
            return updated_schema
        else:
            raise Exception("No valid JSON found in LLM response")
    except Exception as e:
        st.error(f"Failed to parse schema modifications: {str(e)}")
        return schema
