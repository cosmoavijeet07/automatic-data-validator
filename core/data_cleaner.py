import pandas as pd
import numpy as np
import json
import streamlit as st
from typing import Dict, Any, Optional
from .llm_interface import call_llm
from .code_executor import safe_execute_code
from .logger_service import log
from prompts.templates import get_prompt_template

class CodeError(Exception):
    def __init__(self, trace: str):
        self.trace = trace
        super().__init__(trace)

def generate_cleaning_code(schema: Dict[str, Any], analysis_results: Dict[str, Any], model: str, additional_instructions: str = "") -> str:
    """
    Generate data cleaning code based on schema and analysis
    """
    prompt = get_prompt_template('data_cleaning').format(
        schema=json.dumps(schema, indent=2),
        analysis_results=json.dumps(analysis_results, indent=2, default=str),
        user_feedback=additional_instructions
    )
    
    return call_llm(prompt, model)

def execute_cleaning(code: str, dataframes: Dict[str, pd.DataFrame], user_instructions: str = "") -> Dict[str, pd.DataFrame]:
    """
    Execute data cleaning code safely
    """
    namespace = {
        'pd': pd,
        'np': np,
        'df_dict': dataframes.copy(),  # Work on a copy
        'user_instructions': user_instructions
    }
    
    try:
        result = safe_execute_code(code, namespace)
    except Exception as e:
        import traceback
        raise CodeError(traceback.format_exc())
    
    # Look for cleaned dataframes in various possible variable names
    cleaned_data = None
    for var_name in ['cleaned_dataframes', 'clean_df_dict', 'cleaned_data', 'df_dict']:
        if var_name in result:
            cleaned_data = result[var_name]
            break
    
    if cleaned_data is None:
        raise Exception("Cleaning code did not produce cleaned dataframes")
    
    # If single dataframe returned, convert to dict format
    if isinstance(cleaned_data, pd.DataFrame):
        cleaned_data = {'Sheet1': cleaned_data}
    
    return cleaned_data

def fix_cleaning_code(original_code: str, error_message: str, model: str) -> str:
    """
    Fix cleaning code that encountered errors
    """
    prompt = get_prompt_template('code_fix').format(
        original_code=original_code,
        error_message=error_message,
        traceback=error_message
    )
    
    return call_llm(prompt, model)

def create_final_pipeline(schema_code: str, analysis_code: str, cleaning_code: str, model: str) -> str:
    """
    Combine all code into a single reusable pipeline
    """
    prompt = get_prompt_template('pipeline_combination').format(
        schema_code=schema_code,
        analysis_code=analysis_code,
        cleaning_code=cleaning_code
    )
    
    return call_llm(prompt, model)
