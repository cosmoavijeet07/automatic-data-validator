import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, Optional
from .llm_interface import call_llm
from .code_executor import safe_execute_code
from .logger_service import log
from prompts.templates import get_prompt_template

def generate_analysis_code(schema: Dict[str, Any], model: str, additional_instructions: str = "") -> str:
    """
    Generate comprehensive data analysis code
    """
    prompt = get_prompt_template('data_analysis').format(
        schema=json.dumps(schema, indent=2),
        additional_instructions=additional_instructions
    )
    
    return call_llm(prompt, model)

def execute_analysis(code: str, dataframes: Dict[str, pd.DataFrame], user_instructions: str = "") -> Dict[str, Any]:
    """
    Execute data analysis code safely
    """
    namespace = {
        'pd': pd,
        'np': np,
        'px': px,
        'go': go,
        'df_dict': dataframes,
        'user_instructions': user_instructions
    }
    
    result = safe_execute_code(code, namespace)
    
    if 'analysis_results' not in result:
        raise Exception("Analysis code did not produce 'analysis_results' variable")
    
    return result['analysis_results']

def display_analysis_results(analysis_results: Dict[str, Any]):
    """
    Display analysis results in a user-friendly format
    """
    st.subheader("📊 Data Quality Analysis Results")
    
    # Summary metrics
    if 'summary' in analysis_results:
        summary = analysis_results['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", summary.get('total_records', 0))
        with col2:
            st.metric("Total Columns", summary.get('total_columns', 0))
        with col3:
            st.metric("Missing Values", summary.get('total_missing', 0))
        with col4:
            st.metric("Duplicates", summary.get('total_duplicates', 0))
    
    # Sheet-level analysis
    for sheet_name, sheet_analysis in analysis_results.items():
        if sheet_name == 'summary':
            continue
            
        with st.expander(f"📋 {sheet_name} Analysis", expanded=True):
            
            # Missing values chart
            if 'missing_analysis' in sheet_analysis:
                missing_data = sheet_analysis['missing_analysis']
                if any(missing_data.values()):
                    st.subheader("Missing Values")
                    fig = px.bar(
                        x=list(missing_data.keys()),
                        y=list(missing_data.values()),
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Outliers
            if 'outliers' in sheet_analysis:
                outliers = sheet_analysis['outliers']
                if outliers:
                    st.subheader("Outliers Detected")
                    outlier_df = pd.DataFrame(outliers)
                    st.dataframe(outlier_df)
            
            # Data quality issues
            if 'quality_issues' in sheet_analysis:
                issues = sheet_analysis['quality_issues']
                if issues:
                    st.subheader("Data Quality Issues")
                    for issue in issues:
                        st.warning(f"⚠️ {issue}")
            
            # Recommendations
            if 'recommendations' in sheet_analysis:
                recommendations = sheet_analysis['recommendations']
                if recommendations:
                    st.subheader("Recommendations")
                    for rec in recommendations:
                        st.info(f"💡 {rec}")

def validate_again(profile_code: str, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Re-run analysis on cleaned data for validation
    """
    return execute_analysis(profile_code, dataframes)
