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
    Display analysis results in a user-friendly format with safe None handling
    """
    st.subheader("📊 Data Quality Analysis Results")
    
    # ✅ SAFE CHECK: Handle None or empty analysis_results
    if not analysis_results:
        st.warning("⚠️ No analysis results available to display.")
        st.info("Please run the data analysis first to generate results.")
        return
    
    # ✅ SAFE CHECK: Ensure analysis_results is a dictionary
    if not isinstance(analysis_results, dict):
        st.error("❌ Invalid analysis results format. Expected dictionary.")
        st.write(f"Received: {type(analysis_results)}")
        return
    
    # Summary metrics - SAFELY CHECK FOR 'summary' KEY
    if analysis_results.get('summary'):
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
    
    # Sheet-level analysis - SAFELY ITERATE THROUGH RESULTS
    for sheet_name, sheet_analysis in analysis_results.items():
        if sheet_name == 'summary':
            continue
        
        # ✅ SAFE CHECK: Ensure sheet_analysis is a dictionary
        if not isinstance(sheet_analysis, dict):
            st.warning(f"⚠️ Skipping {sheet_name}: Invalid analysis format")
            continue
            
        with st.expander(f"📋 {sheet_name} Analysis", expanded=True):
            
            # Missing values chart - SAFE CHECK
            missing_data = sheet_analysis.get('missing_analysis', {})
            if missing_data and any(missing_data.values()):
                st.subheader("Missing Values")
                try:
                    fig = px.bar(
                        x=list(missing_data.keys()),
                        y=list(missing_data.values()),
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display missing values chart: {str(e)}")
            
            # Outliers - SAFE CHECK
            outliers = sheet_analysis.get('outliers', [])
            if outliers:
                st.subheader("Outliers Detected")
                try:
                    if isinstance(outliers, list) and outliers:
                        outlier_df = pd.DataFrame(outliers)
                        st.dataframe(outlier_df)
                    elif isinstance(outliers, dict):
                        st.json(outliers)
                    else:
                        st.write(outliers)
                except Exception as e:
                    st.warning(f"Could not display outliers: {str(e)}")
            
            # Data quality issues - SAFE CHECK
            issues = sheet_analysis.get('quality_issues', [])
            if issues:
                st.subheader("Data Quality Issues")
                for issue in issues:
                    if issue:  # Only display non-empty issues
                        st.warning(f"⚠️ {issue}")
            
            # Recommendations - SAFE CHECK
            recommendations = sheet_analysis.get('recommendations', [])
            if recommendations:
                st.subheader("Recommendations")
                for rec in recommendations:
                    if rec:  # Only display non-empty recommendations
                        st.info(f"💡 {rec}")

def validate_again(profile_code: str, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Re-run analysis on cleaned data for validation
    """
    return execute_analysis(profile_code, dataframes)
