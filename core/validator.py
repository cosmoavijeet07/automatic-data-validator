import pandas as pd
import numpy as np
import json
import streamlit as st
from typing import Dict, Any, Tuple
from .data_profiler import execute_analysis
from .logger_service import log

def validate_cleaned_data(cleaned_data: Dict[str, pd.DataFrame], analysis_code: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate cleaned data against original analysis
    """
    try:
        # Re-run analysis on cleaned data
        validation_results = execute_analysis(analysis_code, cleaned_data)
        
        # Compare with expected schema
        schema_validation = validate_schema_compliance(cleaned_data, schema)
        
        # Combine results
        validation_results['schema_compliance'] = schema_validation
        
        return validation_results
        
    except Exception as e:
        st.error(f"Validation failed: {str(e)}")
        return {'error': str(e)}

def validate_schema_compliance(dataframes: Dict[str, pd.DataFrame], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if cleaned data complies with expected schema
    """
    compliance_results = {}
    
    for sheet_name, df in dataframes.items():
        if sheet_name not in schema:
            continue
            
        sheet_schema = schema[sheet_name]
        sheet_compliance = {
            'column_count_match': True,
            'dtype_compliance': {},
            'missing_columns': [],
            'extra_columns': [],
            'dtype_mismatches': []
        }
        
        # Check columns
        expected_columns = set(sheet_schema.keys())
        actual_columns = set(df.columns)
        
        sheet_compliance['missing_columns'] = list(expected_columns - actual_columns)
        sheet_compliance['extra_columns'] = list(actual_columns - expected_columns)
        sheet_compliance['column_count_match'] = len(sheet_compliance['missing_columns']) == 0 and len(sheet_compliance['extra_columns']) == 0
        
        # Check data types
        for col_name, col_schema in sheet_schema.items():
            if col_name in df.columns:
                expected_dtype = col_schema.get('dtype', 'str')
                actual_dtype = str(df[col_name].dtype)
                
                # Simple dtype mapping
                dtype_mapping = {
                    'int': ['int64', 'int32', 'int16', 'int8'],
                    'float': ['float64', 'float32'],
                    'str': ['object'],
                    'datetime': ['datetime64[ns]'],
                    'bool': ['bool']
                }
                
                if expected_dtype in dtype_mapping:
                    if actual_dtype not in dtype_mapping[expected_dtype]:
                        sheet_compliance['dtype_mismatches'].append({
                            'column': col_name,
                            'expected': expected_dtype,
                            'actual': actual_dtype
                        })
        
        compliance_results[sheet_name] = sheet_compliance
    
    return compliance_results

def display_validation_results(validation_results: Dict[str, Any]):
    """
    Display validation results in a user-friendly format
    """
    if 'error' in validation_results:
        st.error(f"Validation Error: {validation_results['error']}")
        return
    
    st.subheader("🔍 Validation Results")
    
    # Schema compliance
    if 'schema_compliance' in validation_results:
        compliance = validation_results['schema_compliance']
        
        for sheet_name, sheet_compliance in compliance.items():
            with st.expander(f"📋 {sheet_name} Schema Compliance"):
                
                if sheet_compliance['column_count_match']:
                    st.success("✅ All expected columns present")
                else:
                    if sheet_compliance['missing_columns']:
                        st.error(f"❌ Missing columns: {sheet_compliance['missing_columns']}")
                    if sheet_compliance['extra_columns']:
                        st.warning(f"⚠️ Extra columns: {sheet_compliance['extra_columns']}")
                
                if sheet_compliance['dtype_mismatches']:
                    st.warning("⚠️ Data type mismatches:")
                    for mismatch in sheet_compliance['dtype_mismatches']:
                        st.write(f"- {mismatch['column']}: expected {mismatch['expected']}, got {mismatch['actual']}")
                else:
                    st.success("✅ All data types match expectations")
    
    # Data quality metrics after cleaning
    st.subheader("📊 Post-Cleaning Quality Metrics")
    
    # Summary metrics
    if 'summary' in validation_results:
        summary = validation_results['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", summary.get('total_records', 0))
        with col2:
            st.metric("Missing Values", summary.get('total_missing', 0))
        with col3:
            st.metric("Duplicates", summary.get('total_duplicates', 0))
        with col4:
            completeness = (1 - summary.get('total_missing', 0) / max(summary.get('total_records', 1), 1)) * 100
            st.metric("Completeness", f"{completeness:.1f}%")
