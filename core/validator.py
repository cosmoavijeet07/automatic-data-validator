import pandas as pd
import numpy as np
import json
import streamlit as st
from typing import Dict, Any, Tuple, Optional
from .data_profiler import execute_analysis
from .logger_service import log

def validate_cleaned_data(cleaned_data: Dict[str, pd.DataFrame], analysis_code: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate cleaned data against original analysis with safe handling
    """
    # ✅ SAFE CHECK: Handle None cleaned_data
    if not cleaned_data:
        return {
            'error': 'No cleaned data provided for validation',
            'status': 'failed',
            'details': 'Cleaned data is None or empty'
        }
    
    try:
        # Re-run analysis on cleaned data
        validation_results = execute_analysis(analysis_code, cleaned_data)
        
        # Compare with expected schema
        schema_validation = validate_schema_compliance(cleaned_data, schema)
        
        # Combine results safely
        if not validation_results:
            validation_results = {}
        
        validation_results['schema_compliance'] = schema_validation
        validation_results['status'] = 'success'
        
        return validation_results
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'status': 'failed',
            'details': f'Validation failed: {str(e)}'
        }
        st.error(f"Validation failed: {str(e)}")
        return error_result

def validate_schema_compliance(dataframes: Dict[str, pd.DataFrame], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if cleaned data complies with expected schema
    """
    compliance_results = {}
    
    # ✅ SAFE CHECK: Handle None inputs
    if not dataframes or not schema:
        return {
            'error': 'Missing dataframes or schema for compliance check',
            'dataframes_provided': bool(dataframes),
            'schema_provided': bool(schema)
        }
    
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
                expected_dtype = col_schema.get('dtype', 'str') if isinstance(col_schema, dict) else str(col_schema)
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

def display_validation_results(validation_results: Optional[Dict[str, Any]]):
    """
    Display validation results with comprehensive null safety
    """
    # ✅ SAFE CHECK: Handle None validation_results
    if not validation_results:
        st.warning("⚠️ No validation results available to display.")
        st.info("Please ensure the validation process completed successfully.")
        return
    
    # ✅ SAFE CHECK: Ensure validation_results is a dictionary
    if not isinstance(validation_results, dict):
        st.error("❌ Invalid validation results format. Expected dictionary.")
        st.write(f"Received: {type(validation_results)}")
        return
    
    # ✅ SAFE CHECK: Handle error cases
    if validation_results.get('error'):
        st.error(f"Validation Error: {validation_results['error']}")
        if validation_results.get('details'):
            st.write(f"Details: {validation_results['details']}")
        return
    
    st.subheader("🔍 Validation Results")
    
    # Display status
    status = validation_results.get('status', 'unknown')
    if status == 'success':
        st.success("✅ Validation completed successfully")
    elif status == 'failed':
        st.error("❌ Validation failed")
    else:
        st.info(f"ℹ️ Validation status: {status}")
    
    # Schema compliance
    if validation_results.get('schema_compliance'):
        compliance = validation_results['schema_compliance']
        
        # Check if compliance has error
        if compliance.get('error'):
            st.warning(f"⚠️ Schema compliance check failed: {compliance['error']}")
            return
        
        for sheet_name, sheet_compliance in compliance.items():
            if sheet_name == 'error':  # Skip error key
                continue
                
            if not isinstance(sheet_compliance, dict):
                continue
                
            with st.expander(f"📋 {sheet_name} Schema Compliance"):
                
                if sheet_compliance.get('column_count_match', False):
                    st.success("✅ All expected columns present")
                else:
                    missing_cols = sheet_compliance.get('missing_columns', [])
                    extra_cols = sheet_compliance.get('extra_columns', [])
                    
                    if missing_cols:
                        st.error(f"❌ Missing columns: {missing_cols}")
                    if extra_cols:
                        st.warning(f"⚠️ Extra columns: {extra_cols}")
                
                dtype_mismatches = sheet_compliance.get('dtype_mismatches', [])
                if dtype_mismatches:
                    st.warning("⚠️ Data type mismatches:")
                    for mismatch in dtype_mismatches:
                        st.write(f"- {mismatch.get('column', 'unknown')}: expected {mismatch.get('expected', 'unknown')}, got {mismatch.get('actual', 'unknown')}")
                else:
                    st.success("✅ All data types match expectations")
    
    # Data quality metrics after cleaning
    st.subheader("📊 Post-Cleaning Quality Metrics")
    
    # Summary metrics
    if validation_results.get('summary'):
        summary = validation_results['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", summary.get('total_records', 0))
        with col2:
            st.metric("Missing Values", summary.get('total_missing', 0))
        with col3:
            st.metric("Duplicates", summary.get('total_duplicates', 0))
        with col4:
            total_records = summary.get('total_records', 1)
            total_missing = summary.get('total_missing', 0)
            completeness = (1 - total_missing / max(total_records, 1)) * 100
            st.metric("Completeness", f"{completeness:.1f}%")
    else:
        st.info("ℹ️ No summary metrics available")
