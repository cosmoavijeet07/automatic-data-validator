"""
Main Streamlit Application for Data Cleaning System
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from config import (
    MODELS, PROCESSING_MODES, DATA_TYPES, DATE_FORMATS,
    MAX_FILE_SIZE, OUTPUT_DIR, LOGS_DIR
)
from utils import (
    format_timestamp, create_log_entry, create_pipeline_code,
    format_code_for_display
)
from session_manager import SessionManager
from data_processor import DataProcessor
from text_processor import TextProcessor
from llm_client import LLMClient

# Page configuration
st.set_page_config(
    page_title="Intelligent Data Cleaning System",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .stAlert {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .code-block {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()

if 'session_id' not in st.session_state:
    st.session_state.session_id = st.session_state.session_manager.create_session()

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor(st.session_state.session_id)

if 'text_processor' not in st.session_state:
    st.session_state.text_processor = TextProcessor(st.session_state.session_id)

if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None

if 'current_step' not in st.session_state:
    st.session_state.current_step = 'upload'

if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = 'manual'

# Helper functions
def update_session_state(key: str, value: any):
    """Update session state and persist"""
    st.session_state[key] = value
    st.session_state.session_manager.update_session_state(
        st.session_state.session_id,
        {key: value}
    )

def add_log(step: str, action: str, details: dict, status: str = "success"):
    """Add log entry"""
    log_entry = create_log_entry(step, action, details, status)
    st.session_state.session_manager.add_log(st.session_state.session_id, log_entry)
    return log_entry

def show_code_with_review(code: str, description: str):
    """Show code with review options"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.code(code, language='python')
    
    with col2:
        st.write(f"**{description}**")
        
        if st.session_state.processing_mode == 'manual':
            if st.button("✅ Execute", key=f"exec_{time.time()}"):
                return 'execute'
            if st.button("✏️ Modify", key=f"mod_{time.time()}"):
                return 'modify'
            if st.button("❌ Reject", key=f"rej_{time.time()}"):
                return 'reject'
        else:
            return 'execute'
    
    return None

# Main UI
st.title("🧹 Intelligent Data Cleaning System")
st.markdown("### AI-Powered Data Quality Enhancement Pipeline")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Session info
    st.info(f"Session ID: {st.session_state.session_id[:8]}...")
    
    # API Key
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Enter your OpenAI API key for LLM features")
    
    if api_key:
        model_choice = st.selectbox("Select Model", list(MODELS.keys()))
        selected_model = MODELS[model_choice]
        
        if st.button("Initialize LLM"):
            st.session_state.llm_client = LLMClient(api_key, selected_model)
            st.success("LLM client initialized!")
    
    st.divider()
    
    # Processing mode
    mode_choice = st.selectbox("Processing Mode", list(PROCESSING_MODES.keys()))
    st.session_state.processing_mode = PROCESSING_MODES[mode_choice]
    
    st.divider()
    
    # Progress tracker
    st.header("Progress")
    progress_steps = {
        'upload': 'File Upload',
        'schema': 'Schema Detection',
        'quality': 'Quality Analysis',
        'correction': 'Data Correction',
        'finalize': 'Finalization'
    }
    
    for step, label in progress_steps.items():
        if st.session_state.current_step == step:
            st.success(f"▶️ {label}")
        elif list(progress_steps.keys()).index(step) < list(progress_steps.keys()).index(st.session_state.current_step):
            st.success(f"✅ {label}")
        else:
            st.text(f"⏳ {label}")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 File Upload",
    "📊 Schema Management", 
    "🔍 Quality Analysis",
    "🔧 Data Correction",
    "📝 Pipeline Generation",
    "📈 Reports & Export"
])

# Tab 1: File Upload
with tab1:
    st.header("File Upload & Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'txt'],
            help=f"Maximum file size: {MAX_FILE_SIZE}MB"
        )
        
        if uploaded_file:
            # Save uploaded file
            file_path = Path(f"./temp/{uploaded_file.name}")
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Detect file type
            file_type = file_path.suffix[1:].lower()
            if file_type == 'xls':
                file_type = 'xlsx'
            
            # Load data
            success, message = st.session_state.data_processor.load_data(str(file_path), file_type)
            
            if success:
                st.success(message)
                update_session_state('current_step', 'schema')
                add_log('upload', 'file_uploaded', {'file': uploaded_file.name, 'type': file_type})
                
                # Show preview
                st.subheader("Data Preview")
                preview_df = st.session_state.data_processor.get_preview()
                st.dataframe(preview_df, use_container_width=True)
                
                # Show basic stats
                with col2:
                    st.metric("Rows", st.session_state.data_processor.df.shape[0])
                    st.metric("Columns", st.session_state.data_processor.df.shape[1])
                    st.metric("File Type", file_type.upper())
                    st.metric("Memory Usage", 
                            f"{st.session_state.data_processor.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            else:
                st.error(f"Error loading file: {message}")

# Tab 2: Schema Management
with tab2:
    st.header("Schema Detection & Editing")
    
    if st.session_state.data_processor.df is not None:
        schema_info = st.session_state.data_processor.get_schema_info()
        
        # Display current schema
        st.subheader("Detected Schema")
        
        # Create editable schema table
        schema_df = pd.DataFrame.from_dict(schema_info, orient='index')
        
        # Schema editing interface
        st.subheader("Edit Schema")
        
        column_updates = {}
        
        for col in st.session_state.data_processor.df.columns:
            with st.expander(f"Column: {col}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_dtype = schema_info[col]['dtype']
                    new_dtype = st.selectbox(
                        "Data Type",
                        DATA_TYPES,
                        index=DATA_TYPES.index(current_dtype) if current_dtype in DATA_TYPES else 0,
                        key=f"dtype_{col}"
                    )
                    
                with col2:
                    if new_dtype == 'datetime64':
                        date_format = st.selectbox(
                            "Date Format",
                            DATE_FORMATS,
                            key=f"date_{col}"
                        )
                    else:
                        date_format = None
                
                with col3:
                    st.write("**Statistics:**")
                    st.text(f"Nulls: {schema_info[col]['null_count']} ({schema_info[col]['null_percentage']:.1f}%)")
                    st.text(f"Unique: {schema_info[col]['unique_count']}")
                
                if new_dtype != current_dtype:
                    column_updates[col] = {'dtype': new_dtype}
                    if date_format:
                        column_updates[col]['date_format'] = date_format
        
        # LLM-based schema cleaning
        if st.session_state.llm_client:
            st.subheader("AI-Powered Schema Optimization")
            
            user_instructions = st.text_area(
                "Describe any specific schema requirements or issues:",
                placeholder="e.g., 'Convert date columns to datetime, ensure numeric columns are properly typed'"
            )
            
            if st.button("Generate Schema Cleaning Code"):
                with st.spinner("Generating code..."):
                    result = st.session_state.llm_client.generate_schema_cleaning_code(
                        st.session_state.data_processor.schema,
                        user_instructions
                    )
                    
                    # Show code and summary
                    st.code(result['code'], language='python')
                    
                    summary = st.session_state.llm_client.generate_summary(result['code'])
                    st.info(f"**Summary:** {summary}")
                    
                    # Execute if approved
                    action = show_code_with_review(result['code'], "Schema Cleaning")
                    
                    if action == 'execute':
                        success, output, _ = st.session_state.data_processor.execute_code(
                            result['code'],
                            "Schema cleaning"
                        )
                        
                        if success:
                            st.success("Schema cleaning executed successfully!")
                            update_session_state('current_step', 'quality')
                        else:
                            st.error(f"Execution error: {output}")
        
        # Apply manual updates
        if column_updates and st.button("Apply Schema Updates"):
            success, message = st.session_state.data_processor.update_schema(column_updates)
            if success:
                st.success(message)
                add_log('schema', 'schema_updated', column_updates)
            else:
                st.error(message)
    else:
        st.warning("Please upload a file first")

# Tab 3: Quality Analysis
with tab3:
    st.header("Data Quality Analysis")
    
    if st.session_state.data_processor.df is not None:
        # Generate quality report
        if st.button("Run Quality Analysis"):
            with st.spinner("Analyzing data quality..."):
                quality_report = st.session_state.data_processor.generate_quality_report()
                
                # Display metrics
                st.subheader("Quality Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rows", quality_report['basic_metrics']['total_rows'])
                with col2:
                    st.metric("Total Columns", quality_report['basic_metrics']['total_columns'])
                with col3:
                    st.metric("Duplicate Rows", quality_report['basic_metrics']['duplicate_rows'])
                with col4:
                    st.metric("Duplicate %", f"{quality_report['basic_metrics']['duplicate_percentage']:.2f}%")
                
                # Missing values heatmap
                st.subheader("Missing Values Analysis")
                
                missing_df = pd.DataFrame([
                    {"Column": col, "Missing Count": count, 
                     "Missing %": (count/len(st.session_state.data_processor.df))*100}
                    for col, count in quality_report['basic_metrics']['missing_values'].items()
                ])
                
                if not missing_df.empty:
                    fig = px.bar(missing_df, x='Column', y='Missing %', 
                                title='Missing Values by Column',
                                color='Missing %',
                                color_continuous_scale='Reds')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Quality issues
                if quality_report['quality_issues']:
                    st.subheader("Quality Issues Detected")
                    for issue in quality_report['quality_issues']:
                        st.warning(f"**{issue['column']}**: {', '.join(issue['issues'])}")
                
                # LLM deeper analysis
                if st.session_state.llm_client:
                    if st.button("Run AI Deep Analysis"):
                        with st.spinner("Performing deep analysis..."):
                            result = st.session_state.llm_client.generate_quality_analysis_code(
                                st.session_state.data_processor.schema,
                                quality_report['basic_metrics']
                            )
                            
                            st.code(result['code'], language='python')
                            
                            # Execute analysis
                            success, output, results = st.session_state.data_processor.execute_code(
                                result['code'],
                                "Deep quality analysis"
                            )
                            
                            if success and results and 'quality_issues' in results:
                                st.subheader("AI-Detected Issues")
                                st.json(results['quality_issues'])
                
                update_session_state('current_step', 'correction')
    else:
        st.warning("Please upload a file first")

# Tab 4: Data Correction
with tab4:
    st.header("Data Correction & Cleaning")
    
    if st.session_state.data_processor.df is not None:
        st.subheader("Correction Options")
        
        # Missing value treatment
        with st.expander("Missing Value Treatment"):
            st.write("Select treatment strategy for each column:")
            
            missing_strategy = {}
            for col in st.session_state.data_processor.df.columns:
                if st.session_state.data_processor.df[col].isnull().any():
                    strategy = st.selectbox(
                        f"{col}",
                        ['keep', 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'interpolate', 'custom'],
                        key=f"missing_{col}"
                    )
                    
                    if strategy == 'custom':
                        custom_value = st.text_input(f"Custom value for {col}", key=f"custom_{col}")
                        missing_strategy[col] = custom_value
                    elif strategy != 'keep':
                        missing_strategy[col] = strategy
            
            if missing_strategy and st.button("Apply Missing Value Treatment"):
                success, message = st.session_state.data_processor.apply_missing_value_treatment(missing_strategy)
                if success:
                    st.success(message)
                    add_log('correction', 'missing_values_treated', missing_strategy)
                else:
                    st.error(message)
        
        # Duplicate removal
        with st.expander("Duplicate Removal"):
            if st.button("Remove Duplicate Rows"):
                success, message = st.session_state.data_processor.remove_duplicates()
                if success:
                    st.success(message)
                    add_log('correction', 'duplicates_removed', {'message': message})
                else:
                    st.error(message)
        
        # Outlier handling
        with st.expander("Outlier Treatment"):
            numeric_cols = st.session_state.data_processor.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                col_to_treat = st.selectbox("Select column", numeric_cols)
                method = st.selectbox("Method", ['iqr', 'zscore'])
                threshold = st.slider("Threshold", 0.5, 3.0, 1.5, 0.1)
                
                if st.button("Handle Outliers"):
                    success, message = st.session_state.data_processor.handle_outliers(
                        col_to_treat, method, threshold
                    )
                    if success:
                        st.success(message)
                        add_log('correction', 'outliers_handled', 
                               {'column': col_to_treat, 'method': method, 'threshold': threshold})
                    else:
                        st.error(message)
        
        # LLM-based correction
        if st.session_state.llm_client:
            st.subheader("AI-Powered Data Correction")
            
            correction_instructions = st.text_area(
                "Describe correction requirements:",
                placeholder="e.g., 'Fix inconsistent date formats, standardize text fields, handle outliers'"
            )
            
            if st.button("Generate Correction Code"):
                with st.spinner("Generating correction code..."):
                    # Get current quality report
                    quality_report = st.session_state.data_processor.generate_quality_report()
                    
                    result = st.session_state.llm_client.generate_correction_code(
                        st.session_state.data_processor.schema,
                        quality_report['quality_issues'],
                        correction_instructions
                    )
                    
                    st.code(result['code'], language='python')
                    
                    summary = st.session_state.llm_client.generate_summary(result['code'])
                    st.info(f"**Summary:** {summary}")
                    
                    action = show_code_with_review(result['code'], "Data Correction")
                    
                    if action == 'execute':
                        success, output, _ = st.session_state.data_processor.execute_code(
                            result['code'],
                            "Data correction"
                        )
                        
                        if success:
                            st.success("Correction executed successfully!")
                            update_session_state('current_step', 'finalize')
                        else:
                            st.error(f"Execution error: {output}")
                            
                            # Try to fix error
                            if st.button("Auto-fix Error"):
                                user_feedback = st.text_input("Additional guidance (optional)")
                                
                                fixed_result = st.session_state.llm_client.fix_error(
                                    result['code'],
                                    output,
                                    user_feedback
                                )
                                
                                st.code(fixed_result['code'], language='python')
                                
                                # Retry execution
                                success, output, _ = st.session_state.data_processor.execute_code(
                                    fixed_result['code'],
                                    "Fixed correction"
                                )
                                
                                if success:
                                    st.success("Fixed and executed successfully!")
                                else:
                                    st.error(f"Still has errors: {output}")
    else:
        st.warning("Please upload a file first")

# Tab 5: Pipeline Generation
with tab5:
    st.header("Pipeline Generation")
    
    if st.session_state.data_processor.code_history:
        st.subheader("Code History")
        
        # Display code history
        for i, code in enumerate(st.session_state.data_processor.code_history, 1):
            with st.expander(f"Step {i}"):
                st.code(code, language='python')
        
        # Generate pipeline
        if st.button("Generate Complete Pipeline"):
            if st.session_state.llm_client:
                with st.spinner("Generating pipeline..."):
                    pipeline_code = st.session_state.llm_client.generate_pipeline_code(
                        st.session_state.data_processor.code_history
                    )
            else:
                pipeline_code = create_pipeline_code(st.session_state.data_processor.code_history)
            
            st.subheader("Generated Pipeline Script")
            st.code(pipeline_code, language='python')
            
            # Save pipeline
            pipeline_path = OUTPUT_DIR / f"pipeline_{st.session_state.session_id[:8]}_{format_timestamp()}.py"
            with open(pipeline_path, 'w') as f:
                f.write(pipeline_code)
            
            st.success(f"Pipeline saved to: {pipeline_path}")
            
            # Download button
            st.download_button(
                label="Download Pipeline Script",
                data=pipeline_code,
                file_name=f"data_cleaning_pipeline_{format_timestamp()}.py",
                mime="text/x-python"
            )
    else:
        st.warning("No cleaning operations performed yet")

# Tab 6: Reports & Export
with tab6:
    st.header("Reports & Export")
    
    if st.session_state.data_processor.df is not None:
        # Comparison report
        st.subheader("Before/After Comparison")
        
        comparison = st.session_state.data_processor.get_comparison_report()
        
        if comparison:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Original Shape", f"{comparison['original_shape'][0]} × {comparison['original_shape'][1]}")
                st.metric("Original Memory", f"{comparison['memory_usage_original']:.2f} MB")
            
            with col2:
                st.metric("Current Shape", f"{comparison['current_shape'][0]} × {comparison['current_shape'][1]}")
                st.metric("Current Memory", f"{comparison['memory_usage_current']:.2f} MB")
            
            st.metric("Rows Removed", comparison['rows_removed'])
        
        # Export options
        st.subheader("Export Cleaned Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox("Export Format", ['csv', 'xlsx', 'json'])
        
        with col2:
            if st.button("Export Data"):
                filename = f"cleaned_data_{format_timestamp()}.{export_format}"
                filepath = OUTPUT_DIR / filename
                
                success, message = st.session_state.data_processor.export_data(str(filepath), export_format)
                
                if success:
                    st.success(message)
                    
                    # Download button
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label=f"Download {export_format.upper()}",
                            data=f.read(),
                            file_name=filename,
                            mime='application/octet-stream'
                        )
                else:
                    st.error(message)
        
        # Export logs
        st.subheader("Export Logs")
        
        if st.button("Export Session Logs"):
            log_filename = f"session_logs_{st.session_state.session_id[:8]}_{format_timestamp()}.json"
            log_filepath = LOGS_DIR / log_filename
            
            if st.session_state.session_manager.export_session_logs(
                st.session_state.session_id,
                str(log_filepath)
            ):
                st.success(f"Logs exported to: {log_filepath}")
                
                with open(log_filepath, 'rb') as f:
                    st.download_button(
                        label="Download Logs",
                        data=f.read(),
                        file_name=log_filename,
                        mime='application/json'
                    )
    else:
        st.warning("Please upload a file first")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Intelligent Data Cleaning System v1.0 | Powered by LLMs</p>
    <p>Session: {session_id} | {timestamp}</p>
</div>
""".format(
    session_id=st.session_state.session_id[:8],
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
), unsafe_allow_html=True)