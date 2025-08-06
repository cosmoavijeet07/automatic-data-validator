import streamlit as st
import json
import pandas as pd
import traceback
from datetime import datetime
from pathlib import Path

from core import (
    session_manager, 
    file_handler, 
    schema_detector, 
    data_profiler, 
    data_cleaner,
    validator,
    logger_service
)
from utils.helpers import display_dataframe_info, format_error_message

# Configure Streamlit
st.set_page_config(
    page_title="🤖 Automated Data Validator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session
if 'session_initialized' not in st.session_state:
    session_manager.initialize_session()
    st.session_state.session_initialized = True

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selection
    model_options = {
        "GPT 4.1": "gpt-4.1-2025-04-14",
        "GPT O4 Mini": "o4-mini-2025-04-16", 
        "Claude Sonnet 4": "claude-sonnet-4-20250514"
    }
    
    selected_model = st.selectbox(
        "🧠 Choose AI Model",
        options=list(model_options.keys()),
        index=0,
        help="GPT 4.1 is recommended for most tasks"
    )
    
    st.session_state.current_model = model_options[selected_model]
    
    # Session Info
    st.subheader("📊 Session Info")
    st.code(f"Session ID: {st.session_state.session_id}")
    st.write(f"**Stage:** {st.session_state.current_stage}")
    
    # Live Logs
    if st.checkbox("📜 Show Live Logs", False):
        logs = logger_service.get_recent_logs(st.session_state.session_id)
        if logs:
            st.text_area("Recent Activity", value=logs, height=200)

# Main Application
st.title("🤖 Automated Data Validator")
st.markdown("*Intelligent data validation powered by Large Language Models*")

# Progress Bar
stages = ["upload", "schema_detection", "schema_editing", "data_analysis", "data_cleaning", "validation", "completion"]
current_idx = stages.index(st.session_state.current_stage) if st.session_state.current_stage in stages else 0
progress = (current_idx + 1) / len(stages)
st.progress(progress)

# Stage Router
if st.session_state.current_stage == "upload":
    st.header("1️⃣ Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=['csv', 'xlsx', 'json'],
        help="Supports CSV, Excel (multi-sheet), and JSON formats"
    )
    
    if uploaded_file:
        try:
            with st.spinner("Processing uploaded file..."):
                file_info, dataframes = file_handler.process_uploaded_file(uploaded_file)
                
                st.session_state.file_info = file_info
                st.session_state.original_data = dataframes
                st.session_state.current_data = dataframes.copy()
                
                logger_service.log(
                    st.session_state.session_id, 
                    "file_upload", 
                    f"Uploaded: {file_info['filename']} ({file_info['file_type']})"
                )
            
            st.success(f"✅ Successfully loaded {file_info['filename']}")
            display_dataframe_info(dataframes)
            
            if st.button("🔄 Proceed to Schema Detection", type="primary"):
                st.session_state.current_stage = "schema_detection"
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger_service.log(st.session_state.session_id, "error", f"File upload error: {str(e)}")

elif st.session_state.current_stage == "schema_detection":
    st.header("2️⃣ Schema Detection")
    
    if 'schema_code' not in st.session_state:
        with st.spinner("🔍 Analyzing data structure..."):
            try:
                schema_code = schema_detector.generate_schema_detection_code(
                    st.session_state.current_data,
                    st.session_state.current_model
                )
                st.session_state.schema_code = schema_code
            except Exception as e:
                st.error(f"❌ Failed to generate schema code: {str(e)}")
                st.stop()
    
    st.subheader("Generated Schema Detection Code")
    st.code(st.session_state.schema_code, language="python")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ Execute Schema Detection"):
            try:
                with st.spinner("Executing schema detection..."):
                    schema_info = schema_detector.execute_schema_code(
                        st.session_state.schema_code,
                        st.session_state.current_data
                    )
                    if schema_info is None or not schema_info:
                        st.error("❌ Schema detection returned empty results")
                        logger_service.log(st.session_state.session_id, "error", "Empty schema detection results")
                    else:
                        st.session_state.detected_schema = schema_info
                        logger_service.log(st.session_state.session_id, "schema_detection", "Schema detected successfully")
                        st.success("✅ Schema detection completed!")
            except Exception as e:
                st.error(f"❌ Execution failed: {str(e)}")
                logger_service.log(st.session_state.session_id, "error", f"Schema detection error: {str(e)}")
    
    with col2:
        if st.button("🔄 Regenerate Code"):
            if 'schema_code' in st.session_state:
                del st.session_state.schema_code
            st.rerun()
    
    if 'detected_schema' in st.session_state and st.session_state.detected_schema:
        st.subheader("Detected Schema")
        st.json(st.session_state.detected_schema)
        
        if st.button("➡️ Proceed to Schema Editing", type="primary"):
            st.session_state.current_stage = "schema_editing"
            st.rerun()
    else:
        st.info("ℹ️ Execute schema detection to proceed to the next stage.")

elif st.session_state.current_stage == "schema_editing":
    st.header("3️⃣ Schema Editing & Refinement")
    
    # Check if we have a detected schema first
    if 'detected_schema' not in st.session_state or st.session_state.detected_schema is None:
        st.error("❌ No schema detected. Please go back to Schema Detection stage.")
        if st.button("🔙 Back to Schema Detection"):
            st.session_state.current_stage = "schema_detection"
            st.rerun()
        st.stop()
    
    # Initialize edited_schema safely
    if 'edited_schema' not in st.session_state or st.session_state.edited_schema is None:
        st.session_state.edited_schema = st.session_state.detected_schema.copy()
    
    # Interactive Schema Editor
    try:
        edited_schema = schema_detector.create_schema_editor(
            st.session_state.edited_schema,
            st.session_state.current_data
        )
        st.session_state.edited_schema = edited_schema
    except Exception as e:
        st.error(f"❌ Error in schema editor: {str(e)}")
        st.write("**Debug Info:**")
        st.write(f"- detected_schema exists: {'detected_schema' in st.session_state}")
        st.write(f"- detected_schema value: {st.session_state.get('detected_schema', 'Not found')}")
        st.write(f"- edited_schema exists: {'edited_schema' in st.session_state}")
        st.write(f"- edited_schema value: {st.session_state.get('edited_schema', 'Not found')}")
        st.stop()
    
    # Natural Language Instructions
    st.subheader("✍️ Natural Language Instructions")
    nl_instructions = st.text_area(
        "Describe any specific changes you'd like to make:",
        placeholder="e.g., Convert 'date_column' to datetime with format MM/DD/YYYY, treat 'amount' as currency...",
        height=100
    )
    
    if nl_instructions and st.button("🔄 Apply NL Changes"):
        with st.spinner("Processing natural language instructions..."):
            try:
                updated_schema = schema_detector.apply_nl_instructions(
                    st.session_state.edited_schema,
                    nl_instructions,
                    st.session_state.current_model
                )
                st.session_state.edited_schema = updated_schema
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error applying NL changes: {str(e)}")
    
    if st.button("✅ Confirm Schema", type="primary"):
        st.session_state.final_schema = st.session_state.edited_schema
        st.session_state.current_stage = "data_analysis"
        logger_service.log(st.session_state.session_id, "schema_confirmed", json.dumps(st.session_state.final_schema))
        st.rerun()

elif st.session_state.current_stage == "data_analysis":
    st.header("4️⃣ Data Quality Analysis")
    
    if 'analysis_code' not in st.session_state:
        with st.spinner("🔍 Generating analysis code..."):
            analysis_code = data_profiler.generate_analysis_code(
                st.session_state.final_schema,
                st.session_state.current_model
            )
            st.session_state.analysis_code = analysis_code
    
    st.subheader("Generated Analysis Code")
    st.code(st.session_state.analysis_code, language="python")
    
    # Natural Language Analysis Instructions
    analysis_instructions = st.text_area(
        "Additional analysis requirements:",
        placeholder="e.g., Focus on outliers in sales data, check for duplicate customer IDs...",
        height=80
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("▶️ Execute Analysis"):
            try:
                with st.spinner("Analyzing data quality..."):
                    analysis_results = data_profiler.execute_analysis(
                        st.session_state.analysis_code,
                        st.session_state.current_data,
                        analysis_instructions
                    )
                    st.session_state.analysis_results = analysis_results
                    logger_service.log(st.session_state.session_id, "analysis_complete", "Data analysis completed")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger_service.log(st.session_state.session_id, "error", f"Analysis error: {str(e)}")
    
    with col2:
        if analysis_instructions and st.button("🔄 Regenerate with Instructions"):
            analysis_code = data_profiler.generate_analysis_code(
                st.session_state.final_schema,
                st.session_state.current_model,
                additional_instructions=analysis_instructions
            )
            st.session_state.analysis_code = analysis_code
            st.rerun()
    
    with col3:
        if st.button("🔄 Regenerate Code"):
            del st.session_state.analysis_code
            st.rerun()
    
    if 'analysis_results' in st.session_state:
        st.subheader("📊 Analysis Results")
        data_profiler.display_analysis_results(st.session_state.analysis_results)
        
        if st.button("🧹 Proceed to Data Cleaning", type="primary"):
            st.session_state.current_stage = "data_cleaning"
            st.rerun()

elif st.session_state.current_stage == "data_cleaning":
    st.header("5️⃣ Data Cleaning & Correction")
    
    if 'cleaning_code' not in st.session_state:
        with st.spinner("🛠️ Generating cleaning code..."):
            cleaning_code = data_cleaner.generate_cleaning_code(
                st.session_state.final_schema,
                st.session_state.analysis_results,
                st.session_state.current_model
            )
            st.session_state.cleaning_code = cleaning_code
    
    st.subheader("Generated Cleaning Code")
    st.code(st.session_state.cleaning_code, language="python")
    
    # Natural Language Cleaning Instructions
    cleaning_instructions = st.text_area(
        "Additional cleaning instructions:",
        placeholder="e.g., Remove rows where age > 100, fill missing names with 'Unknown'...",
        height=80
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("▶️ Execute Cleaning"):
            try:
                with st.spinner("Cleaning data..."):
                    cleaned_data = data_cleaner.execute_cleaning(
                        st.session_state.cleaning_code,
                        st.session_state.current_data,
                        cleaning_instructions
                    )
                    st.session_state.cleaned_data = cleaned_data
                    logger_service.log(st.session_state.session_id, "cleaning_complete", "Data cleaning completed")
            except Exception as e:
                st.error(f"Cleaning failed: {str(e)}")
                # Auto-retry with error feedback
                if 'cleaning_retries' not in st.session_state:
                    st.session_state.cleaning_retries = 0
                
                if st.session_state.cleaning_retries < 3:
                    st.session_state.cleaning_retries += 1
                    with st.spinner("Attempting to fix the code..."):
                        fixed_code = data_cleaner.fix_cleaning_code(
                            st.session_state.cleaning_code,
                            str(e),
                            st.session_state.current_model
                        )
                        st.session_state.cleaning_code = fixed_code
                        st.rerun()
                else:
                    logger_service.log(st.session_state.session_id, "error", f"Cleaning error after retries: {str(e)}")
    
    with col2:
        if cleaning_instructions and st.button("🔄 Regenerate with Instructions"):
            cleaning_code = data_cleaner.generate_cleaning_code(
                st.session_state.final_schema,
                st.session_state.analysis_results,
                st.session_state.current_model,
                additional_instructions=cleaning_instructions
            )
            st.session_state.cleaning_code = cleaning_code
            st.rerun()
    
    with col3:
        if st.button("🔄 Regenerate Code"):
            del st.session_state.cleaning_code
            if 'cleaning_retries' in st.session_state:
                del st.session_state.cleaning_retries
            st.rerun()
    
    if 'cleaned_data' in st.session_state:
        st.subheader("✨ Cleaned Data Preview")
        display_dataframe_info(st.session_state.cleaned_data, title="Cleaned Data")
        
        if st.button("🔍 Validate Cleaned Data", type="primary"):
            st.session_state.current_stage = "validation"
            st.rerun()

elif st.session_state.current_stage == "validation":
    st.header("6️⃣ Validation & Quality Check")
    
    if 'validation_results' not in st.session_state:
        with st.spinner("🔍 Validating cleaned data..."):
            validation_results = validator.validate_cleaned_data(
                st.session_state.cleaned_data,
                st.session_state.analysis_code,
                st.session_state.final_schema
            )
            st.session_state.validation_results = validation_results
    
    st.subheader("📊 Validation Results")
    validator.display_validation_results(st.session_state.validation_results)
    
    # User satisfaction check
    satisfaction = st.radio(
        "Are you satisfied with the cleaning results?",
        ["✅ Yes, data looks good", "❌ No, needs improvement"],
        index=0
    )
    
    if satisfaction == "❌ No, needs improvement":
        improvement_notes = st.text_area(
            "What needs improvement?",
            placeholder="Describe the issues you see and how you'd like them addressed..."
        )
        
        if improvement_notes and st.button("🔄 Improve Cleaning"):
            # Go back to cleaning stage with feedback
            st.session_state.current_stage = "data_cleaning"
            st.session_state.improvement_feedback = improvement_notes
            if 'cleaning_code' in st.session_state:
                del st.session_state.cleaning_code
            st.rerun()
    else:
        if st.button("✅ Finalize & Generate Pipeline", type="primary"):
            st.session_state.current_stage = "completion"
            st.rerun()

elif st.session_state.current_stage == "completion":
    st.header("🎉 Completion & Download")
    
    if 'final_pipeline' not in st.session_state:
        with st.spinner("📦 Generating final pipeline..."):
            pipeline_code = data_cleaner.create_final_pipeline(
                st.session_state.schema_code,
                st.session_state.analysis_code,
                st.session_state.cleaning_code,
                st.session_state.current_model
            )
            st.session_state.final_pipeline = pipeline_code
    
    st.success("🎊 Data validation and cleaning completed successfully!")
    
    # Display final statistics
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Original Records", len(next(iter(st.session_state.original_data.values()))))
    with col2:
        st.metric("Cleaned Records", len(next(iter(st.session_state.cleaned_data.values()))))
    with col3:
        records_diff = len(next(iter(st.session_state.cleaned_data.values()))) - len(next(iter(st.session_state.original_data.values())))
        st.metric("Records Change", records_diff, delta=records_diff)
    
    # Download Section
    st.subheader("📥 Downloads")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Cleaned Data Download
        if len(st.session_state.cleaned_data) == 1:
            df = next(iter(st.session_state.cleaned_data.values()))
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📊 Download Cleaned Data (CSV)",
                data=csv_data,
                file_name=f"cleaned_data_{st.session_state.session_id}.csv",
                mime="text/csv"
            )
        else:
            # Multiple sheets - create Excel
            excel_buffer = file_handler.create_excel_download(st.session_state.cleaned_data)
            st.download_button(
                "📊 Download Cleaned Data (Excel)",
                data=excel_buffer,
                file_name=f"cleaned_data_{st.session_state.session_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        # Pipeline Code Download
        st.download_button(
            "🐍 Download Pipeline Code",
            data=st.session_state.final_pipeline,
            file_name=f"data_cleaning_pipeline_{st.session_state.session_id}.py",
            mime="text/plain"
        )
    
    with col3:
        # Session Logs Download
        session_logs = logger_service.get_session_logs(st.session_state.session_id)
        st.download_button(
            "📜 Download Session Logs",
            data=session_logs,
            file_name=f"session_logs_{st.session_state.session_id}.txt",
            mime="text/plain"
        )
    
    # Pipeline Preview
    st.subheader("🔍 Generated Pipeline Preview")
    with st.expander("View Generated Code", expanded=False):
        st.code(st.session_state.final_pipeline, language="python")
    
    # Reset Option
    if st.button("🔄 Start New Session"):
        # Clear session state
        for key in list(st.session_state.keys()):
            if key not in ['session_initialized']:
                del st.session_state[key]
        session_manager.initialize_session()
        st.rerun()

# Error Display (if any)
if 'error_message' in st.session_state and st.session_state.error_message:
    st.error(st.session_state.error_message)
    if st.button("Clear Error"):
        st.session_state.error_message = None
        st.rerun()
