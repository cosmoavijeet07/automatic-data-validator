import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
import traceback
import os

# Import custom modules
from .config import MODELS, SUPPORTED_FILE_TYPES
from .file_handler import FileHandler
from .schema_manager import SchemaManager
from .data_analyzer import DataAnalyzer
from .data_corrector import DataCorrector
from .llm_client import LLMClient
from .session_manager import SessionManager
from .logger import Logger
from .text_processor import TextProcessor
from .pipeline_generator import PipelineGenerator

def initialize_session():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    
    if 'schema' not in st.session_state:
        st.session_state.schema = None
    
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    if 'processing_mode' not in st.session_state:
        st.session_state.processing_mode = 'manual'
    
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "GPT 4.1"
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'

def main():
    st.set_page_config(
        page_title="AI-Powered Data Cleaning Platform",
        page_icon="🧹",
        layout="wide"
    )
    
    initialize_session()
    
    # Initialize components
    session_manager = SessionManager(st.session_state.session_id)
    logger = Logger(st.session_state.session_id)
    llm_client = LLMClient(st.session_state.selected_model)
    
    st.title("🧹 AI-Powered Data Cleaning Platform")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        st.session_state.selected_model = st.selectbox(
            "Select LLM Model",
            options=list(MODELS.keys()),
            index=0
        )
        
        # Processing mode
        st.session_state.processing_mode = st.radio(
            "Processing Mode",
            ["Manual Review", "Automatic"],
            index=0 if st.session_state.processing_mode == 'manual' else 1
        )
        
        # Session info
        st.subheader("Session Info")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        
        if st.session_state.data is not None:
            st.text(f"Data Shape: {st.session_state.data.shape}")
        
        # Progress indicator
        steps = ['Upload', 'Schema', 'Analysis', 'Correction', 'Finalization']
        current_step_idx = steps.index(st.session_state.current_step.title()) if st.session_state.current_step.title() in steps else 0
        
        st.subheader("Progress")
        for i, step in enumerate(steps):
            if i < current_step_idx:
                st.success(f"✅ {step}")
            elif i == current_step_idx:
                st.info(f"🔄 {step}")
            else:
                st.text(f"⏳ {step}")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Data Processing", "Logs", "Downloads"])
    
    with tab1:
        if st.session_state.current_step == 'upload':
            handle_file_upload(logger, llm_client)
        elif st.session_state.current_step == 'schema':
            handle_schema_management(logger, llm_client)
        elif st.session_state.current_step == 'analysis':
            handle_data_analysis(logger, llm_client)
        elif st.session_state.current_step == 'correction':
            handle_data_correction(logger, llm_client)
        elif st.session_state.current_step == 'finalization':
            handle_finalization(logger, llm_client)
    
    with tab2:
        display_logs(logger)
    
    with tab3:
        handle_downloads(logger)

def handle_file_upload(logger, llm_client):
    """Handle file upload and initial processing"""
    st.header("📁 File Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_FILE_TYPES,
        help="Supported formats: CSV, Excel (multi-sheet), JSON, TXT"
    )
    
    if uploaded_file is not None:
        try:
            file_handler = FileHandler()
            
            with st.spinner("Processing file..."):
                # Process the uploaded file
                data, file_info = file_handler.process_file(uploaded_file)
                
                st.session_state.data = data
                st.session_state.original_data = data.copy()
                st.session_state.file_info = file_info
                
                logger.log("File uploaded successfully", {
                    "filename": uploaded_file.name,
                    "file_type": file_info['type'],
                    "shape": data.shape if hasattr(data, 'shape') else None
                })
            
            # Display preview
            st.success("File uploaded successfully!")
            st.subheader("Data Preview")
            
            if isinstance(data, pd.DataFrame):
                st.dataframe(data.head(10), use_container_width=True)
                st.text(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
            else:
                st.text_area("Text Data Preview", str(data)[:1000] + "..." if len(str(data)) > 1000 else str(data), height=300)
            
            if st.button("Proceed to Schema Detection", type="primary"):
                st.session_state.current_step = 'schema'
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.log("File processing error", {"error": str(e), "traceback": traceback.format_exc()})

def handle_schema_management(logger, llm_client):
    """Handle schema detection and editing"""
    st.header("🔧 Schema Management")
    
    if st.session_state.data is None:
        st.warning("No data available. Please upload a file first.")
        return
    
    schema_manager = SchemaManager(llm_client)
    
    # Detect schema
    if st.session_state.schema is None:
        with st.spinner("Detecting schema..."):
            st.session_state.schema = schema_manager.detect_schema(st.session_state.data)
            logger.log("Schema detected", st.session_state.schema)
    
    # Display current schema
    st.subheader("Current Schema")
    schema_df = pd.DataFrame(st.session_state.schema).T
    st.dataframe(schema_df, use_container_width=True)
    
    # Schema editing interface
    st.subheader("Schema Editing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text("Manual Schema Editing")
        
        # Create editable schema table
        edited_schema = {}
        for col_name, col_info in st.session_state.schema.items():
            st.text(f"Column: {col_name}")
            
            new_dtype = st.selectbox(
                f"Data Type for {col_name}",
                ['object', 'int64', 'float64', 'datetime64[ns]', 'bool', 'category'],
                index=0 if col_info['dtype'] == 'object' else 
                      1 if 'int' in col_info['dtype'] else
                      2 if 'float' in col_info['dtype'] else 0,
                key=f"dtype_{col_name}"
            )
            
            edited_schema[col_name] = {
                'dtype': new_dtype,
                'null_count': col_info['null_count'],
                'unique_count': col_info['unique_count']
            }
    
    with col2:
        st.text("Natural Language Schema Editing")
        
        nl_instruction = st.text_area(
            "Describe schema changes in natural language",
            placeholder="e.g., 'Convert date column to datetime, make age column integer, treat category as categorical'"
        )
        
        if st.button("Apply NL Changes") and nl_instruction:
            with st.spinner("Processing natural language instruction..."):
                try:
                    updated_schema = schema_manager.process_nl_schema_change(
                        st.session_state.schema, 
                        nl_instruction
                    )
                    st.session_state.schema = updated_schema
                    st.success("Schema updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing NL instruction: {str(e)}")
    
    # Apply schema changes
    if st.button("Apply Schema Changes", type="primary"):
        with st.spinner("Applying schema changes..."):
            try:
                st.session_state.data = schema_manager.apply_schema_changes(
                    st.session_state.data, 
                    st.session_state.schema
                )
                logger.log("Schema changes applied", st.session_state.schema)
                st.success("Schema changes applied successfully!")
                
                if st.button("Proceed to Data Analysis"):
                    st.session_state.current_step = 'analysis'
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error applying schema changes: {str(e)}")
                logger.log("Schema application error", {"error": str(e)})

def handle_data_analysis(logger, llm_client):
    """Handle data quality analysis"""
    st.header("🔍 Data Quality Analysis")
    
    if st.session_state.data is None:
        st.warning("No data available.")
        return
    
    data_analyzer = DataAnalyzer(llm_client)
    
    # Basic profiling
    with st.spinner("Analyzing data quality..."):
        try:
            # Generate basic quality report
            quality_report = data_analyzer.generate_quality_report(st.session_state.data)
            
            # Enhanced LLM analysis
            enhanced_analysis = data_analyzer.enhanced_llm_analysis(
                st.session_state.data, 
                st.session_state.schema,
                quality_report
            )
            
            logger.log("Data analysis completed", {
                "quality_report": quality_report,
                "enhanced_analysis": enhanced_analysis
            })
            
        except Exception as e:
            st.error(f"Error during data analysis: {str(e)}")
            logger.log("Data analysis error", {"error": str(e)})
            return
    
    # Display results
    st.subheader("Quality Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text("Basic Quality Metrics")
        for key, value in quality_report.items():
            if isinstance(value, dict):
                st.json(value)
            else:
                st.metric(key.replace('_', ' ').title(), value)
    
    with col2:
        st.text("Enhanced LLM Analysis")
        st.text_area("Analysis Summary", enhanced_analysis.get('summary', ''), height=200)
        
        if 'recommendations' in enhanced_analysis:
            st.text("Recommendations:")
            for rec in enhanced_analysis['recommendations']:
                st.text(f"• {rec}")
    
    # Store analysis results
    st.session_state.quality_report = quality_report
    st.session_state.enhanced_analysis = enhanced_analysis
    
    if st.button("Proceed to Data Correction", type="primary"):
        st.session_state.current_step = 'correction'
        st.rerun()

def handle_data_correction(logger, llm_client):
    """Handle data correction"""
    st.header("🔧 Data Correction")
    
    if st.session_state.data is None:
        st.warning("No data available.")
        return
    
    data_corrector = DataCorrector(llm_client)
    
    # User input for additional instructions
    user_instructions = st.text_area(
        "Additional correction instructions (optional)",
        placeholder="e.g., 'Remove outliers beyond 3 standard deviations, standardize categorical labels'"
    )
    
    if st.button("Generate Correction Code"):
        with st.spinner("Generating correction strategy..."):
            try:
                correction_code, strategy_summary = data_corrector.generate_correction_code(
                    st.session_state.data,
                    st.session_state.schema,
                    st.session_state.quality_report,
                    user_instructions
                )
                
                st.session_state.correction_code = correction_code
                st.session_state.strategy_summary = strategy_summary
                
                logger.log("Correction code generated", {
                    "strategy_summary": strategy_summary
                })
                
            except Exception as e:
                st.error(f"Error generating correction code: {str(e)}")
                return
    
    # Display generated code and strategy
    if hasattr(st.session_state, 'correction_code'):
        st.subheader("Correction Strategy")
        st.text_area("Strategy Summary", st.session_state.strategy_summary, height=100)
        
        st.subheader("Generated Correction Code")
        st.code(st.session_state.correction_code, language='python')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Execute Correction", type="primary"):
                execute_correction(data_corrector, logger)
        
        with col2:
            if st.button("Modify Code"):
                modified_code = st.text_area(
                    "Modify the correction code",
                    value=st.session_state.correction_code,
                    height=300
                )
                
                if st.button("Apply Modified Code"):
                    st.session_state.correction_code = modified_code
                    execute_correction(data_corrector, logger)

def execute_correction(data_corrector, logger):
    """Execute the correction code"""
    with st.spinner("Executing correction..."):
        try:
            corrected_data, execution_log = data_corrector.execute_correction_code(
                st.session_state.data,
                st.session_state.correction_code
            )
            
            st.session_state.data = corrected_data
            logger.log("Data correction executed", execution_log)
            
            st.success("Data correction completed successfully!")
            
            # Show before/after comparison
            st.subheader("Correction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.text("Original Data Shape")
                st.text(f"{st.session_state.original_data.shape}")
                
            with col2:
                st.text("Corrected Data Shape")
                st.text(f"{corrected_data.shape}")
            
            if st.button("Proceed to Finalization", type="primary"):
                st.session_state.current_step = 'finalization'
                st.rerun()
                
        except Exception as e:
            st.error(f"Error executing correction: {str(e)}")
            logger.log("Correction execution error", {"error": str(e), "traceback": traceback.format_exc()})

def handle_finalization(logger, llm_client):
    """Handle final pipeline generation"""
    st.header("🎯 Finalization")
    
    pipeline_generator = PipelineGenerator(llm_client)
    
    if st.button("Generate Final Pipeline"):
        with st.spinner("Generating pipeline..."):
            try:
                pipeline_code = pipeline_generator.generate_pipeline(
                    st.session_state.logs,
                    st.session_state.schema,
                    getattr(st.session_state, 'correction_code', '')
                )
                
                st.session_state.pipeline_code = pipeline_code
                logger.log("Pipeline generated", {"pipeline_length": len(pipeline_code)})
                
                st.success("Pipeline generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating pipeline: {str(e)}")
                return
    
    if hasattr(st.session_state, 'pipeline_code'):
        st.subheader("Generated Pipeline")
        st.code(st.session_state.pipeline_code, language='python')
        
        # Final data preview
        st.subheader("Final Cleaned Data")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        st.success("Data cleaning process completed! Check the Downloads tab for files.")

def display_logs(logger):
    """Display session logs"""
    st.header("📋 Session Logs")
    
    logs = logger.get_logs()
    
    if logs:
        for log_entry in reversed(logs):
            with st.expander(f"{log_entry['timestamp']} - {log_entry['action']}"):
                st.json(log_entry['details'])
    else:
        st.info("No logs available.")

def handle_downloads(logger):
    """Handle file downloads"""
    st.header("📥 Downloads")
    
    if st.session_state.data is not None:
        # Cleaned dataset
        csv_data = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name=f"cleaned_data_{st.session_state.session_id[:8]}.csv",
            mime="text/csv"
        )
        
        # Log file
        logs = logger.get_logs()
        log_json = json.dumps(logs, indent=2, default=str)
        st.download_button(
            label="Download Log File (JSON)",
            data=log_json,
            file_name=f"cleaning_log_{st.session_state.session_id[:8]}.json",
            mime="application/json"
        )
        
        # Pipeline script
        if hasattr(st.session_state, 'pipeline_code'):
            st.download_button(
                label="Download Pipeline Script (.py)",
                data=st.session_state.pipeline_code,
                file_name=f"data_pipeline_{st.session_state.session_id[:8]}.py",
                mime="text/plain"
            )
    else:
        st.info("No data available for download. Please process a file first.")

if __name__ == "__main__":
    main()