import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
import traceback
import os

# Import custom modules
from config import MODELS, SUPPORTED_FILE_TYPES
from file_handler import FileHandler
from schema_manager import SchemaManager
from data_analyzer import DataAnalyzer
from data_corrector import DataCorrector
from llm_client import LLMClient
from session_manager import SessionManager
from logger import Logger
from text_processor import TextProcessor
from pipeline_generator import PipelineGenerator

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
    
    # Add flags for button actions
    if 'proceed_to_analysis' not in st.session_state:
        st.session_state.proceed_to_analysis = False
    
    if 'proceed_to_correction' not in st.session_state:
        st.session_state.proceed_to_correction = False
    
    if 'proceed_to_finalization' not in st.session_state:
        st.session_state.proceed_to_finalization = False

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
        # Check for state transitions first
        if st.session_state.proceed_to_analysis:
            st.session_state.current_step = 'analysis'
            st.session_state.proceed_to_analysis = False
            st.rerun()
        
        if st.session_state.proceed_to_correction:
            st.session_state.current_step = 'correction'
            st.session_state.proceed_to_correction = False
            st.rerun()
        
        if st.session_state.proceed_to_finalization:
            st.session_state.current_step = 'finalization'
            st.session_state.proceed_to_finalization = False
            st.rerun()
        
        # Handle current step
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
    
    # Manual Schema Editing in one column
    with st.expander("📝 Manual Schema Editing", expanded=False):
        edited_schema = {}
        for col_name, col_info in st.session_state.schema.items():
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Column: {col_name}")
            with col2:
                new_dtype = st.selectbox(
                    f"Data Type",
                    ['object', 'int64', 'float64', 'datetime64[ns]', 'bool', 'category'],
                    index=['object', 'int64', 'float64', 'datetime64[ns]', 'bool', 'category'].index(
                        col_info.get('suggested_dtype', col_info['dtype'])
                    ) if col_info.get('suggested_dtype', col_info['dtype']) in ['object', 'int64', 'float64', 'datetime64[ns]', 'bool', 'category'] else 0,
                    key=f"dtype_{col_name}"
                )
                edited_schema[col_name] = {
                    'dtype': new_dtype,
                    'null_count': col_info['null_count'],
                    'unique_count': col_info['unique_count']
                }
        
        if st.button("Apply Manual Changes", key="apply_manual_schema"):
            st.session_state.schema = edited_schema
            st.success("Manual schema changes saved!")
            st.rerun()
    
    # Natural Language Schema Editing
    st.subheader("🤖 Natural Language Schema Editing")
    
    nl_instruction = st.text_area(
        "Describe schema changes in natural language",
        placeholder="e.g., 'Convert date column to datetime, make age column integer, treat category as categorical'",
        key="nl_schema_input",
        height=100
    )
    
    if st.button("Process NL Changes", key="nl_schema_button", type="primary"):
        if nl_instruction:
            with st.spinner("Processing natural language instruction..."):
                try:
                    # Generate the schema change code
                    prompt = f"""
                    Current schema: {json.dumps(st.session_state.schema, indent=2, default=str)}
                    
                    User instruction: {nl_instruction}
                    
                    Generate Python code to apply these schema changes to a DataFrame called 'df'.
                    Include comments explaining each change.
                    Return only the code, no explanations.
                    """
                    
                    generated_code = llm_client.generate_code(prompt)
                    
                    # Generate strategy summary
                    strategy = llm_client.summarize_strategy(generated_code, f"Schema changes for {len(st.session_state.schema)} columns")
                    
                    # Store in session state for review
                    st.session_state.nl_schema_code = generated_code
                    st.session_state.nl_schema_strategy = strategy
                    st.session_state.nl_instruction = nl_instruction
                    
                    # Process the schema update
                    updated_schema = schema_manager.process_nl_schema_change(
                        st.session_state.schema, 
                        nl_instruction
                    )
                    st.session_state.proposed_schema = updated_schema
                    
                except Exception as e:
                    st.error(f"Error processing NL instruction: {str(e)}")
                    logger.log("NL schema processing error", {"error": str(e)})
        else:
            st.warning("Please enter an instruction first.")
    
    # Show NL processing results if available
    if hasattr(st.session_state, 'nl_schema_code') and st.session_state.nl_schema_code:
        st.subheader("📋 Generated Schema Change Strategy")
        
        # Show strategy summary
        st.info(f"**Strategy Summary**: {st.session_state.nl_schema_strategy}")
        
        # Show generated code
        with st.expander("View Generated Code", expanded=True):
            st.code(st.session_state.nl_schema_code, language='python')
        
        # Show proposed changes
        if hasattr(st.session_state, 'proposed_schema'):
            st.subheader("⚡ Proposed Schema Changes")
            
            changes_found = False
            change_details = []
            
            for col_name in st.session_state.schema.keys():
                if col_name in st.session_state.proposed_schema:
                    old_type = st.session_state.schema[col_name].get('dtype', 'unknown')
                    new_type = st.session_state.proposed_schema[col_name].get('dtype', 'unknown')
                    
                    if old_type != new_type:
                        changes_found = True
                        change_details.append(f"**{col_name}**: `{old_type}` → `{new_type}`")
            
            if changes_found:
                for change in change_details:
                    st.write(f"📝 {change}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("✅ Approve & Apply", key="approve_nl_changes", type="primary"):
                        st.session_state.schema = st.session_state.proposed_schema
                        
                        # Apply the changes
                        with st.spinner("Applying schema changes..."):
                            try:
                                st.session_state.data = schema_manager.apply_schema_changes(
                                    st.session_state.data, 
                                    st.session_state.schema
                                )
                                st.success("Schema changes applied successfully!")
                                logger.log("NL schema changes applied", {
                                    "instruction": st.session_state.nl_instruction,
                                    "changes": st.session_state.proposed_schema
                                })
                                
                                # Clear the NL state
                                if 'nl_schema_code' in st.session_state:
                                    del st.session_state.nl_schema_code
                                if 'nl_schema_strategy' in st.session_state:
                                    del st.session_state.nl_schema_strategy
                                if 'proposed_schema' in st.session_state:
                                    del st.session_state.proposed_schema
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error applying schema changes: {str(e)}")
                
                with col2:
                    if st.button("✏️ Modify Instruction", key="modify_nl"):
                        # Clear the state to allow new instruction
                        if 'nl_schema_code' in st.session_state:
                            del st.session_state.nl_schema_code
                        if 'nl_schema_strategy' in st.session_state:
                            del st.session_state.nl_schema_strategy
                        if 'proposed_schema' in st.session_state:
                            del st.session_state.proposed_schema
                        st.rerun()
                
                with col3:
                    if st.button("❌ Reject Changes", key="reject_nl_changes"):
                        # Clear all NL-related state
                        if 'nl_schema_code' in st.session_state:
                            del st.session_state.nl_schema_code
                        if 'nl_schema_strategy' in st.session_state:
                            del st.session_state.nl_schema_strategy
                        if 'proposed_schema' in st.session_state:
                            del st.session_state.proposed_schema
                        if 'nl_instruction' in st.session_state:
                            del st.session_state.nl_instruction
                        st.info("Changes rejected. Schema unchanged.")
                        st.rerun()
            else:
                st.info("No schema changes detected from your instruction.")
    
    # Final Apply Schema button (for any pending changes)
    st.markdown("---")
    st.subheader("Apply Schema & Continue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔧 Apply Current Schema", type="primary", key="apply_final_schema"):
            with st.spinner("Applying schema..."):
                try:
                    st.session_state.data = schema_manager.apply_schema_changes(
                        st.session_state.data, 
                        st.session_state.schema
                    )
                    logger.log("Schema applied", st.session_state.schema)
                    st.success("Schema applied successfully!")
                    
                except Exception as e:
                    st.error(f"Error applying schema: {str(e)}")
                    logger.log("Schema application error", {"error": str(e)})
    
    with col2:
        if st.button("🔍 Proceed to Data Analysis", type="primary", key="schema_to_analysis_btn"):
            st.session_state.proceed_to_analysis = True
            st.rerun()

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
            
            # Enhanced LLM analysis (if available)
            if llm_client and llm_client.client:
                try:
                    enhanced_analysis = data_analyzer.enhanced_llm_analysis(
                        st.session_state.data, 
                        st.session_state.schema,
                        quality_report
                    )
                except Exception as e:
                    st.warning(f"LLM analysis failed: {str(e)}. Using basic analysis.")
                    enhanced_analysis = data_analyzer._fallback_analysis(quality_report)
            else:
                st.info("LLM not available. Using basic analysis.")
                enhanced_analysis = data_analyzer._fallback_analysis(quality_report)
            
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
        basic_info = quality_report.get('basic_info', {})
        st.metric("Data Quality Score", f"{quality_report.get('quality_score', 0):.1f}/100")
        st.metric("Total Rows", basic_info.get('row_count', 0))
        st.metric("Total Columns", basic_info.get('column_count', 0))
        st.metric("Missing Values", f"{quality_report['missing_values']['missing_percentage']:.1f}%")
        st.metric("Duplicates", f"{quality_report['duplicates']['duplicate_percentage']:.1f}%")
    
    with col2:
        st.text("Enhanced Analysis")
        st.text_area("Analysis Summary", enhanced_analysis.get('summary', ''), height=200)
        
        if 'recommendations' in enhanced_analysis:
            st.text("Recommendations:")
            for rec in enhanced_analysis['recommendations']:
                st.text(f"• {rec}")
    
    # Store analysis results
    st.session_state.quality_report = quality_report
    st.session_state.enhanced_analysis = enhanced_analysis
    
    if st.button("🔧 Proceed to Data Correction", type="primary", key="analysis_to_correction_btn"):
        st.session_state.proceed_to_correction = True
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
        placeholder="e.g., 'Remove outliers beyond 3 standard deviations, standardize categorical labels'",
        key="correction_instructions"
    )
    
    if st.button("Generate Correction Code", key="generate_correction_button"):
        if not llm_client or not llm_client.client:
            st.error("LLM client not available. Please check your OpenAI API key.")
            return
            
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
                
                st.success("Correction code generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating correction code: {str(e)}")
                logger.log("Correction code generation error", {"error": str(e)})
                return
    
    # Display generated code and strategy
    if hasattr(st.session_state, 'correction_code') and st.session_state.correction_code:
        st.subheader("📋 Correction Strategy")
        st.info(st.session_state.strategy_summary)
        
        st.subheader("🔧 Generated Correction Code")
        st.code(st.session_state.correction_code, language='python')
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Execute Correction", type="primary", key="execute_correction_button"):
                execute_correction(data_corrector, logger)
        
        with col2:
            if st.button("✏️ Modify Code", key="modify_code_button"):
                st.session_state.show_code_editor = True
                st.rerun()
        
        with col3:
            if st.button("🔄 Regenerate Code", key="regenerate_code_button"):
                # Clear existing code to trigger regeneration
                if 'correction_code' in st.session_state:
                    del st.session_state.correction_code
                if 'strategy_summary' in st.session_state:
                    del st.session_state.strategy_summary
                st.rerun()
        
        # Code editor (if user wants to modify)
        if hasattr(st.session_state, 'show_code_editor') and st.session_state.show_code_editor:
            st.subheader("✏️ Modify Correction Code")
            
            modified_code = st.text_area(
                "Edit the correction code:",
                value=st.session_state.correction_code,
                height=400,
                key="code_editor"
            )
            
            col_save, col_cancel = st.columns(2)
            
            with col_save:
                if st.button("💾 Save Changes", key="save_modified_code"):
                    st.session_state.correction_code = modified_code
                    st.session_state.show_code_editor = False
                    st.success("Code updated successfully!")
                    st.rerun()
            
            with col_cancel:
                if st.button("❌ Cancel", key="cancel_code_edit"):
                    st.session_state.show_code_editor = False
                    st.rerun()
    
    else:
        st.info("👆 Click 'Generate Correction Code' to create a data cleaning strategy.")

def execute_correction(data_corrector, logger):
    """Execute the correction code"""
    with st.spinner("Executing correction..."):
        try:
            corrected_data, execution_log = data_corrector.execute_correction_code(
                st.session_state.data,
                st.session_state.correction_code
            )
            
            if execution_log['success']:
                st.session_state.data = corrected_data
                logger.log("Data correction executed", execution_log)
                
                st.success("🎉 Data correction completed successfully!")
                
                # Show before/after comparison
                st.subheader("📊 Correction Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Original Rows", execution_log['original_shape'][0])
                
                with col2:
                    st.metric("Final Rows", execution_log['final_shape'][0])
                
                with col3:
                    rows_removed = execution_log.get('rows_removed', 0)
                    st.metric("Rows Removed", rows_removed)
                
                # Show execution output
                if execution_log.get('output'):
                    with st.expander("📋 Execution Log"):
                        for line in execution_log['output']:
                            if line.strip():
                                st.text(line)
                
                # Proceed button
                st.markdown("---")
                if st.button("🎯 Proceed to Finalization", type="primary", key="correction_to_finalization_btn"):
                    st.session_state.proceed_to_finalization = True
                    st.rerun()
            else:
                st.error("❌ Data correction failed!")
                
                # Show errors
                if execution_log.get('errors'):
                    st.subheader("🚨 Errors")
                    for error in execution_log['errors']:
                        st.error(f"**{error['type']}**: {error['message']}")
                        
                        # Show traceback in expander
                        if 'traceback' in error:
                            with st.expander("Show detailed error"):
                                st.code(error['traceback'])
                
                st.info("💡 Try modifying the code or regenerating with different instructions.")
                    
        except Exception as e:
            st.error(f"❌ Error executing correction: {str(e)}")
            logger.log("Correction execution error", {
                "error": str(e), 
                "traceback": traceback.format_exc()
            })

def handle_finalization(logger, llm_client):
    """Handle final pipeline generation"""
    st.header("🎯 Finalization")
    
    pipeline_generator = PipelineGenerator(llm_client)
    
    if st.button("Generate Final Pipeline", type="primary"):
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
        
        # Log file - Use the logger's export function which handles serialization
        log_json = logger.export_logs(format="json")
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