import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
import traceback
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

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
    
    # Initialize logger and session manager ONCE
    if 'logger' not in st.session_state:
        st.session_state.logger = Logger(st.session_state.session_id)
    
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager(st.session_state.session_id)
    
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = LLMClient(st.session_state.selected_model)

def reset_session_for_new_file():
    """Reset session when a new file is uploaded"""
    # Generate new session ID
    new_session_id = str(uuid.uuid4())
    
    # Keep the processing mode and model selection
    processing_mode = st.session_state.processing_mode
    selected_model = st.session_state.selected_model
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        if key not in ['processing_mode', 'selected_model']:
            del st.session_state[key]
    
    # Set new session ID and reinitialize
    st.session_state.session_id = new_session_id
    st.session_state.processing_mode = processing_mode
    st.session_state.selected_model = selected_model
    st.session_state.current_step = 'upload'
    
    # Create new logger and session manager for the new session
    st.session_state.logger = Logger(new_session_id)
    st.session_state.session_manager = SessionManager(new_session_id)
    st.session_state.llm_client = LLMClient(selected_model)

def main():
    st.set_page_config(
        page_title="AI-Powered Data Cleaning Platform",
        page_icon="🧹",
        layout="wide"
    )
    
    initialize_session()
    
    # Use persistent components from session state
    logger = st.session_state.logger
    session_manager = st.session_state.session_manager
    llm_client = st.session_state.llm_client
    
    st.title("🧹 AI-Powered Data Cleaning Platform")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        new_model = st.selectbox(
            "Select LLM Model",
            options=list(MODELS.keys()),
            index=list(MODELS.keys()).index(st.session_state.selected_model)
        )
        
        if new_model != st.session_state.selected_model:
            st.session_state.selected_model = new_model
            st.session_state.llm_client.switch_model(new_model)
        
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
        progress = st.progress(current_step_idx / (len(steps) - 1))
        
        for i, step in enumerate(steps):
            if i < current_step_idx:
                st.success(f"✅ {step}")
            elif i == current_step_idx:
                st.info(f"🔄 {step}")
            else:
                st.text(f"⏳ {step}")
        
        # Session actions
        st.subheader("Session Actions")
        if st.button("🔄 Reset Session", help="Start over with a new file"):
            reset_session_for_new_file()
            st.rerun()

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
        help="Supported formats: CSV, Excel (multi-sheet), JSON, TXT",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            # This is a new file, reset session
            reset_session_for_new_file()
            st.session_state.last_uploaded_file = uploaded_file.name
            logger = st.session_state.logger
            
        try:
            file_handler = FileHandler()
            
            with st.spinner("Processing file..."):
                # Process the uploaded file
                data, file_info = file_handler.process_file(uploaded_file)
                
                st.session_state.data = data
                st.session_state.original_data = data.copy() if isinstance(data, pd.DataFrame) else data
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
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(data.head(10), use_container_width=True)
                
                with col2:
                    st.metric("Rows", data.shape[0])
                    st.metric("Columns", data.shape[1])
                    st.metric("Size (MB)", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f}")
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
                                for key in ['nl_schema_code', 'nl_schema_strategy', 'proposed_schema']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error applying schema changes: {str(e)}")
                
                with col2:
                    if st.button("✏️ Modify Instruction", key="modify_nl"):
                        # Clear the state to allow new instruction
                        for key in ['nl_schema_code', 'nl_schema_strategy', 'proposed_schema']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                
                with col3:
                    if st.button("❌ Reject Changes", key="reject_nl_changes"):
                        # Clear all NL-related state
                        for key in ['nl_schema_code', 'nl_schema_strategy', 'proposed_schema', 'nl_instruction']:
                            if key in st.session_state:
                                del st.session_state[key]
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
    """Handle data quality analysis with visualizations"""
    st.header("🔍 Data Quality Analysis")
    
    if st.session_state.data is None:
        st.warning("No data available.")
        return
    
    data_analyzer = DataAnalyzer(llm_client)
    
    # Basic profiling
    if 'quality_report' not in st.session_state:
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
                
                # Store in session state
                st.session_state.quality_report = quality_report
                st.session_state.enhanced_analysis = enhanced_analysis
                
            except Exception as e:
                st.error(f"Error during data analysis: {str(e)}")
                logger.log("Data analysis error", {"error": str(e)})
                return
    else:
        quality_report = st.session_state.quality_report
        enhanced_analysis = st.session_state.enhanced_analysis
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Visualizations", "🔍 Complete Analysis", "🤖 AI Insights", "📋 Raw Report"])
    
    with tab1:
        display_analysis_overview(quality_report)
    
    with tab2:
        handle_visualization_generation(logger, llm_client, quality_report)
    
    with tab3:
        display_complete_analysis(quality_report, enhanced_analysis)
    
    with tab4:
        display_ai_insights(enhanced_analysis)
    
    with tab5:
        display_raw_report(quality_report, enhanced_analysis)
    
    # Proceed button
    st.markdown("---")
    if st.button("🔧 Proceed to Data Correction", type="primary", key="analysis_to_correction_btn"):
        st.session_state.proceed_to_correction = True
        st.rerun()

def display_analysis_overview(quality_report):
    """Display overview metrics from the analysis"""
    st.subheader("Quality Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Quality Score", f"{quality_report.get('quality_score', 0):.1f}/100")
    
    with col2:
        basic_info = quality_report.get('basic_info', {})
        st.metric("Total Rows", f"{basic_info.get('row_count', 0):,}")
    
    with col3:
        st.metric("Total Columns", basic_info.get('column_count', 0))
    
    with col4:
        st.metric("Memory Usage", f"{basic_info.get('memory_usage_mb', 0):.2f} MB")
    
    # Issues summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Missing Values", f"{quality_report['missing_values']['missing_percentage']:.1f}%")
    
    with col2:
        st.metric("Duplicates", f"{quality_report['duplicates']['duplicate_percentage']:.1f}%")
    
    with col3:
        outlier_info = quality_report.get('outliers', {})
        st.metric("Outliers", f"{outlier_info.get('outlier_percentage', 0):.1f}%")

def handle_visualization_generation(logger, llm_client, quality_report):
    """Handle visualization generation with edit capability"""
    st.subheader("Data Visualizations")
    
    # Generate visualizations using LLM
    if st.button("Generate EDA Visualizations", key="generate_viz"):
        with st.spinner("Generating visualization code..."):
            try:
                # Create comprehensive prompt with all analysis data
                viz_prompt = f"""
                Generate Python code for comprehensive Exploratory Data Analysis (EDA) visualizations.
                
                DATASET INFORMATION:
                - Columns: {list(st.session_state.data.columns)}
                - Data types: {json.dumps({col: str(dtype) for col, dtype in st.session_state.data.dtypes.to_dict().items()}, indent=2)}
                - Shape: {st.session_state.data.shape}
                
                CURRENT SCHEMA:
                {json.dumps(st.session_state.schema, indent=2, default=str)}
                
                QUALITY ANALYSIS RESULTS:
                - Missing values: {quality_report['missing_values']['missing_percentage']:.1f}%
                - Duplicates: {quality_report['duplicates']['duplicate_percentage']:.1f}%
                - Outliers: {quality_report['outliers'].get('outlier_percentage', 0):.1f}%
                - Numeric columns: {quality_report.get('numeric_analysis', {}).get('numeric_columns', [])}
                - Categorical columns: {quality_report.get('categorical_analysis', {}).get('categorical_columns', [])}
                
                Create at least 6 comprehensive visualizations:
                1. Missing values heatmap showing patterns
                2. Distribution plots for all numeric columns (subplots)
                3. Correlation matrix heatmap for numeric columns
                4. Box plots for outlier detection in numeric columns
                5. Bar charts for categorical column value counts
                6. Pair plot or scatter matrix for numeric relationships
                
                Requirements:
                - Use plotly for interactive visualizations
                - Import all necessary libraries at the beginning
                - The dataframe variable is 'df'
                - Store each figure in variables: fig1, fig2, fig3, fig4, fig5, fig6
                - Add proper titles and labels to all plots
                - Handle edge cases (e.g., no numeric columns, all missing values)
                - Include print statements for any insights discovered
                
                Return only executable Python code.
                """
                
                viz_code = llm_client.generate_code(viz_prompt)
                
                # Store in session state
                st.session_state.viz_code = viz_code
                st.session_state.viz_generated = True
                
                logger.log("Visualization code generated", {"code_length": len(viz_code)})
                st.success("Visualization code generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating visualization code: {str(e)}")
                logger.log("Visualization generation error", {"error": str(e)})
    
    # Display and execute visualization code if generated
    if hasattr(st.session_state, 'viz_code') and st.session_state.viz_code:
        st.subheader("📊 Generated Visualization Code")
        
        # Show the code
        with st.expander("View/Edit Visualization Code", expanded=True):
            if hasattr(st.session_state, 'edit_viz_code') and st.session_state.edit_viz_code:
                # Edit mode
                edited_code = st.text_area(
                    "Edit the visualization code:",
                    value=st.session_state.viz_code,
                    height=400,
                    key="viz_code_editor"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Changes", key="save_viz_code"):
                        st.session_state.viz_code = edited_code
                        st.session_state.edit_viz_code = False
                        st.success("Code updated!")
                        st.rerun()
                
                with col2:
                    if st.button("❌ Cancel", key="cancel_viz_edit"):
                        st.session_state.edit_viz_code = False
                        st.rerun()
            else:
                # Display mode
                st.code(st.session_state.viz_code, language='python')
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Execute Visualizations", type="primary", key="execute_viz"):
                execute_visualization_code(st.session_state.viz_code, logger)
        
        with col2:
            if st.button("✏️ Edit Code", key="edit_viz_button"):
                st.session_state.edit_viz_code = True
                st.rerun()
        
        with col3:
            if st.button("🔄 Regenerate", key="regenerate_viz"):
                if 'viz_code' in st.session_state:
                    del st.session_state.viz_code
                if 'viz_generated' in st.session_state:
                    del st.session_state.viz_generated
                st.rerun()

def execute_visualization_code(viz_code, logger):
    """Execute the visualization code and display results"""
    with st.spinner("Executing visualization code..."):
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import numpy as np
            import warnings
            warnings.filterwarnings('ignore')
            
            # Capture output
            from io import StringIO
            import sys
            captured_output = StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured_output
            
            # Create execution environment
            exec_globals = {
                'df': st.session_state.data.copy(),
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'make_subplots': make_subplots,
                'plt': plt,
                'sns': sns,
                'print': print,
                'warnings': warnings
            }
            
            # Execute the code
            exec(viz_code, exec_globals)
            
            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            # Display any printed output
            if output:
                st.subheader("📝 Analysis Output")
                st.text(output)
            
            # Display the generated visualizations
            st.subheader("📈 Generated Visualizations")
            
            # Check for figures and display them
            for i in range(1, 10):  # Check for up to 9 figures
                fig_name = f'fig{i}'
                if fig_name in exec_globals:
                    st.plotly_chart(exec_globals[fig_name], use_container_width=True)
            
            logger.log("Visualizations executed successfully", {"figures_generated": sum(1 for i in range(1, 10) if f'fig{i}' in exec_globals)})
            st.success("Visualizations generated successfully!")
            
        except Exception as e:
            st.error(f"Error executing visualization code: {str(e)}")
            logger.log("Visualization execution error", {"error": str(e), "traceback": traceback.format_exc()})
            
            # Show the error and offer to fix
            st.warning("Would you like to regenerate the code or edit it manually?")

def display_complete_analysis(quality_report, enhanced_analysis):
    """Display complete analysis from quality report"""
    st.subheader("Complete Data Quality Analysis")
    
    # Basic Information
    with st.expander("📊 Basic Information", expanded=True):
        basic_info = quality_report.get('basic_info', {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Dimensions:**")
            st.write(f"- Rows: {basic_info.get('row_count', 0):,}")
            st.write(f"- Columns: {basic_info.get('column_count', 0)}")
            st.write(f"- Total Cells: {basic_info.get('total_cells', 0):,}")
        
        with col2:
            st.write("**Memory & Data:**")
            st.write(f"- Memory Usage: {basic_info.get('memory_usage_mb', 0):.2f} MB")
            st.write(f"- Non-null Cells: {basic_info.get('non_null_cells', 0):,}")
            st.write(f"- Null Cells: {basic_info.get('null_cells', 0):,}")
    
    # Missing Values Analysis
    with st.expander("🔍 Missing Values Analysis", expanded=True):
        missing_info = quality_report['missing_values']
        
        st.write(f"**Overall Missing: {missing_info['missing_percentage']:.2f}%**")
        
        if missing_info['columns_with_missing']:
            missing_df = pd.DataFrame(missing_info['columns_with_missing']).T
            missing_df = missing_df.sort_values('percentage', ascending=False)
            
            # Create a bar chart for missing values
            fig = px.bar(
                x=missing_df.index,
                y=missing_df['percentage'],
                title="Missing Values by Column",
                labels={'x': 'Column', 'y': 'Missing %'},
                color=missing_df['percentage'],
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(missing_df, use_container_width=True)
            
            if missing_info['problematic_columns']:
                st.error(f"⚠️ High missing columns (>50%): {', '.join(missing_info['problematic_columns'])}")
        
        # Missing patterns
        if missing_info.get('missing_patterns'):
            st.write("**Missing Value Patterns:**")
            patterns = missing_info['missing_patterns']
            st.write(f"- Rows with multiple missing: {patterns.get('rows_with_multiple_missing', 0)}")
            st.write(f"- Max missing per row: {patterns.get('max_missing_per_row', 0)}")
            
            if patterns.get('correlated_missing'):
                st.write("**Correlated Missing Values:**")
                for corr in patterns['correlated_missing']:
                    st.write(f"- {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.2f}")
    
    # Duplicates Analysis
    with st.expander("🔍 Duplicates Analysis"):
        dup_info = quality_report['duplicates']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Duplicates", f"{dup_info['total_duplicates']:,}")
            st.metric("Duplicate %", f"{dup_info['duplicate_percentage']:.2f}%")
        
        with col2:
            st.metric("Unique Rows", f"{dup_info['unique_rows']:,}")
            if dup_info.get('is_problematic'):
                st.error("⚠️ High duplicate rate detected!")
        
        if dup_info.get('partial_duplicates'):
            st.write("**Partial Duplicates by Column:**")
            for col, info in dup_info['partial_duplicates'].items():
                st.write(f"- {col}: {info['count']} ({info['percentage']:.1f}%)")
    
    # Data Types Analysis
    with st.expander("🔍 Data Types Analysis"):
        type_info = quality_report['data_types']
        
        st.write("**Current Data Types Distribution:**")
        type_counts = pd.Series(type_info['current_types'])
        fig = px.pie(values=type_counts.values, names=type_counts.index, title="Data Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        if type_info['type_suggestions']:
            st.write("**Suggested Type Changes:**")
            for col, suggestions in type_info['type_suggestions'].items():
                st.write(f"• **{col}**: {', '.join(suggestions)}")
    
    # Outliers Analysis
    with st.expander("🔍 Outliers Analysis"):
        outlier_info = quality_report['outliers']
        
        st.write(f"**Total Outlier Rows: {outlier_info.get('total_outlier_rows', 0)} ({outlier_info.get('outlier_percentage', 0):.1f}%)**")
        
        if outlier_info['columns_with_outliers']:
            st.write("**Outliers by Column:**")
            outlier_data = []
            for col, info in outlier_info['columns_with_outliers'].items():
                outlier_data.append({
                    'Column': col,
                    'Outlier Count': info['count'],
                    'IQR Method': info['methods']['iqr']['count'],
                    'Z-Score Method': info['methods'].get('zscore', {}).get('count', 0)
                })
            
            outlier_df = pd.DataFrame(outlier_data)
            st.dataframe(outlier_df, use_container_width=True)
    
    # Categorical Analysis
    if 'categorical_analysis' in quality_report:
        with st.expander("🔍 Categorical Columns Analysis"):
            cat_info = quality_report['categorical_analysis']
            
            if cat_info['categorical_columns']:
                st.write(f"**Categorical Columns: {', '.join(cat_info['categorical_columns'])}**")
                
                for col in cat_info['categorical_columns'][:5]:  # Show first 5
                    if col in cat_info['value_distributions']:
                        dist = cat_info['value_distributions'][col]
                        st.write(f"\n**{col}:**")
                        st.write(f"- Unique values: {dist['unique_count']}")
                        st.write(f"- Rare values (appearing once): {dist['rare_values']}")
                        
                        # Show top values
                        if dist['top_values']:
                            top_df = pd.DataFrame(list(dist['top_values'].items()), columns=['Value', 'Count'])
                            st.dataframe(top_df, use_container_width=True)
                
                if cat_info.get('inconsistencies'):
                    st.warning("**Data Inconsistencies Found:**")
                    for col, issues in cat_info['inconsistencies'].items():
                        st.write(f"**{col}:**")
                        for issue in issues:
                            st.write(f"- {issue['type']}: {issue['variants']}")
                            st.write(f"  Suggested: '{issue['suggested_value']}'")
    
    # Numeric Analysis
    if 'numeric_analysis' in quality_report:
        with st.expander("🔍 Numeric Columns Analysis"):
            num_info = quality_report['numeric_analysis']
            
            if num_info['numeric_columns']:
                st.write(f"**Numeric Columns: {', '.join(num_info['numeric_columns'])}**")
                
                # Distribution info
                if num_info.get('distributions'):
                    dist_data = []
                    for col, dist in num_info['distributions'].items():
                        dist_data.append({
                            'Column': col,
                            'Skewness': f"{dist['skewness']:.2f}",
                            'Kurtosis': f"{dist['kurtosis']:.2f}",
                            'Normal': '✅' if dist['is_normal'] else '❌'
                        })
                    
                    if dist_data:
                        dist_df = pd.DataFrame(dist_data)
                        st.dataframe(dist_df, use_container_width=True)
                
                # High correlations
                if num_info.get('correlations') and num_info['correlations'].get('high_correlations'):
                    st.warning("**High Correlations Detected:**")
                    for corr in num_info['correlations']['high_correlations']:
                        st.write(f"- {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.2f}")
    
    # Data Consistency
    if 'data_consistency' in quality_report:
        with st.expander("🔍 Data Consistency Check"):
            consistency_info = quality_report['data_consistency']
            
            if consistency_info['issues']:
                st.warning(f"**Found {len(consistency_info['issues'])} consistency issues**")
                
                for issue in consistency_info['issues']:
                    st.write(f"**{issue['type']}** in column '{issue['column']}'")
                    if 'types_found' in issue:
                        st.write(f"  Types found: {', '.join(issue['types_found'])}")

def display_ai_insights(enhanced_analysis):
    """Display AI-generated insights"""
    st.subheader("AI-Enhanced Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.text_area("**Analysis Summary**", enhanced_analysis.get('summary', ''), height=150, disabled=True)
    
    with col2:
        if 'business_impact' in enhanced_analysis:
            st.info(f"**Business Impact:**\n{enhanced_analysis['business_impact']}")
    
    if 'critical_issues' in enhanced_analysis:
        st.error("**Critical Issues:**")
        for issue in enhanced_analysis['critical_issues']:
            st.write(f"• {issue}")
    
    if 'recommendations' in enhanced_analysis:
        st.warning("**Recommendations:**")
        for i, rec in enumerate(enhanced_analysis['recommendations'], 1):
            st.write(f"{i}. {rec}")
    
    if 'cleaning_strategy' in enhanced_analysis:
        st.success(f"**Suggested Strategy:** {enhanced_analysis['cleaning_strategy']}")

def display_raw_report(quality_report, enhanced_analysis):
    """Display raw analysis reports in JSON format"""
    st.subheader("Raw Analysis Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Quality Report (JSON)**")
        st.json(quality_report)
    
    with col2:
        st.write("**Enhanced Analysis (JSON)**")
        st.json(enhanced_analysis)

def handle_data_correction(logger, llm_client):
    """Handle data correction with comprehensive context"""
    st.header("🔧 Data Correction")
    
    if st.session_state.data is None:
        st.warning("No data available.")
        return
    
    data_corrector = DataCorrector(llm_client)
    
    # User input for additional instructions
    user_instructions = st.text_area(
        "Additional correction instructions (optional)",
        placeholder="e.g., 'Remove outliers beyond 3 standard deviations, standardize categorical labels, handle missing values with median for numeric and mode for categorical'",
        key="correction_instructions",
        height=100
    )
    
    # Error feedback section (if there was a previous error)
    if hasattr(st.session_state, 'correction_error'):
        st.error("Previous execution failed. The error has been included in the regeneration context.")
        with st.expander("View Previous Error"):
            st.code(st.session_state.correction_error)
    
    if st.button("Generate Correction Code", key="generate_correction_button"):
        if not llm_client or not llm_client.client:
            st.error("LLM client not available. Please check your OpenAI API key.")
            return
            
        with st.spinner("Generating comprehensive correction strategy..."):
            try:
                # Include previous error in context if exists
                error_context = ""
                if hasattr(st.session_state, 'correction_error'):
                    error_context = f"\n\nPREVIOUS ERROR TO AVOID:\n{st.session_state.correction_error}"
                    # Clear the error after including it
                    del st.session_state.correction_error
                
                # Enhanced generation with full context
                correction_code, strategy_summary = data_corrector.generate_correction_code_with_full_context(
                    st.session_state.data,
                    st.session_state.schema,
                    st.session_state.quality_report,
                    user_instructions + error_context
                )
                
                st.session_state.correction_code = correction_code
                st.session_state.strategy_summary = strategy_summary
                
                logger.log("Correction code generated", {
                    "strategy_summary": strategy_summary,
                    "includes_error_fix": bool(error_context)
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
        
        # Show what will be addressed
        with st.expander("📊 Issues to be Addressed"):
            quality_report = st.session_state.quality_report
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Quality Issues:**")
                st.write(f"• Missing Values: {quality_report['missing_values']['missing_percentage']:.1f}%")
                st.write(f"• Duplicates: {quality_report['duplicates']['duplicate_percentage']:.1f}%")
                st.write(f"• Outliers: {quality_report['outliers'].get('outlier_percentage', 0):.1f}%")
            
            with col2:
                st.write("**Schema Information:**")
                st.write(f"• Total Columns: {len(st.session_state.schema)}")
                st.write(f"• Data Types: {', '.join(set(str(v.get('dtype', 'unknown')) for v in st.session_state.schema.values()))}")
        
        st.subheader("🔧 Generated Correction Code")
        
        # Code display/edit section
        if hasattr(st.session_state, 'show_code_editor') and st.session_state.show_code_editor:
            # Edit mode
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
            # Display mode
            st.code(st.session_state.correction_code, language='python')
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Execute Correction", type="primary", key="execute_correction_button"):
                execute_correction_with_feedback(data_corrector, logger)
        
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
    
    else:
        st.info("👆 Click 'Generate Correction Code' to create a data cleaning strategy based on the complete analysis.")

def execute_correction_with_feedback(data_corrector, logger):
    """Execute correction code with error feedback capability"""
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
                
                # Clear any previous errors
                if 'correction_error' in st.session_state:
                    del st.session_state.correction_error
                
                # Show before/after comparison
                st.subheader("📊 Correction Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Original Rows", f"{execution_log['original_shape'][0]:,}")
                
                with col2:
                    st.metric("Final Rows", f"{execution_log['final_shape'][0]:,}")
                
                with col3:
                    rows_removed = execution_log.get('rows_removed', 0)
                    st.metric("Rows Removed", f"{rows_removed:,}")
                
                with col4:
                    cols_removed = execution_log.get('columns_removed', 0)
                    st.metric("Columns Removed", cols_removed)
                
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
                
                # Store error for feedback
                error_msg = ""
                if execution_log.get('errors'):
                    for error in execution_log['errors']:
                        error_msg += f"{error['type']}: {error['message']}\n"
                        if 'traceback' in error:
                            error_msg += f"Traceback:\n{error['traceback']}\n"
                
                st.session_state.correction_error = error_msg
                
                # Show errors
                st.subheader("🚨 Errors Encountered")
                for error in execution_log['errors']:
                    st.error(f"**{error['type']}**: {error['message']}")
                    
                    # Show traceback in expander
                    if 'traceback' in error:
                        with st.expander("Show detailed error"):
                            st.code(error['traceback'])
                
                st.warning("💡 The error has been captured. Click 'Regenerate Code' to create a new version that addresses this error.")
                
                # Offer to regenerate with error context
                if st.button("🔄 Regenerate with Error Fix", key="regen_with_error"):
                    # Clear the code to trigger regeneration
                    if 'correction_code' in st.session_state:
                        del st.session_state.correction_code
                    if 'strategy_summary' in st.session_state:
                        del st.session_state.strategy_summary
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error executing correction: {str(e)}")
            logger.log("Correction execution error", {
                "error": str(e), 
                "traceback": traceback.format_exc()
            })
            
            # Store error for feedback
            st.session_state.correction_error = f"Execution error: {str(e)}\n{traceback.format_exc()}"

def handle_finalization(logger, llm_client):
    """Handle final pipeline generation"""
    st.header("🎯 Finalization")
    
    pipeline_generator = PipelineGenerator(llm_client)
    
    if st.button("Generate Final Pipeline", type="primary"):
        with st.spinner("Generating pipeline..."):
            try:
                pipeline_code = pipeline_generator.generate_pipeline(
                    logger.get_logs(),
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
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            original_shape = st.session_state.original_data.shape if hasattr(st.session_state.original_data, 'shape') else (0, 0)
            st.metric("Original Rows", f"{original_shape[0]:,}")
        
        with col2:
            final_shape = st.session_state.data.shape if hasattr(st.session_state.data, 'shape') else (0, 0)
            st.metric("Final Rows", f"{final_shape[0]:,}")
        
        with col3:
            rows_removed = original_shape[0] - final_shape[0]
            st.metric("Total Rows Removed", f"{rows_removed:,}")
        
        st.success("✅ Data cleaning process completed! Check the Downloads tab for files.")

def display_logs(logger):
    """Display session logs"""
    st.header("📋 Session Logs")
    
    # Log filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        level_filter = st.selectbox(
            "Filter by Level",
            ["All", "INFO", "WARNING", "ERROR"],
            key="log_level_filter"
        )
    
    with col2:
        action_filter = st.text_input(
            "Filter by Action",
            placeholder="Enter keyword",
            key="log_action_filter"
        )
    
    with col3:
        limit = st.number_input(
            "Max Logs",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            key="log_limit"
        )
    
    # Get filtered logs
    logs = logger.get_logs(
        level=level_filter if level_filter != "All" else None,
        action_filter=action_filter if action_filter else None,
        limit=int(limit)
    )
    
    # Display logs
    if logs:
        st.write(f"Showing {len(logs)} log entries")
        
        for log_entry in reversed(logs):
            level = log_entry.get('level', 'INFO')
            icon = "ℹ️" if level == "INFO" else "⚠️" if level == "WARNING" else "❌"
            
            with st.expander(f"{icon} [{log_entry['timestamp']}] {log_entry['action']}"):
                # Display details in a formatted way
                details = log_entry.get('details', {})
                if details:
                    for key, value in details.items():
                        if isinstance(value, (dict, list)):
                            st.json(value)
                        else:
                            st.write(f"**{key}**: {value}")
    else:
        st.info("No logs available.")
    
    # Performance metrics
    with st.expander("📊 Performance Metrics"):
        metrics = logger.get_performance_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Operations", metrics['total_operations'])
            st.metric("Successful", metrics['successful_operations'])
        
        with col2:
            st.metric("LLM Interactions", metrics['llm_interactions'])
            st.metric("Data Operations", metrics['data_operations'])
        
        with col3:
            st.metric("Failed Operations", metrics['failed_operations'])
            if metrics['average_execution_time'] > 0:
                st.metric("Avg Execution Time", f"{metrics['average_execution_time']:.2f}s")

def handle_downloads(logger):
    """Handle file downloads"""
    st.header("📥 Downloads")
    
    if st.session_state.data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Cleaned dataset
            csv_data = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="📊 Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name=f"cleaned_data_{st.session_state.session_id[:8]}.csv",
                mime="text/csv",
                help="Download the cleaned dataset in CSV format"
            )
        
        with col2:
            # Log file - Use the logger's export function which handles serialization
            log_json = logger.export_logs(format="json")
            st.download_button(
                label="📋 Download Log File (JSON)",
                data=log_json,
                file_name=f"cleaning_log_{st.session_state.session_id[:8]}.json",
                mime="application/json",
                help="Download complete session logs"
            )
        
        with col3:
            # Pipeline script
            if hasattr(st.session_state, 'pipeline_code'):
                st.download_button(
                    label="🐍 Download Pipeline Script (.py)",
                    data=st.session_state.pipeline_code,
                    file_name=f"data_pipeline_{st.session_state.session_id[:8]}.py",
                    mime="text/plain",
                    help="Download reusable Python pipeline"
                )
        
        # Additional export options
        st.subheader("Additional Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as Excel
            if st.button("Generate Excel Report", key="gen_excel"):
                with st.spinner("Generating Excel report..."):
                    from io import BytesIO
                    output = BytesIO()
                    
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Cleaned data
                        st.session_state.data.to_excel(writer, sheet_name='Cleaned Data', index=False)
                        
                        # Original data (if available)
                        if hasattr(st.session_state, 'original_data') and isinstance(st.session_state.original_data, pd.DataFrame):
                            st.session_state.original_data.to_excel(writer, sheet_name='Original Data', index=False)
                        
                        # Schema
                        if st.session_state.schema:
                            schema_df = pd.DataFrame(st.session_state.schema).T
                            schema_df.to_excel(writer, sheet_name='Schema')
                        
                        # Quality Report Summary
                        if hasattr(st.session_state, 'quality_report'):
                            summary_data = {
                                'Metric': ['Quality Score', 'Missing %', 'Duplicates %', 'Outliers %'],
                                'Value': [
                                    st.session_state.quality_report.get('quality_score', 0),
                                    st.session_state.quality_report['missing_values']['missing_percentage'],
                                    st.session_state.quality_report['duplicates']['duplicate_percentage'],
                                    st.session_state.quality_report['outliers'].get('outlier_percentage', 0)
                                ]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Quality Summary', index=False)
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📑 Download Excel Report",
                        data=excel_data,
                        file_name=f"data_cleaning_report_{st.session_state.session_id[:8]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            # Export logs as text
            if st.button("Generate Text Report", key="gen_text"):
                text_report = logger.export_logs(format="txt")
                
                st.download_button(
                    label="📄 Download Text Report",
                    data=text_report,
                    file_name=f"cleaning_report_{st.session_state.session_id[:8]}.txt",
                    mime="text/plain"
                )
    else:
        st.info("No data available for download. Please process a file first.")

if __name__ == "__main__":
    main()