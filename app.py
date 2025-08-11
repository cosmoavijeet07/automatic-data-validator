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
import time
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
    # if 'proceed_to_analysis' not in st.session_state:
    #     st.session_state.proceed_to_analysis = False
    
    # if 'proceed_to_correction' not in st.session_state:
    #     st.session_state.proceed_to_correction = False
    
    # if 'proceed_to_finalization' not in st.session_state:
    #     st.session_state.proceed_to_finalization = False
    
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
        # if st.session_state.proceed_to_analysis:
        #     st.session_state.current_step = 'analysis'
        #     st.session_state.proceed_to_analysis = False
        #     st.rerun()
        
        # if st.session_state.proceed_to_correction:
        #     st.session_state.current_step = 'correction'
        #     st.session_state.proceed_to_correction = False
        #     st.rerun()
        
        # if st.session_state.proceed_to_finalization:
        #     st.error("Proceeding to finalization step")
        #     st.session_state.current_step = 'finalization'
        #     st.session_state.proceed_to_finalization = False
        #     st.rerun()
        # Handle current step - simplified version
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
        else:
            st.error("Unknown step in the process. Please reset the session.")
    
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
            st.session_state.current_step = 'analysis'
            st.rerun()

def handle_data_analysis(logger, llm_client):
    """Handle data quality analysis with visualizations"""
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
    
    # Store analysis results
    st.session_state.quality_report = quality_report
    st.session_state.enhanced_analysis = enhanced_analysis
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🔍 Detailed Report", "🤖 AI Insights"])
    
    with tab1:
        # Display overview metrics
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
    
    with tab2:
        st.subheader("Data Visualizations")
        
        # Generate visualizations using LLM
        if st.button("Generate Visualizations", key="generate_viz"):
            with st.spinner("Generating visualization code..."):
                try:
                    viz_prompt = f"""
                    Generate Python code to create 4 useful visualizations for this dataset.
                    Dataset columns: {list(st.session_state.data.columns)}
                    Data types: {st.session_state.data.dtypes.to_dict()}
                    
                    Create code that:
                    1. Missing values heatmap
                    2. Distribution plots for numeric columns
                    3. Correlation matrix for numeric columns
                    4. Category counts for categorical columns
                    
                    Use plotly for interactive visualizations.
                    Return only executable Python code.
                    The dataframe variable is 'df'.
                    Store each figure in variables: fig1, fig2, fig3, fig4
                    """
                    
                    viz_code = llm_client.generate_code(viz_prompt)
                    
                    # Execute visualization code
                    import plotly.express as px
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    import numpy as np
                    
                    exec_globals = {
                        'df': st.session_state.data,
                        'pd': pd,
                        'np': np,
                        'px': px,
                        'go': go,
                        'make_subplots': make_subplots,
                        'plt': plt,
                        'sns': sns
                    }
                    
                    exec(viz_code, exec_globals)
                    
                    # Display the generated visualizations
                    if 'fig1' in exec_globals:
                        st.plotly_chart(exec_globals['fig1'], use_container_width=True)
                    
                    if 'fig2' in exec_globals:
                        st.plotly_chart(exec_globals['fig2'], use_container_width=True)
                    
                    if 'fig3' in exec_globals:
                        st.plotly_chart(exec_globals['fig3'], use_container_width=True)
                    
                    if 'fig4' in exec_globals:
                        st.plotly_chart(exec_globals['fig4'], use_container_width=True)
                    
                    with st.expander("View Generated Visualization Code"):
                        st.code(viz_code, language='python')
                    
                except Exception as e:
                    st.error(f"Error generating visualizations: {str(e)}")
                    
                    # Fallback to basic visualizations
                    st.info("Showing basic visualizations instead...")
                    
                    # Missing values bar chart
                    missing_df = pd.DataFrame({
                        'Column': st.session_state.data.columns,
                        'Missing %': [(st.session_state.data[col].isnull().sum() / len(st.session_state.data)) * 100 
                                     for col in st.session_state.data.columns]
                    })
                    fig = px.bar(missing_df, x='Column', y='Missing %', title='Missing Values by Column')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Detailed Quality Report")
        
        # Missing values details
        with st.expander("🔍 Missing Values Analysis", expanded=True):
            missing_info = quality_report['missing_values']
            
            if missing_info['columns_with_missing']:
                missing_df = pd.DataFrame(missing_info['columns_with_missing']).T
                st.dataframe(missing_df, use_container_width=True)
                
                if missing_info['problematic_columns']:
                    st.warning(f"⚠️ Problematic columns (>{50}% missing): {', '.join(missing_info['problematic_columns'])}")
        
        # Duplicates details
        with st.expander("🔍 Duplicates Analysis"):
            dup_info = quality_report['duplicates']
            st.write(f"Total duplicate rows: {dup_info['total_duplicates']}")
            st.write(f"Duplicate percentage: {dup_info['duplicate_percentage']:.2f}%")
            st.write(f"Unique rows: {dup_info['unique_rows']}")
        
        # Outliers details
        with st.expander("🔍 Outliers Analysis"):
            outlier_info = quality_report['outliers']
            if outlier_info['columns_with_outliers']:
                for col, info in outlier_info['columns_with_outliers'].items():
                    st.write(f"**{col}**: {info['count']} outliers detected")
        
        # Data types analysis
        with st.expander("🔍 Data Types Analysis"):
            type_info = quality_report['data_types']
            if type_info['type_suggestions']:
                st.write("**Suggested Type Changes:**")
                for col, suggestions in type_info['type_suggestions'].items():
                    st.write(f"• {col}: {', '.join(suggestions)}")
    
    with tab4:
        st.subheader("AI-Enhanced Analysis")
        st.text_area("Analysis Summary", enhanced_analysis.get('summary', ''), height=150)
        
        if 'critical_issues' in enhanced_analysis:
            st.write("**Critical Issues:**")
            for issue in enhanced_analysis['critical_issues']:
                st.write(f"• {issue}")
        
        if 'recommendations' in enhanced_analysis:
            st.write("**Recommendations:**")
            for rec in enhanced_analysis['recommendations']:
                st.write(f"• {rec}")
        
        if 'cleaning_strategy' in enhanced_analysis:
            st.info(f"**Suggested Strategy:** {enhanced_analysis['cleaning_strategy']}")
    
    # Proceed button
    st.markdown("---")
    if st.button("🔧 Proceed to Data Correction", type="primary", key="analysis_to_correction_btn"):
        st.session_state.current_step = 'correction'
        st.rerun()
from datetime import datetime

def handle_data_correction(logger, llm_client):
    """Handle the complete data correction workflow with proper state transitions"""
    st.header("🔧 Data Correction")
    
    # Initialize session state variables if they don't exist
    if 'correction_state' not in st.session_state:
        st.session_state.correction_state = {
            'generated': False,
            'executed': False,
            'show_proceed': False,
            'execution_log': None
        }
    
    # Debug display (remove in production)
    st.sidebar.write("Debug - Correction State:", st.session_state.correction_state)
    
    if st.session_state.data is None:
        st.warning("No data available. Please complete previous steps first.")
        return
    
    data_corrector = DataCorrector(llm_client)
    
    # Section 1: User Instructions and Code Generation
    with st.expander("⚙️ Correction Setup", expanded=not st.session_state.correction_state['generated']):
        user_instructions = st.text_area(
            "Additional correction instructions (optional)",
            placeholder="e.g., 'Remove outliers beyond 3 standard deviations, standardize categorical labels'",
            key="correction_instructions"
        )
        
        if st.button("Generate Correction Code", 
                    disabled=st.session_state.correction_state['generated'],
                    key="generate_correction_button"):
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
                    st.session_state.correction_state['generated'] = True
                    
                    logger.log("Correction code generated", {
                        "strategy_summary": strategy_summary
                    })
                    
                    st.success("Correction code generated successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating correction code: {str(e)}")
                    logger.log("Correction code generation error", {"error": str(e)})

    # Section 2: Display Generated Code
    if st.session_state.correction_state['generated']:
        with st.expander("📋 Generated Correction Code", expanded=True):
            st.subheader("Cleaning Strategy")
            st.info(st.session_state.strategy_summary)
            
            st.subheader("Implementation Code")
            st.code(st.session_state.correction_code, language='python')
            
            # Action buttons in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                execute_disabled = st.session_state.correction_state['executed']
                if st.button("✅ Execute Correction", 
                           disabled=execute_disabled,
                           type="primary",
                           key="execute_correction_button"):
                    with st.spinner("Executing correction..."):
                        try:
                            corrected_data, execution_log = data_corrector.execute_correction_code(
                                st.session_state.data,
                                st.session_state.correction_code
                            )
                            
                            st.session_state.data = corrected_data
                            st.session_state.correction_state.update({
                                'executed': True,
                                'show_proceed': True,
                                'execution_log': execution_log
                            })
                            
                            logger.log("Data correction executed", execution_log)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error executing correction: {str(e)}")
                            logger.log("Correction execution error", {
                                "error": str(e), 
                                "traceback": traceback.format_exc()
                            })
            
            with col2:
                if st.button("✏️ Modify Code", key="modify_code_button"):
                    st.session_state.show_code_editor = True
                    st.rerun()
            
            with col3:
                if st.button("🔄 Regenerate Code", key="regenerate_code_button"):
                    st.session_state.correction_state['generated'] = False
                    if 'correction_code' in st.session_state:
                        del st.session_state.correction_code
                    if 'strategy_summary' in st.session_state:
                        del st.session_state.strategy_summary
                    st.rerun()
        
        # Code editor (if modification requested)
        if hasattr(st.session_state, 'show_code_editor') and st.session_state.show_code_editor:
            modified_code = st.text_area(
                "Edit the correction code:",
                value=st.session_state.correction_code,
                height=400,
                key="code_editor"
            )
            
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("💾 Save Changes"):
                    st.session_state.correction_code = modified_code
                    st.session_state.show_code_editor = False
                    st.success("Code updated successfully!")
                    st.rerun()
            with col_cancel:
                if st.button("❌ Cancel"):
                    st.session_state.show_code_editor = False
                    st.rerun()

    # Section 3: Show Results and Proceed Button
    if st.session_state.correction_state['executed']:
        st.markdown("---")
        st.subheader("📊 Correction Results")
        
        # Display execution results
        execution_log = st.session_state.correction_state['execution_log']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Rows", execution_log['original_shape'][0])
        
        with col2:
            st.metric("Final Rows", execution_log['final_shape'][0])
        
        with col3:
            rows_removed = execution_log.get('rows_removed', 0)
            st.metric("Rows Removed", rows_removed)
        
        # Show execution output if available
        if execution_log.get('output'):
            with st.expander("📋 Execution Log"):
                for line in execution_log['output']:
                    if line.strip():
                        st.text(line)
        
        # Display updated dataset preview
        st.subheader("🧹 Cleaned Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Only show proceed button when execution is complete
        if st.session_state.correction_state['show_proceed']:
            st.markdown("---")
            if st.button("🎯 Proceed to Finalization", 
                        type="primary", 
                        key="proceed_to_final_btn"):
                # Clear correction state
                del st.session_state.correction_state
                if 'correction_code' in st.session_state:
                    del st.session_state.correction_code
                
                # Move to next step
                st.session_state.current_step = 'finalization'
                st.rerun()

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
                st.session_state.correction_executed = True  # Add flag for successful execution
                logger.log("Data correction executed", execution_log)
                
                st.success("🎉 Data correction completed successfully!")
                st.write("DEBUG: Inside execute_correction")
                st.write(f"Current step: {st.session_state.get('current_step')}")
                st.write(f"Correction complete: {st.session_state.get('correction_complete', False)}")
                
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
                    
                
                # Display the updated dataset
                st.subheader("📊 Updated Dataset Preview")
                st.dataframe(st.session_state.data.head(10), use_container_width=True)
                
                # Show data info
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Shape:** {st.session_state.data.shape}")
                with col2:
                    st.info(f"**Memory:** {st.session_state.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                
                # Proceed button
                st.markdown("---")
                if st.button("🎯 Proceed to Finalization", type="primary", key="correction_to_finalization_btn"):
                    del st.session_state.correction_complete
                    st.session_state.current_step = 'finalization'
                    st.session_state.correction_complete = False
                    time.sleep(0.1)
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
    
    if st.session_state.data is None:
        st.warning("No data available. Please complete the previous steps.")
        return
    
    pipeline_generator = PipelineGenerator(llm_client)
    
    # Generate pipeline button
    if st.button("🚀 Generate Final Pipeline", type="primary", key="generate_pipeline_btn"):
        with st.spinner("Generating complete pipeline..."):
            try:
                # Get all logs for pipeline generation
                all_logs = logger.get_logs()
                
                # Generate the pipeline
                pipeline_code = pipeline_generator.generate_pipeline(
                    all_logs,
                    st.session_state.schema,
                    getattr(st.session_state, 'correction_code', '')
                )
                
                st.session_state.pipeline_code = pipeline_code
                st.session_state.pipeline_generated = True
                logger.log("Pipeline generated", {"pipeline_length": len(pipeline_code)})
                
                st.success("✅ Pipeline generated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating pipeline: {str(e)}")
                logger.log("Pipeline generation error", {"error": str(e)})
                return
    
    # Display generated pipeline and downloads
    if hasattr(st.session_state, 'pipeline_code') and st.session_state.pipeline_code:
        st.subheader("📝 Generated Pipeline")
        
        # Show pipeline code in expander
        with st.expander("View Pipeline Code", expanded=False):
            st.code(st.session_state.pipeline_code, language='python')
        
        # Final data preview
        st.subheader("✨ Final Cleaned Data")
        
        # Data preview
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Summary metrics
        st.subheader("📊 Cleaning Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            original_shape = st.session_state.original_data.shape if hasattr(st.session_state, 'original_data') and hasattr(st.session_state.original_data, 'shape') else (0, 0)
            st.metric("Original Rows", f"{original_shape[0]:,}")
        
        with col2:
            final_shape = st.session_state.data.shape if hasattr(st.session_state.data, 'shape') else (0, 0)
            st.metric("Final Rows", f"{final_shape[0]:,}")
        
        with col3:
            rows_removed = original_shape[0] - final_shape[0]
            st.metric("Rows Removed", f"{rows_removed:,}")
        
        with col4:
            reduction_pct = (rows_removed / original_shape[0] * 100) if original_shape[0] > 0 else 0
            st.metric("Reduction %", f"{reduction_pct:.1f}%")
        
        # Downloads section
        st.markdown("---")
        st.subheader("📥 Download Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download cleaned dataset
            st.markdown("### 📊 Cleaned Dataset")
            csv_data = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"cleaned_data_{st.session_state.session_id[:8]}.csv",
                mime="text/csv",
                help="Download the final cleaned dataset",
                type="primary"
            )
        
        with col2:
            # Download pipeline script
            st.markdown("### 🐍 Pipeline Script")
            st.download_button(
                label="Download Pipeline (.py)",
                data=st.session_state.pipeline_code,
                file_name=f"data_pipeline_{st.session_state.session_id[:8]}.py",
                mime="text/plain",
                help="Reusable Python pipeline script",
                type="primary"
            )
        
        with col3:
            # Download complete log file
            st.markdown("### 📋 Session Logs")
            log_json = logger.export_logs(format="json")
            st.download_button(
                label="Download Logs (JSON)",
                data=log_json,
                file_name=f"cleaning_log_{st.session_state.session_id[:8]}.json",
                mime="application/json",
                help="Complete session logs with all operations",
                type="primary"
            )
        
        # Additional export options
        st.markdown("---")
        st.subheader("📑 Additional Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate Excel report
            if st.button("📑 Generate Excel Report", key="gen_excel_final"):
                with st.spinner("Generating comprehensive Excel report..."):
                    try:
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
                                quality_summary = pd.DataFrame({
                                    'Metric': ['Quality Score', 'Missing %', 'Duplicates %', 'Outliers %'],
                                    'Value': [
                                        st.session_state.quality_report.get('quality_score', 0),
                                        st.session_state.quality_report.get('missing_values', {}).get('missing_percentage', 0),
                                        st.session_state.quality_report.get('duplicates', {}).get('duplicate_percentage', 0),
                                        st.session_state.quality_report.get('outliers', {}).get('outlier_percentage', 0)
                                    ]
                                })
                                quality_summary.to_excel(writer, sheet_name='Quality Report', index=False)
                            
                            # Processing Summary
                            summary_data = {
                                'Metric': ['Original Rows', 'Final Rows', 'Rows Removed', 'Original Columns', 'Final Columns', 'Processing Steps'],
                                'Value': [
                                    st.session_state.original_data.shape[0] if hasattr(st.session_state, 'original_data') and hasattr(st.session_state.original_data, 'shape') else 0,
                                    st.session_state.data.shape[0],
                                    (st.session_state.original_data.shape[0] if hasattr(st.session_state, 'original_data') and hasattr(st.session_state.original_data, 'shape') else 0) - st.session_state.data.shape[0],
                                    st.session_state.original_data.shape[1] if hasattr(st.session_state, 'original_data') and hasattr(st.session_state.original_data, 'shape') else 0,
                                    st.session_state.data.shape[1],
                                    len(logger.get_logs())
                                ]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="📑 Download Excel Report",
                            data=excel_data,
                            file_name=f"complete_report_{st.session_state.session_id[:8]}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        st.success("Excel report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating Excel report: {str(e)}")
        
        with col2:
            # Generate text report
            if st.button("📄 Generate Text Report", key="gen_text_final"):
                try:
                    text_report = f"""
DATA CLEANING REPORT
====================
Session ID: {st.session_state.session_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Original Data Shape: {st.session_state.original_data.shape if hasattr(st.session_state, 'original_data') and hasattr(st.session_state.original_data, 'shape') else 'N/A'}
Final Data Shape: {st.session_state.data.shape}
Rows Removed: {rows_removed:,}
Reduction Percentage: {reduction_pct:.1f}%

QUALITY METRICS
--------------
Initial Quality Score: {st.session_state.quality_report.get('quality_score', 'N/A')}/100
Missing Values: {st.session_state.quality_report.get('missing_values', {}).get('missing_percentage', 0):.1f}%
Duplicates: {st.session_state.quality_report.get('duplicates', {}).get('duplicate_percentage', 0):.1f}%
Outliers: {st.session_state.quality_report.get('outliers', {}).get('outlier_percentage', 0):.1f}%

PROCESSING STEPS
---------------
1. File Upload and Validation
2. Schema Detection and Optimization
3. Data Quality Analysis
4. Data Correction and Cleaning
5. Pipeline Generation

LOGS
----
{logger.export_logs(format="txt")}
"""
                    
                    st.download_button(
                        label="📄 Download Text Report",
                        data=text_report,
                        file_name=f"cleaning_report_{st.session_state.session_id[:8]}.txt",
                        mime="text/plain"
                    )
                    st.success("Text report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating text report: {str(e)}")
        
        # Success message
        st.markdown("---")
        st.success("🎉 **Data cleaning process completed successfully!**")
        st.info("📌 All files are ready for download. The pipeline script can be used to replicate this cleaning process on new data.")
        
        # Option to start new session
        if st.button("🔄 Start New Session", key="new_session_btn"):
            reset_session_for_new_file()
            st.rerun()
    
    else:
        st.info("👆 Click 'Generate Final Pipeline' to complete the data cleaning process and enable downloads.")

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
                        
                        # Summary
                        summary_data = {
                            'Metric': ['Original Rows', 'Final Rows', 'Rows Removed', 'Columns'],
                            'Value': [
                                st.session_state.original_data.shape[0] if hasattr(st.session_state.original_data, 'shape') else 0,
                                st.session_state.data.shape[0],
                                (st.session_state.original_data.shape[0] if hasattr(st.session_state.original_data, 'shape') else 0) - st.session_state.data.shape[0],
                                st.session_state.data.shape[1]
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
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