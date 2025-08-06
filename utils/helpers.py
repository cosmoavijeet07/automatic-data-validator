import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional

def display_dataframe_info(dataframes: Optional[Dict[str, pd.DataFrame]], title: str = "Data Overview"):
    """
    Display overview information about dataframes with safe None handling
    """
    st.subheader(title)
    
    # ✅ SAFE CHECK: Handle None or empty dataframes
    if not dataframes:
        st.warning("⚠️ No data available to display.")
        st.info("Please ensure the previous processing step completed successfully.")
        return
    
    # ✅ SAFE CHECK: Ensure dataframes is a dictionary
    if not isinstance(dataframes, dict):
        st.error("❌ Invalid data format. Expected dictionary of dataframes.")
        st.write(f"Received: {type(dataframes)}")
        return
    
    # Safe iteration over dataframes
    for sheet_name, df in dataframes.items():
        # ✅ SAFE CHECK: Ensure df is a valid DataFrame
        if df is None:
            st.warning(f"⚠️ Sheet '{sheet_name}' contains no data (None)")
            continue
            
        if not isinstance(df, pd.DataFrame):
            st.warning(f"⚠️ Sheet '{sheet_name}' is not a valid DataFrame")
            continue
        
        with st.expander(f"📊 {sheet_name} ({len(df)} rows, {len(df.columns)} columns)"):
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Data Types:**")
                try:
                    dtype_df = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': df.count(),
                        'Null %': ((len(df) - df.count()) / len(df) * 100).round(2)
                    })
                    st.dataframe(dtype_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display data types: {str(e)}")
            
            with col2:
                st.write("**Sample Data:**")
                try:
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display sample data: {str(e)}")

def format_error_message(error: Exception) -> str:
    """
    Format error messages for user display
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    return f"**{error_type}**: {error_message}"

def estimate_processing_time(dataframes: Dict[str, pd.DataFrame]) -> str:
    """
    Estimate processing time based on data size
    """
    if not dataframes:
        return "No data to process"
        
    total_cells = sum(len(df) * len(df.columns) for df in dataframes.values() if df is not None)
    
    if total_cells < 10000:
        return "< 1 minute"
    elif total_cells < 100000:
        return "1-3 minutes"
    elif total_cells < 1000000:
        return "3-10 minutes"
    else:
        return "10+ minutes"

def create_download_filename(base_name: str, session_id: str, extension: str) -> str:
    """
    Create standardized download filename
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{session_id}_{timestamp}.{extension}"

def validate_file_size(uploaded_file) -> bool:
    """
    Validate uploaded file size (limit to 50MB)
    """
    if uploaded_file is None:
        return False
    max_size = 50 * 1024 * 1024  # 50MB in bytes
    return uploaded_file.size <= max_size

def get_memory_usage(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Get memory usage information for dataframes
    """
    usage_info = {}
    
    if not dataframes:
        return usage_info
    
    for sheet_name, df in dataframes.items():
        if df is None:
            usage_info[sheet_name] = "No data"
            continue
            
        try:
            memory_usage = df.memory_usage(deep=True).sum()
            
            if memory_usage < 1024:
                usage_str = f"{memory_usage} bytes"
            elif memory_usage < 1024 * 1024:
                usage_str = f"{memory_usage / 1024:.1f} KB"
            else:
                usage_str = f"{memory_usage / (1024 * 1024):.1f} MB"
            
            usage_info[sheet_name] = usage_str
        except Exception:
            usage_info[sheet_name] = "Unknown"
    
    return usage_info
