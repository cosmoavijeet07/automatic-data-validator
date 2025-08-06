import pandas as pd
import json
import io
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import streamlit as st

def process_uploaded_file(uploaded_file) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Process uploaded file and return file info and dataframes
    """
    filename = uploaded_file.name
    file_type = filename.split('.')[-1].lower()
    
    file_info = {
        'filename': filename,
        'file_type': file_type,
        'size': uploaded_file.size
    }
    
    try:
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
            dataframes = {'Sheet1': df}
            
        elif file_type in ['xlsx', 'xls']:
            # Read all sheets
            excel_file = pd.ExcelFile(uploaded_file)
            dataframes = {}
            for sheet_name in excel_file.sheet_names:
                dataframes[sheet_name] = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                
        elif file_type == 'json':
            json_data = json.load(uploaded_file)
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                df = pd.json_normalize(json_data)
                dataframes = {'Sheet1': df}
            elif isinstance(json_data, dict):
                # Check if it's a single record or multiple sheets
                if all(isinstance(v, list) for v in json_data.values()):
                    # Multiple sheets format
                    dataframes = {}
                    for sheet_name, data in json_data.items():
                        dataframes[sheet_name] = pd.json_normalize(data)
                else:
                    # Single record
                    df = pd.json_normalize([json_data])
                    dataframes = {'Sheet1': df}
            else:
                raise ValueError("Unsupported JSON structure")
                
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Add metadata to file_info
        file_info.update({
            'sheets': list(dataframes.keys()),
            'total_rows': sum(len(df) for df in dataframes.values()),
            'total_columns': sum(len(df.columns) for df in dataframes.values())
        })
        
        return file_info, dataframes
        
    except Exception as e:
        raise Exception(f"Error processing {filename}: {str(e)}")

def create_excel_download(dataframes: Dict[str, pd.DataFrame]) -> bytes:
    """
    Create Excel file with multiple sheets for download
    """
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    buffer.seek(0)
    return buffer.getvalue()

def save_dataframes(dataframes: Dict[str, pd.DataFrame], session_id: str, prefix: str = "data") -> str:
    """
    Save dataframes to session directory
    """
    save_path = Path(f"downloads/{session_id}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    if len(dataframes) == 1:
        # Single sheet - save as CSV
        filename = f"{prefix}_{session_id}.csv"
        filepath = save_path / filename
        next(iter(dataframes.values())).to_csv(filepath, index=False)
    else:
        # Multiple sheets - save as Excel
        filename = f"{prefix}_{session_id}.xlsx"
        filepath = save_path / filename
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return str(filepath)

def get_data_sample(dataframes: Dict[str, pd.DataFrame], max_rows: int = 5) -> Dict[str, Any]:
    """
    Get sample data for schema detection
    """
    samples = {}
    
    for sheet_name, df in dataframes.items():
        sample_size = min(max_rows, len(df))
        sample_df = df.head(sample_size)
        
        samples[sheet_name] = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'sample_data': sample_df.to_dict('records'),
            'total_rows': len(df),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': df.nunique().to_dict()
        }
    
    return samples
