"""
File handler module for processing different file types
"""

import pandas as pd
import json
import io
from pathlib import Path
from typing import Tuple, Dict, Any, Union
import streamlit as st
from config import MAX_FILE_SIZE_MB, SUPPORTED_FILE_TYPES

class FileHandler:
    """Handles file upload and processing for different file types"""
    
    def __init__(self):
        self.supported_types = SUPPORTED_FILE_TYPES
        self.max_size_mb = MAX_FILE_SIZE_MB
    
    def process_file(self, uploaded_file) -> Tuple[Union[pd.DataFrame, str], Dict[str, Any]]:
        """
        Process uploaded file and return data with file info
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (processed_data, file_info)
        """
        # Validate file
        self._validate_file(uploaded_file)
        
        file_extension = self._get_file_extension(uploaded_file.name)
        file_info = {
            'name': uploaded_file.name,
            'type': file_extension,
            'size_mb': uploaded_file.size / (1024 * 1024)
        }
        
        # Process based on file type
        if file_extension == 'csv':
            data = self._process_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            data = self._process_excel(uploaded_file)
        elif file_extension == 'json':
            data = self._process_json(uploaded_file)
        elif file_extension == 'txt':
            data = self._process_text(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return data, file_info
    
    def _validate_file(self, uploaded_file):
        """Validate uploaded file"""
        if uploaded_file is None:
            raise ValueError("No file provided")
        
        # Check file size
        if uploaded_file.size > self.max_size_mb * 1024 * 1024:
            raise ValueError(f"File size exceeds {self.max_size_mb}MB limit")
        
        # Check file extension
        file_extension = self._get_file_extension(uploaded_file.name)
        if file_extension not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        return Path(filename).suffix[1:].lower()
    
    def _process_csv(self, uploaded_file) -> pd.DataFrame:
        """Process CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    def _process_excel(self, uploaded_file) -> pd.DataFrame:
        """Process Excel file (handles multi-sheet)"""
        try:
            # Read Excel file to check sheets
            uploaded_file.seek(0)
            excel_file = pd.ExcelFile(uploaded_file)
            
            if len(excel_file.sheet_names) == 1:
                # Single sheet - read directly
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, sheet_name=0)
                return df
            else:
                # Multiple sheets - let user choose
                st.subheader("📊 Multiple Sheets Detected")
                st.info(f"This Excel file contains {len(excel_file.sheet_names)} sheets")
                
                # Create a unique key for the session state
                sheet_key = f"selected_sheets_{uploaded_file.name}"
                combine_key = f"combine_sheets_{uploaded_file.name}"
                
                # Initialize session state if not exists
                if sheet_key not in st.session_state:
                    st.session_state[sheet_key] = [excel_file.sheet_names[0]]
                if combine_key not in st.session_state:
                    st.session_state[combine_key] = False
                
                # Sheet selection
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_sheets = st.multiselect(
                        "Select sheets to process:",
                        excel_file.sheet_names,
                        default=st.session_state[sheet_key],
                        key=f"sheet_selector_{uploaded_file.name}"
                    )
                    
                    if selected_sheets:
                        st.session_state[sheet_key] = selected_sheets
                
                with col2:
                    if len(selected_sheets) > 1:
                        combine_sheets = st.checkbox(
                            "Combine selected sheets",
                            value=st.session_state[combine_key],
                            key=f"combine_checkbox_{uploaded_file.name}",
                            help="Check to combine all selected sheets into one DataFrame"
                        )
                        st.session_state[combine_key] = combine_sheets
                    else:
                        combine_sheets = False
                
                # Process based on selection
                if not selected_sheets:
                    st.warning("Please select at least one sheet")
                    # Default to first sheet
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file, sheet_name=excel_file.sheet_names[0])
                    st.info(f"Using default sheet: {excel_file.sheet_names[0]}")
                elif len(selected_sheets) == 1:
                    # Single sheet selected
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheets[0])
                    st.success(f"✅ Loaded sheet: {selected_sheets[0]}")
                else:
                    # Multiple sheets selected
                    if combine_sheets:
                        # Combine sheets
                        dfs = []
                        for sheet in selected_sheets:
                            uploaded_file.seek(0)
                            sheet_df = pd.read_excel(uploaded_file, sheet_name=sheet)
                            sheet_df['_source_sheet'] = sheet  # Add source sheet column
                            dfs.append(sheet_df)
                        
                        df = pd.concat(dfs, ignore_index=True)
                        st.success(f"✅ Combined {len(selected_sheets)} sheets into one DataFrame")
                        st.info(f"Added '_source_sheet' column to track data origin")
                    else:
                        # Process only the first selected sheet
                        uploaded_file.seek(0)
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheets[0])
                        st.info(f"Processing first selected sheet: {selected_sheets[0]}")
                        st.warning(f"Note: Only processing the first sheet. Select 'Combine sheets' to process all.")
                
                return df
                
        except Exception as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")
    
    def _process_json(self, uploaded_file) -> pd.DataFrame:
        """Process JSON file and convert to DataFrame"""
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read()
            
            # Try to decode JSON
            json_data = json.loads(content.decode('utf-8'))
            
            # Convert to DataFrame based on structure
            if isinstance(json_data, list):
                # Array of objects
                if len(json_data) > 0 and isinstance(json_data[0], dict):
                    df = pd.json_normalize(json_data)
                else:
                    # Simple list
                    df = pd.DataFrame(json_data, columns=['value'])
            elif isinstance(json_data, dict):
                # Check if it's a nested structure with arrays
                has_arrays = any(isinstance(v, list) for v in json_data.values())
                
                if has_arrays:
                    # Try to find the main data array
                    for key, value in json_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict):
                                df = pd.json_normalize(value)
                                break
                    else:
                        # No suitable array found, try normalizing the whole structure
                        df = pd.json_normalize(json_data)
                else:
                    # Single record or flat structure
                    df = pd.DataFrame([json_data])
            else:
                raise ValueError("JSON structure not suitable for tabular data")
            
            return df
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing JSON file: {str(e)}")
    
    def _process_text(self, uploaded_file) -> str:
        """Process text file"""
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read()
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text_content = content.decode(encoding)
                    return text_content
                except UnicodeDecodeError:
                    continue
            
            # If all fail, use utf-8 with error handling
            text_content = content.decode('utf-8', errors='ignore')
            return text_content
            
        except Exception as e:
            raise ValueError(f"Error processing text file: {str(e)}")
    
    def save_processed_data(self, data: Union[pd.DataFrame, str], session_id: str, 
                          file_type: str) -> str:
        """Save processed data to temporary location"""
        try:
            from config import TEMP_DIR
            
            if isinstance(data, pd.DataFrame):
                file_path = TEMP_DIR / f"{session_id}_data.csv"
                data.to_csv(file_path, index=False)
            else:
                file_path = TEMP_DIR / f"{session_id}_text.txt"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(data)
            
            return str(file_path)
            
        except Exception as e:
            raise ValueError(f"Error saving processed data: {str(e)}")
    
    def load_processed_data(self, file_path: str) -> Union[pd.DataFrame, str]:
        """Load previously processed data"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix == '.csv':
                return pd.read_csv(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
        except Exception as e:
            raise ValueError(f"Error loading processed data: {str(e)}")
    
    def get_file_stats(self, data: Union[pd.DataFrame, str]) -> Dict[str, Any]:
        """Get basic statistics about the processed file"""
        stats = {}
        
        if isinstance(data, pd.DataFrame):
            stats.update({
                'type': 'tabular',
                'rows': len(data),
                'columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
                'has_missing_values': data.isnull().any().any(),
                'numeric_columns': len(data.select_dtypes(include=['number']).columns),
                'text_columns': len(data.select_dtypes(include=['object']).columns),
                'datetime_columns': len(data.select_dtypes(include=['datetime']).columns)
            })
        else:
            stats.update({
                'type': 'text',
                'character_count': len(data),
                'word_count': len(data.split()) if data else 0,
                'line_count': len(data.splitlines()) if data else 0
            })
        
        return stats