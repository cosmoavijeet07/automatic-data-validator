import os
import pandas as pd
from typing import Dict, Any
from core.errors import ProfilingError
from core.config import DATA_DIR

def basic_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate basic quality report - same logic as before."""
    report = {
        "missing": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "outliers_numeric_cols": [],
        "inconsistent_categories": {},
        "empty_strings": {c: int((df[c] == "").sum()) for c in df.columns if df[c].dtype == "object"},
        "special_missing_tokens": {}
    }
    
    # outliers via IQR for numeric columns
    for c in df.select_dtypes(include=["number"]).columns:
        s = df[c].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        ub = q3 + 1.5 * iqr
        lb = q1 - 1.5 * iqr
        outliers = int(((s > ub) | (s < lb)).sum())
        report["outliers_numeric_cols"].append({"column": c, "outliers": outliers})
    
    # inconsistent categories: case variants
    for c in df.select_dtypes(include=["object"]).columns:
        s = df[c].dropna().astype(str)
        lowered = s.str.lower().value_counts()
        if len(lowered) < s.nunique():
            report["inconsistent_categories"][c] = "Case variations detected"
    
    # special missing tokens
    tokens = {"?","na","n/a","none","null","-","--"}
    special = {}
    for c in df.columns:
        sc = df[c].astype(str).str.strip().str.lower()
        special[c] = int(sc.isin(tokens).sum())
    report["special_missing_tokens"] = special
    return report

def export_sweetviz(df: pd.DataFrame, session_id: str) -> str:
    """Use sweetviz for comparison reports."""
    try:
        import sweetviz as sv
        report = sv.analyze(df)
        path = os.path.join(DATA_DIR, f"sweetviz_{session_id}.html")
        report.show_html(filepath=path, open_browser=False)
        return path
    except Exception as e:
        return export_custom_profile(df, session_id)

def export_dtale_profile(df: pd.DataFrame, session_id: str) -> str:
    """Use dtale for interactive profiling."""
    try:
        import dtale
        # Simple dtale integration without complex server setup
        path = os.path.join(DATA_DIR, f"dtale_profile_{session_id}.html")
        
        # Create a summary report instead of full dtale server
        html_content = f"""
        <html>
        <head><title>D-Tale Style Profile - {session_id}</title></head>
        <body>
        <h1>Dataset Profile</h1>
        <p><strong>Shape:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns</p>
        <h2>Basic Statistics</h2>
        {df.describe().to_html()}
        <h2>Data Types</h2>
        {df.dtypes.to_frame('Data Type').to_html()}
        <h2>Missing Values</h2>
        {df.isnull().sum().to_frame('Missing Count').to_html()}
        </body>
        </html>
        """
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return path
    except Exception as e:
        return export_custom_profile(df, session_id)

def export_missingno_profile(df: pd.DataFrame, session_id: str) -> str:
    """Use missingno for missing data visualization."""
    try:
        import missingno as msno
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        # Create missing data visualizations
        plots_html = ""
        
        # Missing data matrix
        fig, ax = plt.subplots(figsize=(12, 6))
        msno.matrix(df, ax=ax)
        plt.title('Missing Data Pattern')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plots_html += f'<h3>Missing Data Matrix</h3><img src="data:image/png;base64,{plot_data}" style="max-width:100%;"><br><br>'
        plt.close()
        
        # Missing data bar chart
        if df.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            msno.bar(df, ax=ax)
            plt.title('Missing Data Count by Column')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots_html += f'<h3>Missing Data Bar Chart</h3><img src="data:image/png;base64,{plot_data}" style="max-width:100%;"><br><br>'
            plt.close()
        
        html_content = f"""
        <html>
        <head><title>Missing Data Analysis - {session_id}</title></head>
        <body>
        <h1>Missing Data Profile</h1>
        <p><strong>Dataset:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns</p>
        <p><strong>Total Missing:</strong> {df.isnull().sum().sum():,} values</p>
        {plots_html}
        <h2>Missing Data Summary</h2>
        {df.isnull().sum().to_frame('Missing Count').to_html()}
        </body>
        </html>
        """
        
        path = os.path.join(DATA_DIR, f"missingno_{session_id}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return path
    except Exception as e:
        return export_custom_profile(df, session_id)

def export_custom_profile(df: pd.DataFrame, session_id: str) -> str:
    """Enhanced custom HTML profiling report using only pandas and matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import base64
        from io import BytesIO
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Generate comprehensive statistics
        numeric_summary = df.describe()
        object_summary = df.describe(include=['object'])
        missing_data = df.isnull().sum()
        data_types = df.dtypes
        
        # Additional insights
        duplicates = df.duplicated().sum()
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        plots_html = ""
        
        # 1. Missing values plot
        if missing_data.sum() > 0:
            plt.figure(figsize=(12, 6))
            missing_data[missing_data > 0].plot(kind='bar', color='coral')
            plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
            plt.xlabel('Columns')
            plt.ylabel('Missing Count')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots_html += f'<div style="text-align:center;"><h3>Missing Values Distribution</h3><img src="data:image/png;base64,{plot_data}" style="max-width:100%; border:1px solid #ddd; border-radius:8px;"></div><br>'
            plt.close()
        
        # 2. Data types distribution
        dtype_counts = df.dtypes.value_counts()
        plt.figure(figsize=(8, 6))
        dtype_counts.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        plt.title('Data Types Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plots_html += f'<div style="text-align:center;"><h3>Data Types Distribution</h3><img src="data:image/png;base64,{plot_data}" style="max-width:100%; border:1px solid #ddd; border-radius:8px;"></div><br>'
        plt.close()
        
        # 3. Correlation heatmap for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = numeric_df.corr()
            mask = corr_matrix.isnull()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, mask=mask, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix (Numeric Columns)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots_html += f'<div style="text-align:center;"><h3>Correlation Matrix</h3><img src="data:image/png;base64,{plot_data}" style="max-width:100%; border:1px solid #ddd; border-radius:8px;"></div><br>'
            plt.close()
        
        # Value counts for categorical columns
        categorical_html = ""
        cat_cols = df.select_dtypes(include=['object']).columns[:5]
        for col in cat_cols:
            top_values = df[col].value_counts().head(10)
            categorical_html += f"""
            <div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                <h4 style="color: #495057;">Top 10 Values in '{col}'</h4>
                {top_values.to_frame('Count').to_html(classes='table table-striped', border=0)}
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report - {session_id}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                    line-height: 1.6;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 15px 0; 
                    font-size: 0.9em;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px 8px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #4a90e2; 
                    color: white; 
                    font-weight: 600;
                }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e3f2fd; }}
                h1 {{ 
                    color: #2c3e50; 
                    text-align: center; 
                    margin-bottom: 30px; 
                    font-size: 2.5em;
                }}
                h2 {{ 
                    color: #34495e; 
                    border-bottom: 2px solid #4a90e2; 
                    padding-bottom: 10px; 
                    margin-top: 30px;
                }}
                h3 {{ color: #495057; margin-top: 25px; }}
                .overview {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin: 20px 0;
                }}
                .overview p {{ margin: 8px 0; font-size: 1.1em; }}
                .overview strong {{ color: #fff3cd; }}
                img {{ 
                    margin: 15px 0; 
                    border-radius: 8px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .footer {{ 
                    margin-top: 40px; 
                    padding-top: 20px; 
                    border-top: 2px solid #dee2e6; 
                    text-align: center; 
                    color: #6c757d; 
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Data Profile Report</h1>
                <p style="text-align: center; color: #6c757d; font-size: 1.1em;">Session: {session_id}</p>
                
                <div class="overview">
                    <h2 style="border: none; color: white; margin-top: 0;">📋 Dataset Overview</h2>
                    <p><strong>📏 Shape:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns</p>
                    <p><strong>💾 Memory Usage:</strong> {memory_usage:.2f} MB</p>
                    <p><strong>🔄 Duplicate Rows:</strong> {duplicates:,}</p>
                    <p><strong>❓ Missing Values:</strong> {missing_data.sum():,} total ({(missing_data.sum()/df.size*100):.2f}%)</p>
                </div>
                
                <h2>📈 Visualizations</h2>
                {plots_html}
                
                <h2>🏷️ Data Types</h2>
                {data_types.to_frame('Data Type').to_html(classes='table', border=0)}
                
                <h2>❓ Missing Values by Column</h2>
                {'<p style="color: #28a745; font-weight: bold;">✅ No missing values found!</p>' if missing_data.sum() == 0 else missing_data[missing_data > 0].to_frame('Missing Count').to_html(classes='table', border=0)}
                
                <h2>🔢 Numeric Columns Summary</h2>
                {'<p style="color: #6c757d;">No numeric columns found.</p>' if numeric_summary.empty else numeric_summary.to_html(classes='table', border=0)}
                
                <h2>📝 Text Columns Summary</h2>
                {'<p style="color: #6c757d;">No text columns found.</p>' if object_summary.empty else object_summary.to_html(classes='table', border=0)}
                
                <h2>📊 Categorical Data Analysis</h2>
                {categorical_html if categorical_html else '<p style="color: #6c757d;">No categorical columns to display.</p>'}
                
                <div class="footer">
                    <p>🕒 Generated on {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
                    <p>Powered by Custom Data Profiler v1.0</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        path = os.path.join(DATA_DIR, f"profile_{session_id}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return path
        
    except Exception as e:
        # Ultra-minimal fallback
        basic_html = f"""
        <html><body style="font-family: Arial, sans-serif; padding: 20px;">
        <h1>Basic Profile - {session_id}</h1>
        <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
        <h2>Basic Statistics</h2>
        {df.describe().to_html()}
        <h2>Data Types</h2>
        {df.dtypes.to_frame('Data Type').to_html()}
        </body></html>
        """
        path = os.path.join(DATA_DIR, f"basic_profile_{session_id}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(basic_html)
        return path

# Main functions for backward compatibility
def export_ydata_profile(df: pd.DataFrame, session_id: str) -> str:
    """Main profiling function - tries alternatives in order."""
    try:
        return export_sweetviz(df, session_id)
    except:
        try:
            return export_dtale_profile(df, session_id)
        except:
            try:
                return export_missingno_profile(df, session_id)
            except:
                return export_custom_profile(df, session_id)
