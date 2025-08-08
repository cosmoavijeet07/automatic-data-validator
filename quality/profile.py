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
    
    # outliers via IQR for numeric columns - same logic
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
    
    # inconsistent categories: case variants - same logic
    for c in df.select_dtypes(include=["object"]).columns:
        s = df[c].dropna().astype(str)
        lowered = s.str.lower().value_counts()
        if len(lowered) < s.nunique():
            report["inconsistent_categories"][c] = "Case variations detected"
    
    # special missing tokens - same logic
    tokens = {"?","na","n/a","none","null","-","--"}
    special = {}
    for c in df.columns:
        sc = df[c].astype(str).str.strip().str.lower()
        special[c] = int(sc.isin(tokens).sum())
    report["special_missing_tokens"] = special
    return report

def export_pandas_profile(df: pd.DataFrame, session_id: str) -> str:
    """Use pandas-profiling 3.2.0 (Python 3.13 compatible)."""
    try:
        from pandas_profiling import ProfileReport
        profile = ProfileReport(df, title=f"Profile {session_id}", explorative=True, minimal=True)
        path = os.path.join(DATA_DIR, f"profile_{session_id}.html")
        profile.to_file(path)
        return path
    except ImportError:
        return export_custom_profile(df, session_id)
    except Exception as e:
        raise ProfilingError(str(e))

def export_sweetviz(df: pd.DataFrame, session_id: str) -> str:
    """Use sweetviz for comparison reports."""
    try:
        import sweetviz as sv
        report = sv.analyze(df)
        path = os.path.join(DATA_DIR, f"sweetviz_{session_id}.html")
        report.show_html(filepath=path, open_browser=False)
        return path
    except ImportError:
        return export_custom_profile(df, session_id)
    except Exception as e:
        return export_custom_profile(df, session_id)

def export_dtale_profile(df: pd.DataFrame, session_id: str) -> str:
    """Use dtale for interactive profiling."""
    try:
        import dtale
        # Create a static HTML report
        d = dtale.show(df, ignore_duplicate=True, open_browser=False)
        path = os.path.join(DATA_DIR, f"dtale_profile_{session_id}.html")
        
        # Generate basic dtale export
        html_content = f"""
        <html>
        <head><title>D-Tale Profile - {session_id}</title></head>
        <body>
        <h1>D-Tale Interactive Profile</h1>
        <p>Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns</p>
        <p>For full interactive experience, run: <code>dtale.show(df)</code></p>
        {df.describe().to_html()}
        </body>
        </html>
        """
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        d.kill()  # Clean up dtale instance
        return path
    except ImportError:
        return export_custom_profile(df, session_id)
    except Exception as e:
        return export_custom_profile(df, session_id)

def export_custom_profile(df: pd.DataFrame, session_id: str) -> str:
    """Enhanced custom HTML profiling report."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        import base64
        from io import BytesIO
        
        # Generate comprehensive statistics
        numeric_summary = df.describe()
        object_summary = df.describe(include=['object'])
        missing_data = df.isnull().sum()
        data_types = df.dtypes
        
        # Additional insights
        duplicates = df.duplicated().sum()
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Generate visualizations
        plots_html = ""
        
        # Missing values plot
        if missing_data.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing_data[missing_data > 0].plot(kind='bar')
            plt.title('Missing Values by Column')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots_html += f'<img src="data:image/png;base64,{plot_data}" style="max-width:100%;"><br><br>'
            plt.close()
        
        # Correlation heatmap for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots_html += f'<img src="data:image/png;base64,{plot_data}" style="max-width:100%;"><br><br>'
            plt.close()
        
        # Value counts for categorical columns
        categorical_html = ""
        for col in df.select_dtypes(include=['object']).columns[:5]:
            top_values = df[col].value_counts().head(10)
            categorical_html += f"""
            <h3>Top Values in '{col}'</h3>
            {top_values.to_frame('Count').to_html()}
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report - {session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h1, h2, h3 {{ color: #333; }}
                .overview {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                img {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Data Profile Report - {session_id}</h1>
            
            <div class="overview">
                <h2>Dataset Overview</h2>
                <p><strong>Shape:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns</p>
                <p><strong>Memory Usage:</strong> {memory_usage:.2f} MB</p>
                <p><strong>Duplicate Rows:</strong> {duplicates:,}</p>
                <p><strong>Missing Values:</strong> {missing_data.sum():,} total</p>
            </div>
            
            <h2>Visualizations</h2>
            {plots_html}
            
            <h2>Data Types</h2>
            {data_types.to_frame('Data Type').to_html()}
            
            <h2>Missing Values by Column</h2>
            {missing_data[missing_data > 0].to_frame('Missing Count').to_html() if missing_data.sum() > 0 else '<p>No missing values found.</p>'}
            
            <h2>Numeric Columns Summary</h2>
            {numeric_summary.to_html() if not numeric_summary.empty else '<p>No numeric columns found.</p>'}
            
            <h2>Text Columns Summary</h2>
            {object_summary.to_html() if not object_summary.empty else '<p>No text columns found.</p>'}
            
            <h2>Categorical Columns - Top Values</h2>
            {categorical_html if categorical_html else '<p>No categorical columns to display.</p>'}
            
            <footer style="margin-top: 30px; padding-top: 15px; border-top: 1px solid #ccc; color: #666;">
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """
        
        path = os.path.join(DATA_DIR, f"profile_{session_id}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return path
        
    except Exception as e:
        # Minimal fallback without visualizations
        html_content = f"""
        <html><body>
        <h1>Basic Profile - {session_id}</h1>
        <p>Shape: {df.shape[0]} rows × {df.shape[1]} columns</p>
        {df.describe().to_html()}
        {df.dtypes.to_frame('Data Type').to_html()}
        </body></html>
        """
        path = os.path.join(DATA_DIR, f"profile_{session_id}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return path

# Main profiling function
def export_ydata_profile(df: pd.DataFrame, session_id: str) -> str:
    """Try multiple profiling approaches in order."""
    try:
        return export_pandas_profile(df, session_id)
    except:
        try:
            return export_sweetviz(df, session_id)
        except:
            try:
                return export_dtale_profile(df, session_id)
            except:
                return export_custom_profile(df, session_id)
