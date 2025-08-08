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
    """Use pandas-profiling as replacement for ydata-profiling."""
    try:
        from pandas_profiling import ProfileReport
        profile = ProfileReport(df, title=f"Profile {session_id}", explorative=True, minimal=True)
        path = os.path.join(DATA_DIR, f"profile_{session_id}.html")
        profile.to_file(path)
        return path
    except ImportError:
        # Fallback to dataprep if pandas_profiling is not available
        return export_dataprep_profile(df, session_id)
    except Exception as e:
        raise ProfilingError(str(e))

def export_dataprep_profile(df: pd.DataFrame, session_id: str) -> str:
    """Use dataprep as alternative profiling library."""
    try:
        from dataprep.eda import create_report
        report = create_report(df, title=f"DataPrep Profile {session_id}")
        path = os.path.join(DATA_DIR, f"dataprep_profile_{session_id}.html")
        report.save(path)
        return path
    except ImportError:
        # Final fallback to custom HTML report
        return export_custom_profile(df, session_id)
    except Exception as e:
        raise ProfilingError(str(e))

def export_custom_profile(df: pd.DataFrame, session_id: str) -> str:
    """Custom HTML profiling report as final fallback."""
    try:
        # Generate comprehensive statistics
        numeric_summary = df.describe()
        object_summary = df.describe(include=['object'])
        missing_data = df.isnull().sum()
        data_types = df.dtypes
        
        # Additional insights
        duplicates = df.duplicated().sum()
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Correlation matrix for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        correlation_html = ""
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            correlation_html = f"""
            <h2>Correlation Matrix</h2>
            {corr_matrix.to_html()}
            """
        
        # Value counts for categorical columns (top 10)
        categorical_html = ""
        for col in df.select_dtypes(include=['object']).columns[:5]:  # Limit to first 5 columns
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
            
            <h2>Data Types</h2>
            {data_types.to_frame('Data Type').to_html()}
            
            <h2>Missing Values by Column</h2>
            {missing_data[missing_data > 0].to_frame('Missing Count').to_html() if missing_data.sum() > 0 else '<p>No missing values found.</p>'}
            
            <h2>Numeric Columns Summary</h2>
            {numeric_summary.to_html() if not numeric_summary.empty else '<p>No numeric columns found.</p>'}
            
            <h2>Text Columns Summary</h2>
            {object_summary.to_html() if not object_summary.empty else '<p>No text columns found.</p>'}
            
            {correlation_html}
            
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
        raise ProfilingError(f"Custom profile report generation failed: {str(e)}")

def export_dtale_profile(df: pd.DataFrame, session_id: str) -> str:
    """Use dtale as another alternative for interactive profiling."""
    try:
        import dtale
        d = dtale.show(df, ignore_duplicate=True)
        # Generate static HTML export
        path = os.path.join(DATA_DIR, f"dtale_profile_{session_id}.html")
        # Note: dtale typically runs as a server, this is a placeholder for static export
        # You might need to customize this based on dtale's current API
        return path
    except ImportError:
        # Fallback to custom profile
        return export_custom_profile(df, session_id)
    except Exception as e:
        raise ProfilingError(str(e))

def export_autoviz_profile(df: pd.DataFrame, session_id: str) -> str:
    """Use AutoViz as visualization alternative."""
    try:
        from autoviz.AutoViz_Class import AutoViz_Class
        av = AutoViz_Class()
        path_dir = os.path.join(DATA_DIR, f"autoviz_{session_id}")
        os.makedirs(path_dir, exist_ok=True)
        
        # AutoViz generates multiple files, return directory path
        av.AutoViz(
            filename="",
            dfte=df,
            depVar="",
            verbose=0,
            chart_format="html",
            max_rows_analyzed=10000,
            max_cols_analyzed=30,
            save_plot_dir=path_dir
        )
        return path_dir
    except ImportError:
        # Fallback to custom profile
        return export_custom_profile(df, session_id)
    except Exception as e:
        raise ProfilingError(str(e))

# Main function that tries libraries in order of preference
def export_comprehensive_profile(df: pd.DataFrame, session_id: str) -> str:
    """Try multiple profiling libraries in order of preference."""
    try:
        # First try pandas_profiling (most similar to ydata-profiling)
        return export_pandas_profile(df, session_id)
    except:
        try:
            # Then try dataprep
            return export_dataprep_profile(df, session_id)
        except:
            try:
                # Then try dtale
                return export_dtale_profile(df, session_id)
            except:
                # Final fallback to custom HTML
                return export_custom_profile(df, session_id)

# Legacy function names for backward compatibility
def export_ydata_profile(df: pd.DataFrame, session_id: str) -> str:
    """Replacement for ydata-profiling using compatible alternatives."""
    return export_comprehensive_profile(df, session_id)

def export_sweetviz(df: pd.DataFrame, session_id: str) -> str:
    """Replacement for sweetviz using compatible alternatives."""
    try:
        import sweetviz as sv
        report = sv.analyze(df)
        path = os.path.join(DATA_DIR, f"sweetviz_{session_id}.html")
        report.show_html(filepath=path, open_browser=False)
        return path
    except ImportError:
        # Fallback if sweetviz is not compatible
        return export_comprehensive_profile(df, session_id)
    except Exception as e:
        # Fallback on any error
        return export_comprehensive_profile(df, session_id)
