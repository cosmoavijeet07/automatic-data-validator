import os
import pandas as pd
from typing import Dict, Any
from core.errors import ProfilingError
from core.config import DATA_DIR

def basic_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
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

def export_ydata_profile(df: pd.DataFrame, session_id: str) -> str:
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, title=f"Profile {session_id}", explorative=True, minimal=True)
        path = os.path.join(DATA_DIR, f"profile_{session_id}.html")
        profile.to_file(path)
        return path
    except Exception as e:
        raise ProfilingError(str(e))

def export_sweetviz(df: pd.DataFrame, session_id: str) -> str:
    try:
        import sweetviz as sv
        report = sv.analyze(df)
        path = os.path.join(DATA_DIR, f"sweetviz_{session_id}.html")
        report.show_html(filepath=path, open_browser=False)
        return path
    except Exception as e:
        raise ProfilingError(str(e))
