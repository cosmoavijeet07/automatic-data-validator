import pandas as pd
from typing import Dict, Any

def apply_corrections(df: pd.DataFrame, plan: Dict[str, Any]) -> pd.DataFrame:
    # plan keys: fill_missing, drop_duplicates, normalize_categories, outlier_handling, replacements
    df2 = df.copy()
    if plan.get("drop_duplicates", False):
        df2 = df2.drop_duplicates()
    fills = plan.get("fill_missing", {})
    for col, strat in fills.items():
        if strat == "median" and col in df2.select_dtypes(include=["number"]).columns:
            df2[col] = df2[col].fillna(df2[col].median())
        elif strat == "mean" and col in df2.select_dtypes(include=["number"]).columns:
            df2[col] = df2[col].fillna(df2[col].mean())
        elif strat == "mode":
            df2[col] = df2[col].fillna(df2[col].mode().iloc[0] if not df2[col].mode().empty else df2[col])
        elif isinstance(strat, (int, float, str)):
            df2[col] = df2[col].fillna(strat)
    # normalize_categories: lower-case trim
    for col in plan.get("normalize_categories", []):
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip().str.lower()
    # outlier handling
    for spec in plan.get("outlier_handling", []):
        col = spec.get("column")
        method = spec.get("method","clip")
        if col in df2.select_dtypes(include=["number"]).columns:
            s = df2[col]
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
            if method == "clip":
                df2[col] = s.clip(lb, ub)
            elif method == "remove":
                df2 = df2[(s >= lb) & (s <= ub)]
    # replacements
    for col, mapping in plan.get("replacements", {}).items():
        if col in df2.columns and isinstance(mapping, dict):
            df2[col] = df2[col].replace(mapping)
    return df2
