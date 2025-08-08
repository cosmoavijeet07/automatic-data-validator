import pandas as pd
from core.models import ColumnSchema, DatasetSchema

def detect_schema(df: pd.DataFrame, source_type: str, sheets=None) -> DatasetSchema:
    cols = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        null_count = int(s.isna().sum())
        ex_vals = s.dropna().astype(str).head(5).tolist()
        is_cat = False
        if s.dtype == "object":
            unique_ratio = s.nunique(dropna=True) / max(1, len(s))
            is_cat = unique_ratio < 0.2 and s.nunique(dropna=True) < 1000
        date_fmt = None
        cols.append(ColumnSchema(name=str(c), dtype=dtype, is_categorical=is_cat, date_format=date_fmt, null_count=null_count, example_values=ex_vals))
    return DatasetSchema(columns=cols, n_rows=len(df), n_cols=len(df.columns), source_type=source_type, sheets=sheets)
