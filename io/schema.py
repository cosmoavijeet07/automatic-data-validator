import pandas as pd
from typing import Dict
from dateutil import parser as dateparser

def apply_schema_edits(df: pd.DataFrame, edits: Dict[str, Dict]) -> pd.DataFrame:
    # edits: {col: {"dtype": "int64/float/object/datetime", "date_format": "..."}}
    df2 = df.copy()
    for col, spec in edits.items():
        if "dtype" in spec:
            dt = spec["dtype"]
            if dt == "datetime":
                if "date_format" in spec and spec["date_format"]:
                    df2[col] = pd.to_datetime(df2[col], format=spec["date_format"], errors="coerce")
                else:
                    df2[col] = pd.to_datetime(df2[col], errors="coerce", infer_datetime_format=True)
            else:
                try:
                    df2[col] = df2[col].astype(dt)
                except Exception:
                    # best-effort for numeric
                    if dt in ("int64", "float64"):
                        df2[col] = pd.to_numeric(df2[col], errors="coerce")
    return df2
