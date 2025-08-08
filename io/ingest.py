import os, json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from core.errors import IngestionError

def ingest_file(path: str) -> Tuple[str, Dict[str, pd.DataFrame], Optional[List[str]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
        return "csv", {"default": df}, None
    elif ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(path)
        sheets = xls.sheet_names
        dfs = {s: xls.parse(s) for s in sheets}
        return "excel", dfs, sheets
    elif ext in [".json"]:
        # Load JSON, convert to normalized table if possible
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            df = pd.json_normalize(data)
        except Exception:
            # try list of dicts assumption
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        return "json", {"default": df}, None
    else:
        raise IngestionError(f"Unsupported file type: {ext}")
