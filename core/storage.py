import os
import pandas as pd
from core.config import DATA_DIR, CLEANED_DIR

def save_df(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(CLEANED_DIR, filename)
    df.to_parquet(path, index=False)
    return path

def load_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
