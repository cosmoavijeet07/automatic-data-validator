import pandas as pd

def to_preview(df: pd.DataFrame, n=50):
    return df.head(n)
