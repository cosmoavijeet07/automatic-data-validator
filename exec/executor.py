import pandas as pd
from typing import Dict, Any
from core.errors import CodeExecutionError

def exec_generated_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    # Restricted globals/locals for safety
    local_env = {"df": df.copy(), "pd": pd}
    try:
        exec(code, {"__builtins__": {"len": len, "range": range, "min": min, "max": max}}, local_env)
        # Expect the code to modify df or set df_out
        df_out = local_env.get("df_out", local_env.get("df"))
        if not isinstance(df_out, pd.DataFrame):
            raise CodeExecutionError("Generated code did not produce a DataFrame named df or df_out")
        return df_out
    except Exception as e:
        raise CodeExecutionError(str(e))
