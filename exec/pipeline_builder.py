import os
from typing import List
from core.config import PIPELINE_DIR

PIPELINE_TEMPLATE = '''"""
Auto-generated cleaning pipeline.
Accepts input path, returns cleaned dataset saved to output path.
"""

import pandas as pd

def run_pipeline(input_path: str, output_path: str):
    # Ingest
    df = pd.read_parquet(input_path)
    # Steps
{steps}
    # Save
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True)
    ap.add_argument("--out", dest="output_path", required=True)
    args = ap.parse_args()
    run_pipeline(args.input_path, args.output_path)
'''

def build_pipeline(code_blocks: List[str], session_id: str) -> str:
    steps = []
    for i, blk in enumerate(code_blocks, start=1):
        # Indent and ensure uses df -> df
        body = "\n".join("    " + line for line in blk.splitlines())
        steps.append(f"    # Step {i}\n{body}\n")
    content = PIPELINE_TEMPLATE.replace("{steps}", "\n".join(steps))
    path = os.path.join(PIPELINE_DIR, f"pipeline_{session_id}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
