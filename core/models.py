from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import uuid
import time

@dataclass
class ColumnSchema:
    name: str
    dtype: str
    is_categorical: bool = False
    date_format: Optional[str] = None
    null_count: int = 0
    example_values: List[Any] = field(default_factory=list)

@dataclass
class DatasetSchema:
    columns: List[ColumnSchema] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    source_type: str = ""  # csv, excel, json, text
    sheets: Optional[List[str]] = None  # for excel

@dataclass
class StepLog:
    step: str
    timestamp: float
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    responses: List[Dict[str, Any]] = field(default_factory=list)
    code: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    result_summary: str = ""

@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    model_key: str = ""
    mode: str = "Human Reviewed"  # or "Automatic"
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    source_type: Optional[str] = None
    active_sheet: Optional[str] = None
    df_snapshot_path: Optional[str] = None
    schema: Optional[DatasetSchema] = None
    logs: List[StepLog] = field(default_factory=list)
    executed_code_blocks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    text_mode: bool = False
