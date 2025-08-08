from typing import List
from core.models import DatasetSchema

def validate_schema(schema: DatasetSchema):
    assert schema.n_cols == len(schema.columns), "Schema mismatch: n_cols"
    assert schema.n_rows >= 0, "Negative rows"
