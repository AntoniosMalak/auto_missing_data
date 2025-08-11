
from pydantic import BaseModel, Field
from typing import List

class EvalConfig(BaseModel):
    mask_frac: float = 0.1
    seeds: List[int] = [1, 2, 3]
    random_state: int = 42

class MethodsConfig(BaseModel):
    numeric: List[str] = ["median", "knn", "mice"]
    categorical: List[str] = ["most_frequent"]
    boolean: List[str] = ["most_frequent"]
    datetime: List[str] = ["ffill_bfill"]

class LimitsConfig(BaseModel):
    max_candidates_per_type: int = 3
    max_rows_for_mice: int = 300000
    max_missing_rate_for_knn: float = 0.4

class PipelineConfig(BaseModel):
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    methods: MethodsConfig = Field(default_factory=MethodsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
