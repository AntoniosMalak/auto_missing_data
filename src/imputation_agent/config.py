
from pydantic import BaseModel, Field
from typing import List, Dict


class EvalConfig(BaseModel):
    mask_frac: float = 0.1
    seeds: List[int] = [1, 2, 3]  # average metrics across seeds
    random_state: int = 42


class MethodsConfig(BaseModel):
    numeric: List[str] = ["median", "knn", "mice"]
    categorical: List[str] = ["most_frequent"]
    boolean: List[str] = ["most_frequent"]
    datetime: List[str] = ["ffill_bfill"]  # handled specially


class LimitsConfig(BaseModel):
    max_candidates_per_type: int = 3  # cap for agent or auto planner
    max_rows_for_mice: int = 300000
    max_missing_rate_for_knn: float = 0.4  # skip knn if missing higher than this


class PipelineConfig(BaseModel):
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    methods: MethodsConfig = Field(default_factory=MethodsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
