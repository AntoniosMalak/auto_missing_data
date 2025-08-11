
from __future__ import annotations
from typing import Dict, List, Tuple
from .config import PipelineConfig
from .profiling import Profile, ColumnProfile

LOW_MISS = 0.05
MID_MISS = 0.3
HIGH_MISS = 0.6
HIGH_CARD = 50
BIG_DATA = 300_000

def bucket_missing(m: float) -> str:
    if m <= LOW_MISS: return "low"
    if m <= MID_MISS: return "medium"
    if m <= HIGH_MISS: return "high"
    return "very_high"

def taxonomy_methods_for_column(cp: ColumnProfile, profile: Profile, cfg: PipelineConfig) -> List[str]:
    miss_bucket = bucket_missing(cp.missing_rate)
    is_big = profile.n_rows >= cfg.limits.max_rows_for_mice
    methods: List[str] = []

    if cp.dtype == "numeric":
        methods.append("median")
        if cp.missing_rate <= cfg.limits.max_missing_rate_for_knn and not is_big:
            methods.append("knn")
        if not is_big:
            methods.append("mice")
    elif cp.dtype == "categorical":
        methods.append("most_frequent")
        if (cp.cardinality or 0) >= HIGH_CARD and miss_bucket in {"high", "very_high"}:
            methods.append("constant")
    elif cp.dtype == "boolean":
        methods.append("most_frequent")
    elif cp.dtype == "datetime":
        methods.append("ffill_bfill")
    else:
        methods.append("median")

    return methods[:cfg.limits.max_candidates_per_type]

def build_methods_map_from_taxonomy(profile: Profile, cfg: PipelineConfig):
    per_type_defaults = {
        "numeric": cfg.methods.numeric,
        "categorical": cfg.methods.categorical,
        "boolean": cfg.methods.boolean,
        "datetime": cfg.methods.datetime,
        "unknown": ["median"],
    }
    per_column = {cp.name: taxonomy_methods_for_column(cp, profile, cfg) for cp in profile.columns}
    return per_type_defaults, per_column
