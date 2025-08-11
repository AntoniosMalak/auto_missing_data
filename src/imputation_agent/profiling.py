
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_rate: float
    cardinality: Optional[int] = None
    is_time: bool = False

@dataclass
class Profile:
    n_rows: int
    n_cols: int
    columns: List[ColumnProfile]
    has_datetime_index: bool

def infer_profile(df: pd.DataFrame) -> Profile:
    cols = []
    for c in df.columns:
        s = df[c]
        nonna = s.dropna()
        if len(nonna) == 0:
            dtype = "unknown"
        elif np.issubdtype(nonna.dtype, np.number):
            dtype = "numeric"
        elif np.issubdtype(nonna.dtype, np.datetime64):
            dtype = "datetime"
        elif nonna.dtype == bool:
            dtype = "boolean"
        else:
            dtype = "categorical"
        miss = s.isna().mean()
        card = s.nunique(dropna=True) if dtype in ("categorical","boolean") else None
        cols.append(ColumnProfile(c, dtype, miss, card, dtype=="datetime"))
    has_dt_index = np.issubdtype(df.index.dtype, np.datetime64)
    return Profile(len(df), df.shape[1], cols, has_dt_index)

def parse_datetimes_inplace(df: pd.DataFrame) -> None:
    for c in df.columns:
        if df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="raise")
                if parsed.notna().mean() > 0.9:
                    df[c] = parsed
            except Exception:
                pass
