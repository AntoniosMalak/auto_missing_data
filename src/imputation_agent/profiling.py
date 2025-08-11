
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
        # infer dtype
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
        card = None
        if dtype in ("categorical", "boolean"):
            card = s.nunique(dropna=True)
        is_time = dtype == "datetime"
        cols.append(ColumnProfile(name=c, dtype=dtype, missing_rate=miss, cardinality=card, is_time=is_time))

    has_dt_index = np.issubdtype(df.index.dtype, np.datetime64)
    return Profile(n_rows=len(df), n_cols=df.shape[1], columns=cols, has_datetime_index=has_dt_index)


def parse_datetimes_inplace(df: pd.DataFrame) -> None:
    """Attempt to parse object columns as datetime where feasible."""
    for c in df.columns:
        if df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="raise")
                if parsed.notna().mean() > 0.9:
                    df[c] = parsed
            except Exception:
                pass
