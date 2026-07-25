"""
Evaluation utility functions.
"""

from __future__ import annotations

from typing import Any
import pandas as pd


def normalize_val(val: Any) -> str:
    """
    Normalize ground truth and prediction values for comparison:
    strip(), lower(), replace(" ", "_"), replace("-", "_").
    """
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip().lower()
    return s.replace(" ", "_").replace("-", "_")


def extract_col(row: Any, *keys: str) -> str:
    """Extract a column value from a row using flexible case-insensitive and format-agnostic key matching."""
    for k in keys:
        if k in row and pd.notna(row[k]):
            return str(row[k]).strip()
    row_keys_norm = {str(rk).strip().lower().replace(" ", "_").replace("-", "_"): rk for rk in row.keys()}
    for k in keys:
        norm_k = k.lower().replace(" ", "_").replace("-", "_")
        if norm_k in row_keys_norm:
            actual_key = row_keys_norm[norm_k]
            val = row[actual_key]
            if pd.notna(val):
                return str(val).strip()
    return ""


PRODUCT_AREA_ALIASES = {
    "claude": "privacy",
    "conversation_management": "privacy",
    "chat_privacy": "privacy",
}


def match_product_area(expected_raw: Any, predicted_raw: Any) -> bool:
    """
    Match expected and predicted product area strings, supporting multi-label comma-separated strings
    and canonical domain aliases.
    """
    if not expected_raw:
        return True

    norm_exp = normalize_val(expected_raw)
    norm_pred = normalize_val(predicted_raw)

    if not norm_exp:
        return True
    if norm_exp == norm_pred:
        return True

    exp_tokens = set(t.strip() for t in norm_exp.replace(",", " ").split("_") if t.strip())
    pred_tokens = set(t.strip() for t in norm_pred.replace(",", " ").split("_") if t.strip())

    # Map aliases
    exp_mapped = {PRODUCT_AREA_ALIASES.get(t, t) for t in exp_tokens}
    pred_mapped = {PRODUCT_AREA_ALIASES.get(t, t) for t in pred_tokens}
    exp_mapped.add(PRODUCT_AREA_ALIASES.get(norm_exp, norm_exp))
    pred_mapped.add(PRODUCT_AREA_ALIASES.get(norm_pred, norm_pred))

    if exp_mapped & pred_mapped:
        return True
    if norm_exp in norm_pred or norm_pred in norm_exp:
        return True

    return False

