"""Utility validation helpers."""

import pandas as pd

def validate_non_empty(value: str) -> bool:
    return bool(value and value.strip())


def validate_dataframe_columns(df: pd.DataFrame, required: list[str]) -> tuple[bool, str]:
    missing = [col for col in required if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "valid"
