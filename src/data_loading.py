import re
from pathlib import Path
import pandas as pd

def clean_column_name(column: str) -> str:
    column = column.strip().lower()
    column = re.sub(r"[()]", "", column)
    column = re.sub(r"['’]", "", column)
    column = re.sub(r"[/\-\s]+", "_", column)
    column = re.sub(r"[^a-z0-9_]", "", column)
    column = re.sub(r"_+", "_", column)
    return column.strip("_")

def load_source_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))

def clean_source_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    return df.rename(columns={"nacionality": "nationality"})

def validate_source_data(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": df["target"].value_counts(dropna=False).to_dict(),
    }

def build_binary_outcome(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["target"].isin(["Dropout", "Graduate"])].copy()
    out["dropout"] = (out["target"] == "Dropout").astype(int)
    return out
