from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

def load_split(csv_path: str | Path, set_col: str = "set"):
    """
    Load CSV and return (train_df, test_df), split by set_col == 'train'/'test'.
    """
    df = pd.read_csv(Path(csv_path))
    if set_col not in df.columns:
        raise ValueError(f"`{set_col}` not found in columns: {list(df.columns)}")
    train = df[df[set_col] == "train"].copy()
    test  = df[df[set_col] == "test"].copy()
    return train, test

class DataLoader:
    """
    Class with a primary method that loads the data and returns (train_df, test_df).
    """
    def __init__(self, csv_path: str | Path, set_col: str = "set"):
        self.csv_path = Path(csv_path)
        self.set_col = set_col

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return load_split(self.csv_path, self.set_col)
