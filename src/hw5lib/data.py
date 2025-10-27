from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_TARGET = "diabetes_mellitus"

def load_split(
    csv_path: str | Path,
    *,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split on the target column (default: 'diabetes_mellitus').
    Returns (train_df, test_df).
    """
    df = pd.read_csv(Path(csv_path))
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    vc = df[target].dropna().value_counts()
    if len(vc) < 2:
        raise ValueError(f"Target '{target}' has <2 classes; cannot stratify. Value counts: {vc.to_dict()}")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target]
    )
    return train_df.copy(), test_df.copy()


class DataLoader:
    """
    Always performs a stratified train/test split on the target column.
    """
    def __init__(
        self,
        csv_path: str | Path,
        *,
        target: str = DEFAULT_TARGET,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return load_split(
            self.csv_path,
            target=self.target,
            test_size=self.test_size,
            random_state=self.random_state,
        )
