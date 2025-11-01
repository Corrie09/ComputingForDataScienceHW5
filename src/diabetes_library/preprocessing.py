import pandas as pd

def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN in essential columns."""
    required_cols = ["age", "gender", "ethnicity"]
    return df.dropna(subset=required_cols)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing height and weight with mean."""
    for col in ["height", "weight"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Convert gender M/F to 1/0."""
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    return df


def encode_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ethnicity."""
    return pd.get_dummies(df, columns=["ethnicity"], drop_first=True)
