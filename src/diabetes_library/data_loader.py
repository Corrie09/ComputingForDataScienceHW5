import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load diabetes dataset from CSV file."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filepath}") from e
