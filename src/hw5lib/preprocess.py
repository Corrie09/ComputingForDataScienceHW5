"""Preprocessing classes for data cleaning."""

from typing import List
import pandas as pd


class NaNRowRemover:
    """Removes rows containing NaN values in specified columns."""
    
    def __init__(self, columns_to_check: List[str]):
        """
        Initialize the NaN row remover.
        
        Args:
            columns_to_check: List of column names to check for NaN values
        """
        self.columns_to_check = columns_to_check
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'NaNRowRemover':
        """
        Fit the preprocessor by validating columns exist.
        
        Args:
            df: Input dataframe
            
        Returns:
            self for method chaining
        """
        missing_cols = set(self.columns_to_check) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with NaN in specified columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe with NaN rows removed
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")
        
        return df.dropna(subset=self.columns_to_check).copy()
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        return self.fit(df).transform(df)


class NaNMeanFiller:
    """Fills NaN values with the mean of each column."""
    
    def __init__(self, columns_to_fill: List[str]):
        """
        Initialize the NaN mean filler.
        
        Args:
            columns_to_fill: List of column names to fill NaN values with mean
        """
        self.columns_to_fill = columns_to_fill
        self.means_ = {}  # Store learned means (sklearn convention: _ suffix for learned attributes)
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'NaNMeanFiller':
        """
        Fit the preprocessor by learning mean values from data.
        
        Args:
            df: Input dataframe (typically training data)
            
        Returns:
            self for method chaining
        """
        missing_cols = set(self.columns_to_fill) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")
        
        # Learn and store the mean for each column
        for col in self.columns_to_fill:
            self.means_[col] = df[col].mean()
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN values with learned means.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with NaN values filled
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")
        
        df_filled = df.copy()
        for col in self.columns_to_fill:
            df_filled[col] = df_filled[col].fillna(self.means_[col])
        
        return df_filled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with NaN values filled
        """
        return self.fit(df).transform(df)