"""Feature engineering classes for creating new features from data."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFeature(ABC):
    """
    Abstract base class for feature transformers.
    All feature classes must inherit from this and implement fit() and transform().
    """
    
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BaseFeature':
        """
        Learn parameters from the data if needed.
        
        Args:
            df: Input dataframe
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature transformation.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with new features added
        """
        pass
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        return self.fit(df).transform(df)


class BMICalculator(BaseFeature):
    """
    Creates BMI (Body Mass Index) feature from height and weight.
    BMI = weight (kg) / (height (m))^2
    """
    
    def __init__(
        self, 
        height_col: str = "height", 
        weight_col: str = "weight",
        output_col: str = "bmi"
    ):
        """
        Initialize BMI calculator.
        
        Args:
            height_col: Name of height column (assumed in cm)
            weight_col: Name of weight column (assumed in kg)
            output_col: Name for the new BMI column
        """
        super().__init__()
        self.height_col = height_col
        self.weight_col = weight_col
        self.output_col = output_col
    
    def fit(self, df: pd.DataFrame) -> 'BMICalculator':
        """
        Validate that required columns exist.
        
        Args:
            df: Input dataframe
            
        Returns:
            self
        """
        required_cols = [self.height_col, self.weight_col]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate BMI and add as new column.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with BMI column added
        """
        if not self.is_fitted:
            raise RuntimeError("Feature must be fitted before transform. Call fit() first.")
        
        df_transformed = df.copy()
        
        # Convert height from cm to meters and calculate BMI
        height_m = df_transformed[self.height_col] / 100
        df_transformed[self.output_col] = df_transformed[self.weight_col] / (height_m ** 2)
        
        return df_transformed


class GenderEncoder(BaseFeature):
    """
    Converts gender to numeric values.
    Learns the mapping from the training data.
    """
    
    def __init__(self, gender_col: str = "gender", output_col: str = "gender_numeric"):
        """
        Initialize gender encoder.
        
        Args:
            gender_col: Name of gender column
            output_col: Name for the new numeric gender column
        """
        super().__init__()
        self.gender_col = gender_col
        self.output_col = output_col
        self.gender_mapping_ = {}  # Will be learned during fit
    
    def fit(self, df: pd.DataFrame) -> 'GenderEncoder':
        """
        Learn unique gender values and create mapping.
        
        Args:
            df: Input dataframe
            
        Returns:
            self
        """
        if self.gender_col not in df.columns:
            raise ValueError(f"Column '{self.gender_col}' not found")
        
        # Get unique gender values and create numeric mapping
        unique_genders = df[self.gender_col].dropna().unique()
        self.gender_mapping_ = {gender: idx for idx, gender in enumerate(sorted(unique_genders))}
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply gender encoding using learned mapping.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with numeric gender column added
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        df_transformed = df.copy()
        df_transformed[self.output_col] = df_transformed[self.gender_col].map(self.gender_mapping_)
        
        return df_transformed


class AgeSquared(BaseFeature):
    """
    Creates age squared feature to capture non-linear age effects.
    Useful when the relationship between age and outcome is quadratic.
    """
    
    def __init__(self, age_col: str = "age", output_col: str = "age_squared"):
        """
        Initialize age squared feature creator.
        
        Args:
            age_col: Name of age column
            output_col: Name for the new age squared column
        """
        super().__init__()
        self.age_col = age_col
        self.output_col = output_col
    
    def fit(self, df: pd.DataFrame) -> 'AgeSquared':
        """
        Validate that age column exists.
        
        Args:
            df: Input dataframe
            
        Returns:
            self
        """
        if self.age_col not in df.columns:
            raise ValueError(f"Column '{self.age_col}' not found")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age squared feature.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with age squared column added
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        df_transformed = df.copy()
        df_transformed[self.output_col] = df_transformed[self.age_col] ** 2
        
        return df_transformed