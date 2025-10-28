"""Model classes for training and prediction."""

from typing import List, Optional, Dict, Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class DiabetesModel:
    """
    Model class for diabetes prediction.
    Wraps sklearn models with a clean interface for training and prediction.
    """
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the diabetes prediction model.
        
        Args:
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            hyperparameters: Optional dictionary of model hyperparameters
            (e.g., {'n_estimators': 100, 'max_depth': 5})
        """
        # Private attributes (convention: prefix with _)
        self._feature_columns = feature_columns
        self._target_column = target_column
        self._hyperparameters = hyperparameters if hyperparameters is not None else {}
        
        # Public attribute: the sklearn model
        # Using RandomForestClassifier, but could be LogisticRegression, etc.
        self.model = RandomForestClassifier(**self._hyperparameters)
        
        # Track if model has been trained
        self._is_trained = False
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the model on the provided dataframe.
        
        Args:
            df: Training dataframe containing both features and target
            
        Returns:
            None
        """
        # Validate that required columns exist
        missing_features = set(self._feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        if self._target_column not in df.columns:
            raise ValueError(f"Target column '{self._target_column}' not found")
        
        # Extract features (X) and target (y)
        X = df[self._feature_columns]
        y = df[self._target_column]
        
        # Fit the model
        self.model.fit(X, y)
        self._is_trained = True
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict probabilities for the provided dataframe.
        
        Args:
            df: Dataframe containing feature columns
            
        Returns:
            DataFrame with predicted probabilities for each class
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before making predictions. Call train() first.")
        
        # Validate that feature columns exist
        missing_features = set(self._feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        # Extract features
        X = df[self._feature_columns]
        
        # Get predicted probabilities
        probabilities = self.model.predict_proba(X)
        
        # Return as DataFrame with class labels as columns
        prob_df = pd.DataFrame(
            probabilities,
            columns=[f"prob_class_{i}" for i in range(probabilities.shape[1])],
            index=df.index
        )
        
        return prob_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores (if model supports it).
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not support feature importances")
        
        importance_df = pd.DataFrame({
            'feature': self._feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
