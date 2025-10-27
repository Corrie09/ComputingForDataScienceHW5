"""
Diabetes prediction library for data loading, preprocessing, feature engineering, and modeling.

This library provides a complete pipeline for diabetes mellitus prediction including:
- Data loading and train/test splitting
- Data preprocessing (NaN handling)
- Feature engineering (BMI, gender encoding, age transformations)
- Model training and prediction
"""

# Import from data module
from .data import DataLoader, load_split, DEFAULT_TARGET

# Import from preprocessing module
from .preprocess import NaNRowRemover, NaNMeanFiller

# Import from features module
from .features import BaseFeature, BMICalculator, GenderEncoder, AgeSquared

# Import from models module
from .model import DiabetesModel  # ← CHANGED: .models → .model

# Define what gets exported with "from your_library import *"
__all__ = [
    # Data loading
    'DataLoader',
    'load_split',
    'DEFAULT_TARGET',
    
    # Preprocessing
    'NaNRowRemover',
    'NaNMeanFiller',
    
    # Features
    'BaseFeature',
    'BMICalculator',
    'GenderEncoder',
    'AgeSquared',
    
    # Models
    'DiabetesModel',
]

# Version info (optional but nice to have)
__version__ = '0.1.0'
__author__ = 'SF_TS_CM'