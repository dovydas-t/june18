"""
Validation utilities for data integrity and environment setup.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple
import warnings


def validate_environment() -> bool:
    """
    Validate that required packages are available.
    
    Returns:
        bool: True if environment is properly set up
    """
    required_packages = ['pandas', 'numpy', 'sklearn']
    missing_packages = []
    
    for package_name in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install pandas numpy scikit-learn")
        return False
    
    return True


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file path exists and is readable.
    
    Args:
        file_path: Path to file
        
    Returns:
        bool: True if file exists and is readable
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> bool:
    """
    Validate basic DataFrame properties.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        min_cols: Minimum number of columns required
        
    Returns:
        bool: True if DataFrame passes validation
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False
    
    if len(df) < min_rows or len(df.columns) < min_cols:
        return False
    
    return not df.empty


def validate_target_column(df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
    """
    Validate target column exists and has appropriate properties.
    
    Args:
        df: DataFrame containing the target
        target_column: Name of target column
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if target_column not in df.columns:
        return False, f"Target column '{target_column}' not found in DataFrame"
    
    target_series = df[target_column]
    
    if target_series.isnull().all():
        return False, "Target column contains only missing values"
    
    non_null_target = target_series.dropna()
    if len(non_null_target.unique()) == 1:
        return False, "Target column has no variation"
    
    return True, "Target column validation passed"


def validate_data_quality(df: pd.DataFrame, quality_thresholds: dict = None) -> dict:
    """
    Basic data quality validation.
    
    Args:
        df: DataFrame to validate
        quality_thresholds: Custom quality thresholds
        
    Returns:
        Dictionary with quality assessment results
    """
    results = {
        'overall_score': 100.0,
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Missing values assessment
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / df.size) * 100
    
    if missing_pct > 20:
        results['overall_score'] -= 30
        results['issues'].append(f"High missing data: {missing_pct:.1f}%")
    elif missing_pct > 5:
        results['overall_score'] -= 15
        results['warnings'].append(f"Moderate missing data: {missing_pct:.1f}%")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        duplicate_pct = (duplicates / len(df)) * 100
        if duplicate_pct > 10:
            results['issues'].append(f"High duplicate rows: {duplicate_pct:.1f}%")
    
    results['overall_score'] = max(0, results['overall_score'])
    return results
