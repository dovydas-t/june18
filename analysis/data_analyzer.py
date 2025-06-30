"""
Simplified data analysis module
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class SmartDataAnalyzer:
    """
    Simplified data analysis.
    """
    
    def __init__(self, ui):
        """Initialize data analyzer."""
        self.ui = ui
    
    def analyze_dataset_comprehensive(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Perform basic dataset analysis.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            analysis = {
                'basic_info': self._get_basic_info(df),
                'data_quality': self._assess_data_quality(df),
                'recommendations': self._generate_recommendations(df)
            }
            return analysis
        except Exception as e:
            self.ui.print(f"Warning: Analysis failed: {e}")
            return {}
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum(),
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        missing_total = df.isnull().sum().sum()
        missing_percentage = (missing_total / df.size) * 100
        
        quality_score = 100.0
        issues = []
        
        if missing_percentage > 20:
            quality_score -= 30
            issues.append(f"High missing data: {missing_percentage:.1f}%")
        elif missing_percentage > 5:
            quality_score -= 15
            issues.append(f"Moderate missing data: {missing_percentage:.1f}%")
        
        return {
            'quality_score': max(0, quality_score),
            'total_missing': int(missing_total),
            'missing_percentage': missing_percentage,
            'issues': issues
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> list:
        """Generate basic recommendations."""
        recommendations = []
        
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            recommendations.append("Handle missing values before training")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            recommendations.append("Encode categorical variables")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            recommendations.append("Consider feature scaling")
        
        return recommendations
