import pandas as pd
import numpy as np

class OutlierDetector:
    """Handles outlier detection and removal with various methods."""
    
    def __init__(self, ui):
        self.ui = ui
    
    def remove_outliers(self, pipeline):
        """Enhanced outlier removal with user configuration."""
        if not hasattr(pipeline, 'outlier_method') or not pipeline.outlier_method:
            return
        
        method = pipeline.outlier_method
        threshold = getattr(pipeline, 'outlier_threshold', 1.5 if method == 'iqr' else 3.0)
        
        self.ui.print(f"🎯 Removing outliers using {method} method (threshold: {threshold})")
        
        numerical_cols = pipeline.X_train.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.ui.print("⚠️ No numerical columns for outlier detection")
            return
        
        outlier_mask = np.zeros(len(pipeline.X_train), dtype=bool)
        
        for col in numerical_cols:
            if method == 'iqr':
                col_outliers = self._detect_iqr_outliers(pipeline.X_train[col], threshold)
            elif method == 'zscore':
                col_outliers = self._detect_zscore_outliers(pipeline.X_train[col], threshold)
            else:
                continue
            
            outlier_mask |= col_outliers
        
        outliers_count = outlier_mask.sum()
        if outliers_count > 0:
            # Remove outliers from training data
            pipeline.X_train = pipeline.X_train[~outlier_mask].reset_index(drop=True)
            pipeline.y_train = pipeline.y_train[~outlier_mask].reset_index(drop=True)
            
            self.ui.print(f"✅ Removed {outliers_count} outlier samples")
        else:
            self.ui.print("✅ No outliers detected")
    
    def _detect_iqr_outliers(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_zscore_outliers(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
