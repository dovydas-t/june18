"""
Evaluation metrics utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, Optional, Union


class MetricsCalculator:
    """Calculate various performance metrics."""
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                pass
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}
        
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(y_true, y_pred) -> Dict[str, Any]:
        """Get detailed classification report."""
        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)