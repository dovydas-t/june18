"""
Database schema models for experiment tracking.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class ExperimentModel:
    """Model for experiment records."""
    
    def __init__(self, experiment_id: int = None, experiment_name: str = None,
                 dataset_name: str = None, dataset_hash: str = None,
                 problem_type: str = None, target_column: str = None,
                 n_samples: int = None, n_features: int = None,
                 train_size: float = None, test_size: float = None,
                 cv_folds: int = None, preprocessing_steps: List[str] = None,
                 user_notes: str = None, created_at: datetime = None,
                 updated_at: datetime = None):
        
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.dataset_hash = dataset_hash
        self.problem_type = problem_type
        self.target_column = target_column
        self.n_samples = n_samples
        self.n_features = n_features
        self.train_size = train_size
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.preprocessing_steps = preprocessing_steps or []
        self.user_notes = user_notes
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'dataset_name': self.dataset_name,
            'dataset_hash': self.dataset_hash,
            'problem_type': self.problem_type,
            'target_column': self.target_column,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'cv_folds': self.cv_folds,
            'preprocessing_steps': json.dumps(self.preprocessing_steps),
            'user_notes': self.user_notes,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class ModelResultModel:
    """Model for model training results."""
    
    def __init__(self, result_id: int = None, experiment_id: int = None,
                 model_name: str = None, model_type: str = None,
                 hyperparameters: Dict[str, Any] = None,
                 cv_scores: List[float] = None, mean_cv_score: float = None,
                 std_cv_score: float = None, test_score: float = None,
                 training_time: float = None, prediction_time: float = None,
                 feature_importance: Dict[str, float] = None,
                 confusion_matrix: List[List[int]] = None,
                 classification_report: Dict[str, Any] = None,
                 created_at: datetime = None):
        
        self.result_id = result_id
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.cv_scores = cv_scores or []
        self.mean_cv_score = mean_cv_score
        self.std_cv_score = std_cv_score
        self.test_score = test_score
        self.training_time = training_time
        self.prediction_time = prediction_time
        self.feature_importance = feature_importance or {}
        self.confusion_matrix = confusion_matrix or []
        self.classification_report = classification_report or {}
        self.created_at = created_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'result_id': self.result_id,
            'experiment_id': self.experiment_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'hyperparameters': json.dumps(self.hyperparameters),
            'cv_scores': json.dumps(self.cv_scores),
            'mean_cv_score': self.mean_cv_score,
            'std_cv_score': self.std_cv_score,
            'test_score': self.test_score,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'feature_importance': json.dumps(self.feature_importance),
            'confusion_matrix': json.dumps(self.confusion_matrix),
            'classification_report': json.dumps(self.classification_report),
            'created_at': self.created_at
        }


class MetricModel:
    """Model for performance metrics."""
    
    def __init__(self, metric_id: int = None, result_id: int = None,
                 metric_name: str = None, metric_value: float = None,
                 metric_type: str = None, created_at: datetime = None):
        
        self.metric_id = metric_id
        self.result_id = result_id
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.metric_type = metric_type
        self.created_at = created_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_id': self.metric_id,
            'result_id': self.result_id,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_type': self.metric_type,
            'created_at': self.created_at
        }