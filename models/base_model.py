"""
Base model interface for all ML models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.is_fitted = False
        self.model = None
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **params):
        """Set model parameters."""
        pass
