"""
Factory pattern for creating model instances.
"""

from typing import Optional, Any
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                             ExtraTreesClassifier, ExtraTreesRegressor,
                             BaggingClassifier, BaggingRegressor,
                             AdaBoostClassifier, AdaBoostRegressor,
                             HistGradientBoostingClassifier, HistGradientBoostingRegressor)


class ModelFactory:
    """Factory for creating ML model instances."""
    
    @staticmethod
    def create_model(model_name: str, problem_type: str, **kwargs) -> Optional[Any]:
        """
        Create a model instance based on name and problem type.
        
        Args:
            model_name: Name of the model
            problem_type: 'classification' or 'regression'
            **kwargs: Additional model parameters
            
        Returns:
            Model instance or None if not found
        """
        models = ModelFactory._get_model_mapping()
        
        model_key = f"{model_name}_{problem_type}"
        if model_key in models:
            model_class = models[model_key]
            return model_class(**kwargs)
        
        return None
    
    @staticmethod
    def _get_model_mapping() -> Dict[str, Any]:
        """Get mapping of model names to classes."""
        return {
            # KNN
            'KNN_classification': KNeighborsClassifier,
            'KNN_regression': KNeighborsRegressor,
            
            # SVM
            'SVM_classification': SVC,
            'SVM_regression': SVR,
            
            # Decision Tree
            'Decision Tree_classification': DecisionTreeClassifier,
            'Decision Tree_regression': DecisionTreeRegressor,
            
            # Linear Models
            'Linear Model_classification': LogisticRegression,
            'Linear Model_regression': LinearRegression,
            
            # Random Forest
            'Random Forest_classification': RandomForestClassifier,
            'Random Forest_regression': RandomForestRegressor,
            
            # Extra Trees
            'Extra Trees_classification': ExtraTreesClassifier,
            'Extra Trees_regression': ExtraTreesRegressor,
            
            # Bagging
            'Bagging_classification': BaggingClassifier,
            'Bagging_regression': BaggingRegressor,
            
            # AdaBoost
            'AdaBoost_classification': AdaBoostClassifier,
            'AdaBoost_regression': AdaBoostRegressor,
            
            # Histogram Gradient Boosting
            'Hist Gradient Boosting_classification': HistGradientBoostingClassifier,
            'Hist Gradient Boosting_regression': HistGradientBoostingRegressor,
        }