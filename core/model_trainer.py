"""
Enhanced model training module with comprehensive metrics and interactive training
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import comprehensive metrics system
try:
    from metrics.metrics_manager import MetricsManager
    METRICS_MANAGER_AVAILABLE = True
except ImportError:
    METRICS_MANAGER_AVAILABLE = False
    print("Warning: Advanced metrics system not available")


class ModelTrainer:
    """
    Enhanced model training with comprehensive metrics and interactive controls.
    """
    
    def __init__(self, ui):
        """
        Initialize model trainer.
        
        Args:
            ui: Terminal UI instance
        """
        self.ui = ui
        self.trained_models = {}
        self.model_scores = {}
        self.training_times = {}
        self.cv_scores = {}
        self.model_metadata = {}
        self.training_interrupted = False

        
        # Initialize comprehensive metrics manager
        if METRICS_MANAGER_AVAILABLE:
            self.metrics_manager = MetricsManager(ui)
        else:
            self.metrics_manager = None
        
        # Training state
        self.current_training_session = {}
        self.training_interrupted = False

    def train_multiple_models(self, enabled_models: List[str], pipeline) -> Dict[str, Any]:
        """Train multiple models with enhanced error handling."""
        results = {}
        
        if not enabled_models:
            self.ui.print("❌ No models enabled for training")
            return results
        
        self.ui.print(f"\n🚀 Training {len(enabled_models)} models...")
        
        for i, model_name in enumerate(enabled_models, 1):
            self.ui.print(f"\n[{i}/{len(enabled_models)}] Training {model_name}...")
            
            try:
                result = self.train_model_with_comprehensive_metrics(model_name, pipeline)
                
                if 'error' not in result:
                    results[model_name] = result
                    self.ui.print(f"✅ {model_name} completed successfully")
                else:
                    self.ui.print(f"❌ {model_name} failed: {result['error']}")
                    
            except Exception as e:
                self.ui.print(f"❌ {model_name} crashed: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        successful_models = len([r for r in results.values() if 'error' not in r])
        self.ui.print(f"\n🎯 Training completed: {successful_models}/{len(enabled_models)} models successful")
        
        return results
    
    def generate_predictions(self, model_name: str, X_submission) -> np.ndarray:
        """Generate predictions using trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.trained_models[model_name]
        return model.predict(X_submission)
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model."""
        if not self.model_scores:
            return None, None
        
        # Determine primary metric based on available metrics
        first_model_metrics = list(self.model_scores.values())[0]
        
        if 'Accuracy' in first_model_metrics:
            primary_metric = 'Accuracy'
            higher_better = True
        elif 'R²' in first_model_metrics:
            primary_metric = 'R²'
            higher_better = True
        elif 'RMSE' in first_model_metrics:
            primary_metric = 'RMSE'
            higher_better = False
        else:
            primary_metric = list(first_model_metrics.keys())[0]
            higher_better = True
        
        # Find best model
        best_model_name = None
        best_score = -float('inf') if higher_better else float('inf')
        
        for model_name, scores in self.model_scores.items():
            score = scores.get(primary_metric, 0)
            
            if higher_better and score > best_score:
                best_score = score
                best_model_name = model_name
            elif not higher_better and score < best_score:
                best_score = score
                best_model_name = model_name
        
        best_model = self.trained_models.get(best_model_name) if best_model_name else None
        return best_model_name, best_model
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict]:
        """Get feature importance for models that support it."""
        if model_name not in self.trained_models:
            return None
        
        model = self.trained_models[model_name]
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                return {i: importance for i, importance in enumerate(importance_values)}
            elif hasattr(model, 'coef_'):
                importance_values = abs(model.coef_).flatten()
                return {i: importance for i, importance in enumerate(importance_values)}
            else:
                return None
        except Exception:
            return None
    
    def _get_scoring_metric(self, problem_type: str) -> str:
        """Get appropriate scoring metric for cross-validation."""
        if problem_type == 'classification':
            return 'accuracy'
        else:
            return 'neg_mean_squared_error'
    
    def _calculate_basic_metrics(self, y_true, y_pred, problem_type: str) -> Dict[str, float]:
        """Calculate basic metrics as fallback when MetricsManager not available."""
        metrics = {}
        
        try:
            if problem_type == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics['Accuracy'] = accuracy_score(y_true, y_pred)
                metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['F1_Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
            else:  # regression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics['MAE'] = mean_absolute_error(y_true, y_pred)
                metrics['R²'] = r2_score(y_true, y_pred)
                
        except Exception as e:
            self.ui.print(f"⚠️ Error calculating metrics: {e}")
            metrics['Score'] = 0.0
        
        return metrics
    
    def _get_primary_metric_name(self, problem_type: str) -> str:
        """Get primary metric name for display."""
        if problem_type == 'classification':
            return 'Accuracy'
        else:
            return 'R²'

    def get_model_instance(self, model_name: str, problem_type: str, random_state: int = 42):
        """
        Get model instance based on name and problem type.
        
        Args:
            model_name: Name of the model
            problem_type: 'classification' or 'regression'
            random_state: Random seed
            
        Returns:
            Model instance
        """
        models = {
            'classification': {
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'SVM': SVC(random_state=random_state, probability=True),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=10),
                'Linear Model': LogisticRegression(random_state=random_state, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
                'Extra Trees': ExtraTreesClassifier(random_state=random_state, n_estimators=100),
                'Bagging': BaggingClassifier(random_state=random_state, n_estimators=100),
                'AdaBoost': AdaBoostClassifier(random_state=random_state, n_estimators=100),
                'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=random_state)
            },
            'regression': {
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'SVM': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=random_state, max_depth=10),
                'Linear Model': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=random_state, n_estimators=100),
                'Extra Trees': ExtraTreesRegressor(random_state=random_state, n_estimators=100),
                'Bagging': BaggingRegressor(random_state=random_state, n_estimators=100),
                'AdaBoost': AdaBoostRegressor(random_state=random_state, n_estimators=100),
                'Hist Gradient Boosting': HistGradientBoostingRegressor(random_state=random_state)
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE and model_name == 'XGBoost':
            if problem_type == 'classification':
                models['classification']['XGBoost'] = xgb.XGBClassifier(
                    random_state=random_state, 
                    eval_metric='logloss',
                    verbosity=0
                )
            else:
                models['regression']['XGBoost'] = xgb.XGBRegressor(
                    random_state=random_state,
                    eval_metric='rmse',
                    verbosity=0
                )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE and model_name == 'LightGBM':
            if problem_type == 'classification':
                models['classification']['LightGBM'] = lgb.LGBMClassifier(
                    random_state=random_state,
                    verbosity=-1
                )
            else:
                models['regression']['LightGBM'] = lgb.LGBMRegressor(
                    random_state=random_state,
                    verbosity=-1
                )
        
        return models.get(problem_type, {}).get(model_name)
    
    def train_model_with_comprehensive_metrics(self, model_name: str, pipeline) -> Dict[str, Any]:
        """
        Train a single model with comprehensive metric evaluation.
        
        Args:
            model_name: Name of the model to train
            pipeline: Pipeline object with data and configuration
            
        Returns:
            Dictionary with training results including all metrics
        """
        start_time = time.time()
        
        try:
            # Get model instance
            model = self.get_model_instance(model_name, pipeline.problem_type, pipeline.random_state)
            
            if model is None:
                return {'error': f'Model {model_name} not available for {pipeline.problem_type}'}
            
            self.ui.print(f"🤖 Training {model_name}...")
            
            # Perform cross-validation
            scoring = self._get_scoring_metric(pipeline.problem_type)
            
            try:
                cv_scores = cross_val_score(
                    model, pipeline.X_train, pipeline.y_train,
                    cv=min(pipeline.cv_folds, 5),
                    scoring=scoring,
                    n_jobs=-1
                )
                self.cv_scores[model_name] = cv_scores.tolist()
            except Exception as cv_error:
                self.ui.print(f"⚠️ CV failed for {model_name}: {cv_error}")
                cv_scores = np.array([0.0])
            
            # Train final model
            try:
                model.fit(pipeline.X_train, pipeline.y_train)
                self.ui.print(f"  ✅ Model fitted successfully")
            except Exception as fit_error:
                self.ui.print(f"❌ Training failed for {model_name}: {fit_error}")
                return {'error': f'Training failed: {fit_error}'}
            
            # Make predictions
            try:
                val_predictions = model.predict(pipeline.X_test)
                
                # Get prediction probabilities for comprehensive metrics
                y_prob = None
                if hasattr(model, 'predict_proba') and pipeline.problem_type == 'classification':
                    try:
                        y_prob = model.predict_proba(pipeline.X_test)
                    except:
                        pass
                
                self.ui.print(f"  ✅ Predictions generated")
            except Exception as pred_error:
                self.ui.print(f"❌ Prediction failed for {model_name}: {pred_error}")
                return {'error': f'Prediction failed: {pred_error}'}
            
            # Calculate comprehensive metrics using MetricsManager
            if self.metrics_manager:
                metrics = self.metrics_manager.calculate_comprehensive_metrics(
                    pipeline.y_test, val_predictions, pipeline.problem_type, y_prob, model_name
                )
            else:
                # Fallback to basic metrics
                metrics = self._calculate_basic_metrics(
                    pipeline.y_test, val_predictions, pipeline.problem_type
                )
            
            # Add CV score to metrics
            metrics['CV_Score'] = cv_scores.mean()
            metrics['CV_Std'] = cv_scores.std()
            
            training_time = time.time() - start_time
            
            # Store results
            results = {
                'model': model,
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'val_predictions': val_predictions,
                'val_probabilities': y_prob,
                'training_time': training_time,
                'metrics': metrics
            }
            
            self.trained_models[model_name] = model
            self.model_scores[model_name] = metrics
            self.training_times[model_name] = training_time
            
            # Show primary metric with all active metrics
            if self.metrics_manager:
                primary_metric = self.metrics_manager.get_primary_metric()
                primary_score = metrics.get(primary_metric, 0)
                self.ui.print(f"  📊 {primary_metric}: {primary_score:.4f}")
                
                # Show additional active metrics
                active_metrics = self.metrics_manager.get_active_metrics(pipeline.problem_type)
                other_metrics = [m for m in active_metrics if m != primary_metric][:2]  # Show top 2 others
                for metric in other_metrics:
                    score = metrics.get(metric, 0)
                    self.ui.print(f"  📊 {metric}: {score:.4f}")
            else:
                # Fallback display
                primary_metric = self._get_primary_metric_name(pipeline.problem_type)
                primary_score = metrics.get(primary_metric, 0)
                self.ui.print(f"  📊 {primary_metric}: {primary_score:.4f}")
            
            return results
            
        except Exception as e:
            self.ui.print(f"❌ Training error for {model_name}: {str(e)}")
            return {'error': f'Training error: {str(e)}'}

    def _get_scoring_metric(self, problem_type: str) -> str:
        """Get appropriate scoring metric for cross-validation."""
        if problem_type == 'classification':
            return 'accuracy'
        else:
            return 'neg_mean_squared_error'

    def _calculate_basic_metrics(self, y_true, y_pred, problem_type: str) -> Dict[str, float]:
        """Calculate basic metrics as fallback."""
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        metrics = {}
        
        if problem_type == 'classification':
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        else:
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['R²'] = r2_score(y_true, y_pred)
        
        return metrics

    def _get_primary_metric_name(self, problem_type: str) -> str:
        """Get primary metric name based on problem type."""
        if problem_type == 'classification':
            return 'Accuracy'
        else:
            return 'RMSE'

    def train_multiple_models(self, model_names: List[str], pipeline) -> Dict[str, Any]:
        """Train multiple models and return results."""
        results = {}
        
        for model_name in model_names:
            if model_name in pipeline.available_models and pipeline.available_models[model_name]:
                result = self.train_model_with_comprehensive_metrics(model_name, pipeline)
                results[model_name] = result
        
        return results

    def get_best_model(self) -> Tuple[Optional[str], Optional[Any]]:
        """Get the best performing model."""
        if not self.model_scores:
            return None, None
        
        # Simple scoring based on first available metric
        best_model_name = max(self.model_scores.keys(), 
                            key=lambda x: list(self.model_scores[x].values())[0])
        best_model = self.trained_models.get(best_model_name)
        
        return best_model_name, best_model

    def generate_predictions(self, model_name: str, X_test) -> np.ndarray:
        """Generate predictions using specified model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.trained_models[model_name]
        return model.predict(X_test)

    def get_feature_importance(self, model_name: str) -> Optional[Dict[int, float]]:
        """Get feature importance if available."""
        if model_name not in self.trained_models:
            return None
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return {i: imp for i, imp in enumerate(importances)}
        
        return None