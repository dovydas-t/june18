# metrics/metrics_manager.py - Comprehensive Metrics Management System
"""
Advanced metrics management system for comprehensive model evaluation.
Provides interactive metric selection, dynamic ranking, and statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, log_loss,
    matthews_corrcoef, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MetricsManager:
    """
    Comprehensive metrics management system with interactive configuration.
    """
    
    def __init__(self, ui):
        """Initialize metrics manager."""
        self.ui = ui
        self.available_metrics = self._define_available_metrics()
        self.active_metrics = self._get_default_active_metrics()
        self.primary_metric = 'Accuracy'  # Default primary metric
        self.metric_weights = {}
        self.metric_descriptions = self._get_metric_descriptions()
        self.model_metrics = {}  # Store calculated metrics for all models
        self.metric_history = {}  # Track metric changes over time
        
    def _define_available_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Define all available metrics with their configurations."""
        return {
            'classification': {
                'Accuracy': {
                    'function': accuracy_score,
                    'params': {},
                    'higher_better': True,
                    'range': (0, 1),
                    'default_active': True,
                    'category': 'Overall Performance'
                },
                'Precision': {
                    'function': precision_score,
                    'params': {'average': 'weighted', 'zero_division': 0},
                    'higher_better': True,
                    'range': (0, 1),
                    'default_active': True,
                    'category': 'Positive Prediction Quality'
                },
                'Recall': {
                    'function': recall_score,
                    'params': {'average': 'weighted', 'zero_division': 0},
                    'higher_better': True,
                    'range': (0, 1),
                    'default_active': True,
                    'category': 'Positive Detection Rate'
                },
                'F1_Score': {
                    'function': f1_score,
                    'params': {'average': 'weighted', 'zero_division': 0},
                    'higher_better': True,
                    'range': (0, 1),
                    'default_active': True,
                    'category': 'Balanced Performance'
                },
                'ROC_AUC': {
                    'function': self._calculate_roc_auc,
                    'params': {},
                    'higher_better': True,
                    'range': (0, 1),
                    'default_active': False,
                    'category': 'Probability Ranking'
                },
                'Log_Loss': {
                    'function': self._calculate_log_loss,
                    'params': {},
                    'higher_better': False,
                    'range': (0, float('inf')),
                    'default_active': False,
                    'category': 'Probability Quality'
                },
                'MCC': {
                    'function': matthews_corrcoef,
                    'params': {},
                    'higher_better': True,
                    'range': (-1, 1),
                    'default_active': False,
                    'category': 'Correlation-based'
                }
            },
            'regression': {
                'RMSE': {
                    'function': self._calculate_rmse,
                    'params': {},
                    'higher_better': False,
                    'range': (0, float('inf')),
                    'default_active': True,
                    'category': 'Error Magnitude'
                },
                'MAE': {
                    'function': mean_absolute_error,
                    'params': {},
                    'higher_better': False,
                    'range': (0, float('inf')),
                    'default_active': True,
                    'category': 'Error Magnitude'
                },
                'R²': {
                    'function': r2_score,
                    'params': {},
                    'higher_better': True,
                    'range': (float('-inf'), 1),
                    'default_active': True,
                    'category': 'Variance Explained'
                },
                'MAPE': {
                    'function': self._calculate_mape,
                    'params': {},
                    'higher_better': False,
                    'range': (0, float('inf')),
                    'default_active': False,
                    'category': 'Percentage Error'
                },
                'MSLE': {
                    'function': self._calculate_msle,
                    'params': {},
                    'higher_better': False,
                    'range': (0, float('inf')),
                    'default_active': False,
                    'category': 'Log Error'
                }
            }
        }
    
    def _get_default_active_metrics(self) -> Dict[str, List[str]]:
        """Get default active metrics for each problem type."""
        active = {}
        for problem_type, metrics in self.available_metrics.items():
            active[problem_type] = [
                name for name, config in metrics.items()
                if config['default_active']
            ]
        return active
    
    def _get_metric_descriptions(self) -> Dict[str, str]:
        """Get detailed descriptions for all metrics."""
        return {
            'Accuracy': 'Overall classification correctness (TP+TN)/(TP+TN+FP+FN)',
            'Precision': 'Of positive predictions, how many were correct (TP)/(TP+FP)',
            'Recall': 'Of actual positives, how many were found (TP)/(TP+FN)',
            'F1_Score': 'Harmonic mean of precision and recall',
            'ROC_AUC': 'Area under ROC curve - probability ranking quality',
            'Log_Loss': 'Logarithmic loss - penalizes confident wrong predictions',
            'MCC': 'Matthews Correlation Coefficient - balanced measure for all classes',
            'RMSE': 'Root Mean Square Error - sensitive to outliers',
            'MAE': 'Mean Absolute Error - robust to outliers',
            'R²': 'Coefficient of determination - proportion of variance explained',
            'MAPE': 'Mean Absolute Percentage Error - relative error measure',
            'MSLE': 'Mean Squared Logarithmic Error - for positive skewed targets'
        }
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, problem_type: str, 
                                      y_prob: Optional[np.ndarray] = None, 
                                      model_name: str = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            problem_type: 'classification' or 'regression'
            y_prob: Prediction probabilities (for classification)
            model_name: Name of the model
            
        Returns:
            Dictionary with calculated metrics
        """
        if problem_type not in self.available_metrics:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        metrics = {}
        active_metrics = self.get_active_metrics(problem_type)
        
        for metric_name in active_metrics:
            try:
                metric_config = self.available_metrics[problem_type][metric_name]
                metric_func = metric_config['function']
                
                # Calculate metric based on function requirements
                if metric_name in ['ROC_AUC', 'Log_Loss'] and y_prob is not None:
                    value = metric_func(y_true, y_prob)
                else:
                    value = metric_func(y_true, y_pred, **metric_config['params'])
                
                metrics[metric_name] = float(value)
                
            except Exception as e:
                self.ui.print(f"⚠️ Warning: Could not calculate {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        # Store metrics for this model
        if model_name:
            self.model_metrics[model_name] = metrics
            
        return metrics
    
    def _calculate_roc_auc(self, y_true, y_prob):
        """Calculate ROC-AUC with proper handling for multiclass."""
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) == 2:
                # Binary classification
                if y_prob.ndim == 2:
                    return roc_auc_score(y_true, y_prob[:, 1])
                else:
                    return roc_auc_score(y_true, y_prob)
            else:
                # Multiclass classification
                return roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except Exception:
            return 0.5  # Random performance baseline
    
    def _calculate_log_loss(self, y_true, y_prob):
        """Calculate log loss with proper handling."""
        try:
            return log_loss(y_true, y_prob)
        except Exception:
            return float('inf')
    
    def _calculate_rmse(self, y_true, y_pred):
        """Calculate RMSE."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _calculate_mape(self, y_true, y_pred):
        """Calculate MAPE with zero division handling."""
        mask = y_true != 0
        if mask.sum() > 0:
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return 0.0
    
    def _calculate_msle(self, y_true, y_pred):
        """Calculate MSLE with proper handling for negative values."""
        try:
            # Ensure positive values for log
            y_true_pos = np.maximum(y_true, 1e-10)
            y_pred_pos = np.maximum(y_pred, 1e-10)
            return np.mean((np.log1p(y_true_pos) - np.log1p(y_pred_pos)) ** 2)
        except Exception:
            return float('inf')
    
    def get_active_metrics(self, problem_type: str) -> List[str]:
        """Get list of currently active metrics for problem type."""
        return self.active_metrics.get(problem_type, [])
    
    def set_active_metrics(self, problem_type: str, metrics: List[str]):
        """Set active metrics for a problem type."""
        available = list(self.available_metrics.get(problem_type, {}).keys())
        valid_metrics = [m for m in metrics if m in available]
        self.active_metrics[problem_type] = valid_metrics
        
        if valid_metrics and self.primary_metric not in valid_metrics:
            self.primary_metric = valid_metrics[0]
    
    def set_primary_metric(self, metric_name: str, problem_type: str = None):
        """Set the primary metric for ranking models."""
        if problem_type:
            active = self.get_active_metrics(problem_type)
            if metric_name in active:
                self.primary_metric = metric_name
                return True
        else:
            # Check if metric is active in any problem type
            for pt in self.active_metrics:
                if metric_name in self.active_metrics[pt]:
                    self.primary_metric = metric_name
                    return True
        return False
    
    def get_primary_metric(self) -> str:
        """Get the current primary metric."""
        return self.primary_metric
    
    def rank_models(self, problem_type: str, metric: str = None) -> List[Tuple[str, Dict[str, float]]]:
        """
        Rank models by specified metric or primary metric.
        
        Args:
            problem_type: Type of problem
            metric: Metric to rank by (uses primary if None)
            
        Returns:
            List of (model_name, metrics) tuples sorted by performance
        """
        if not self.model_metrics:
            return []
        
        ranking_metric = metric or self.primary_metric
        
        # Check if metric is higher_better or lower_better
        is_higher_better = True
        if problem_type in self.available_metrics:
            metric_config = self.available_metrics[problem_type].get(ranking_metric, {})
            is_higher_better = metric_config.get('higher_better', True)
        
        # Sort models by metric
        model_items = list(self.model_metrics.items())
        sorted_models = sorted(
            model_items,
            key=lambda x: x[1].get(ranking_metric, -float('inf') if is_higher_better else float('inf')),
            reverse=is_higher_better
        )
        
        return sorted_models
    
    def show_interactive_metrics_config(self, problem_type: str):
        """Show interactive metrics configuration interface."""
        available = self.available_metrics.get(problem_type, {})
        if not available:
            self.ui.print(f"❌ No metrics available for {problem_type}")
            return
        
        while True:
            self._display_metrics_config_panel(problem_type)
            
            choice = self.ui.input("Action ([1-9] toggle, [P]rimary, [A]ll, [N]one, [D]one)", default="D").upper()
            
            if choice == "D":
                break
            elif choice == "A":
                self.set_active_metrics(problem_type, list(available.keys()))
            elif choice == "N":
                self.set_active_metrics(problem_type, [])
            elif choice == "P":
                self._configure_primary_metric(problem_type)
            elif choice.isdigit():
                metric_idx = int(choice) - 1
                metric_names = list(available.keys())
                if 0 <= metric_idx < len(metric_names):
                    self._toggle_metric(problem_type, metric_names[metric_idx])
    
    def _display_metrics_config_panel(self, problem_type: str):
        """Display the metrics configuration panel."""
        available = self.available_metrics[problem_type]
        active = self.get_active_metrics(problem_type)
        
        self.ui.print(f"\n🎯 METRICS CONFIGURATION PANEL ({problem_type.title()})")
        self.ui.print("=" * 50)
        self.ui.print(f"Current Primary: {self.primary_metric}")
        
        self.ui.print(f"\n📊 AVAILABLE METRICS:")
        
        if hasattr(self.ui, 'console') and self.ui.console:
            # Rich table display
            from rich.table import Table
            table = Table(title=f"{problem_type.title()} Metrics")
            table.add_column("ID", style="cyan", width=3)
            table.add_column("Metric", style="white", width=12)
            table.add_column("Status", style="green", width=8)
            table.add_column("Description", style="white", width=35)
            table.add_column("Range", style="yellow", width=15)
            
            for i, (name, config) in enumerate(available.items(), 1):
                status = "✅ Active" if name in active else "⬜ Hidden"
                if name == self.primary_metric:
                    status = "🎯 Primary"
                
                desc = self.metric_descriptions.get(name, "No description")
                range_info = f"{config['range'][0]:.1f} to {config['range'][1]:.1f}"
                if config['range'][1] == float('inf'):
                    range_info = f"{config['range'][0]:.1f}+"
                
                table.add_row(str(i), name, status, desc[:30] + "..." if len(desc) > 30 else desc, range_info)
            
            self.ui.console.print(table)
        else:
            # Basic text display
            for i, (name, config) in enumerate(available.items(), 1):
                status = "✅" if name in active else "⬜"
                if name == self.primary_metric:
                    status = "🎯"
                desc = self.metric_descriptions.get(name, "")
                self.ui.print(f"{i:2d}. {status} {name:<12} - {desc}")
        
        self.ui.print(f"\n🎛️ ACTIONS:")
        self.ui.print("[1-9] Toggle metric | [P]rimary | [A]ll [N]one | [D]one")
    
    def _toggle_metric(self, problem_type: str, metric_name: str):
        """Toggle a metric's active status."""
        active = self.get_active_metrics(problem_type)
        if metric_name in active:
            active.remove(metric_name)
            # If removing primary metric, set new primary
            if metric_name == self.primary_metric and active:
                self.primary_metric = active[0]
        else:
            active.append(metric_name)
        
        self.set_active_metrics(problem_type, active)
    
    def _configure_primary_metric(self, problem_type: str):
        """Configure primary metric interactively."""
        active = self.get_active_metrics(problem_type)
        if not active:
            self.ui.print("❌ No active metrics. Please activate metrics first.")
            return
        
        self.ui.print(f"\n🎯 SELECT PRIMARY METRIC:")
        for i, metric in enumerate(active, 1):
            marker = "🎯" if metric == self.primary_metric else "  "
            self.ui.print(f"{marker} {i}. {metric}")
        
        try:
            choice = int(self.ui.input(f"Select primary metric (1-{len(active)})", default="1"))
            if 1 <= choice <= len(active):
                self.primary_metric = active[choice - 1]
                self.ui.print(f"✅ Primary metric set to: {self.primary_metric}")
        except ValueError:
            self.ui.print("❌ Invalid selection")
    
    def show_metrics_comparison_table(self, problem_type: str, top_n: int = None):
        """Show comprehensive metrics comparison table."""
        if not self.model_metrics:
            self.ui.print("❌ No model metrics available")
            return
        
        ranked_models = self.rank_models(problem_type)
        if top_n:
            ranked_models = ranked_models[:top_n]
        
        active_metrics = self.get_active_metrics(problem_type)
        
        if hasattr(self.ui, 'console') and self.ui.console:
            # Rich table display
            from rich.table import Table
            table = Table(title=f"📊 Model Performance Comparison ({self.primary_metric} Ranking)")
            
            table.add_column("Rank", style="cyan", width=6)
            table.add_column("Model", style="white", width=18)
            
            for metric in active_metrics:
                style = "bold green" if metric == self.primary_metric else "white"
                table.add_column(metric, style=style, width=10)
            
            for rank, (model_name, metrics) in enumerate(ranked_models, 1):
                rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}"
                
                row_data = [rank_emoji, model_name]
                for metric in active_metrics:
                    value = metrics.get(metric, 0.0)
                    formatted = f"{value:.4f}" if abs(value) < 1000 else f"{value:.2e}"
                    row_data.append(formatted)
                
                table.add_row(*row_data)
            
            self.ui.console.print(table)
        else:
            # Basic text display
            self.ui.print(f"\n📊 MODEL PERFORMANCE COMPARISON (Ranked by {self.primary_metric})")
            self.ui.print("=" * 80)
            
            # Header
            header = f"{'Rank':<6} {'Model':<18}"
            for metric in active_metrics:
                header += f" {metric:<10}"
            self.ui.print(header)
            self.ui.print("-" * 80)
            
            # Data rows
            for rank, (model_name, metrics) in enumerate(ranked_models, 1):
                rank_str = f"{rank:2d}"
                if rank <= 3:
                    rank_str = ["🥇", "🥈", "🥉"][rank-1]
                
                row = f"{rank_str:<6} {model_name:<18}"
                for metric in active_metrics:
                    value = metrics.get(metric, 0.0)
                    formatted = f"{value:.4f}"
                    row += f" {formatted:<10}"
                self.ui.print(row)
    
    def show_metric_insights(self, problem_type: str):
        """Show insights and recommendations based on current metrics."""
        if not self.model_metrics:
            return
        
        ranked_models = self.rank_models(problem_type)
        if len(ranked_models) < 2:
            return
        
        best_model = ranked_models[0]
        second_model = ranked_models[1]
        
        self.ui.print(f"\n💡 METRIC INSIGHTS:")
        
        # Best model analysis
        primary_score = best_model[1].get(self.primary_metric, 0)
        self.ui.print(f"• {best_model[0]} leads with {self.primary_metric}: {primary_score:.4f}")
        
        # Performance gaps
        if len(ranked_models) > 1:
            gap = abs(primary_score - second_model[1].get(self.primary_metric, 0))
            if gap < 0.01:
                self.ui.print(f"• Very close competition with {second_model[0]} (gap: {gap:.4f})")
            elif gap > 0.1:
                self.ui.print(f"• Clear winner - significant gap of {gap:.4f}")
        
        # Metric-specific insights
        self._show_metric_specific_insights(problem_type, ranked_models)
    
    def _show_metric_specific_insights(self, problem_type: str, ranked_models: List[Tuple[str, Dict]]):
        """Show insights specific to different metrics."""
        if problem_type == 'classification':
            self._show_classification_insights(ranked_models)
        else:
            self._show_regression_insights(ranked_models)
    
    def _show_classification_insights(self, ranked_models: List[Tuple[str, Dict]]):
        """Show classification-specific insights."""
        if not ranked_models:
            return
        
        best_metrics = ranked_models[0][1]
        
        # Precision vs Recall analysis
        precision = best_metrics.get('Precision', 0)
        recall = best_metrics.get('Recall', 0)
        
        if abs(precision - recall) > 0.1:
            if precision > recall:
                self.ui.print("• High precision, lower recall - model is conservative")
            else:
                self.ui.print("• High recall, lower precision - model is aggressive")
        
        # F1-Score balance
        f1 = best_metrics.get('F1_Score', 0)
        if f1 > 0.9:
            self.ui.print("• Excellent F1-Score indicates balanced performance")
        elif f1 < 0.7:
            self.ui.print("• Consider improving precision/recall balance")
    
    def _show_regression_insights(self, ranked_models: List[Tuple[str, Dict]]):
        """Show regression-specific insights."""
        if not ranked_models:
            return
        
        best_metrics = ranked_models[0][1]
        
        # R² analysis
        r2 = best_metrics.get('R²', 0)
        if r2 > 0.8:
            self.ui.print("• Excellent R² indicates strong predictive power")
        elif r2 < 0.5:
            self.ui.print("• Low R² suggests poor model fit or complex patterns")
        
        # Error analysis
        rmse = best_metrics.get('RMSE', float('inf'))
        mae = best_metrics.get('MAE', float('inf'))
        
        if rmse != float('inf') and mae != float('inf'):
            ratio = rmse / mae if mae > 0 else float('inf')
            if ratio > 2:
                self.ui.print("• High RMSE/MAE ratio suggests outlier sensitivity")
    
    def export_metrics_summary(self, filename: str = "metrics_summary.csv") -> bool:
        """Export comprehensive metrics summary to CSV."""
        try:
            if not self.model_metrics:
                return False
            
            # Create summary DataFrame
            summary_data = []
            for model_name, metrics in self.model_metrics.items():
                row = {'Model': model_name}
                row.update(metrics)
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(filename, index=False)
            
            self.ui.print(f"✅ Metrics summary exported to {filename}")
            return True
            
        except Exception as e:
            self.ui.print(f"❌ Export failed: {e}")
            return False
    
    def get_metric_configuration(self) -> Dict[str, Any]:
        """Get current metrics configuration."""
        return {
            'active_metrics': self.active_metrics.copy(),
            'primary_metric': self.primary_metric,
            'metric_weights': self.metric_weights.copy()
        }
    
    def set_metric_configuration(self, config: Dict[str, Any]):
        """Set metrics configuration from saved config."""
        if 'active_metrics' in config:
            self.active_metrics = config['active_metrics']
        if 'primary_metric' in config:
            self.primary_metric = config['primary_metric']
        if 'metric_weights' in config:
            self.metric_weights = config['metric_weights']
    
    def clear_model_metrics(self):
        """Clear all stored model metrics."""
        self.model_metrics.clear()
        self.metric_history.clear()