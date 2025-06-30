import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from config import *
from core.data_operations import DataOperations
from features.data_exploration.exploration_engine import ExplorationEngine
from managers.menu_manager import MenuManager
from preprocessing.data_splitter import DataSplitter
from preprocessing.missing_values_handler import MissingValuesHandler
from preprocessing.feature_scaler import FeatureScaler

# Import with error handling
try:
    from database.experiment_db import ExperimentDatabase
except ImportError:
    ExperimentDatabase = None

try:
    from ui.terminal_ui import EnhancedTerminalUI
except ImportError:
    class EnhancedTerminalUI:
        def print(self, *args, **kwargs): print(*args)
        def input(self, prompt, default=""): return input(f"{prompt} [{default}]: ") or default
        def confirm(self, prompt, default=True): return input(f"{prompt} (y/n): ").lower() in ['y', 'yes'] or default

try:
    from analysis.data_analyzer import SmartDataAnalyzer
except ImportError:
    class SmartDataAnalyzer:
        def __init__(self, ui): self.ui = ui
        def analyze_dataset_comprehensive(self, df, target): return {}

try:
    from core.preprocessor import DataPreprocessor
except ImportError:
    class DataPreprocessor:
        def __init__(self, ui): self.ui = ui
        def preprocess_data(self, pipeline): return False

try:
    from core.model_trainer import ModelTrainer
except ImportError:
    class ModelTrainer:
        def __init__(self, ui): self.ui = ui
        def train_multiple_models(self, models, pipeline): return {}

class EnhancedMLPipeline:
    """
    Enhanced ML Pipeline v3.0 - Simplified main orchestrator.
    Delegates functionality to specialized managers and components.
    """
    
    def __init__(self, problem_type='auto', target_column=None, remove_outliers=True, 
                 test_size=DEFAULT_TEST_SIZE, cv_folds=DEFAULT_CV_FOLDS, 
                 random_state=RANDOM_STATE, experiment_name=None, user_notes=None):
        """Initialize Enhanced ML Pipeline v3.0."""
        
        # Initialize UI and database
        self.ui = EnhancedTerminalUI()
        self.db = ExperimentDatabase() if ExperimentDatabase else None
        
        # Initialize specialized components
        self.data_ops = DataOperations(self.ui, self.db)
        self.exploration_engine = ExplorationEngine(self.ui)
        self.menu_manager = MenuManager(self)
        self.analyzer = SmartDataAnalyzer(self.ui)
        
        # Initialize preprocessing components
        self.data_splitter = DataSplitter(self.ui)
        self.missing_handler = MissingValuesHandler(self.ui)
        self.feature_scaler = FeatureScaler(self.ui)
        self.preprocessor = DataPreprocessor(self.ui)
        self.model_trainer = ModelTrainer(self.ui)
        
        # Configuration parameters
        self.problem_type = problem_type
        self.target_column = target_column
        self.remove_outliers = remove_outliers
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.experiment_name = experiment_name or f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_notes = user_notes
        
        # Data containers
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_submission = None
        
        # Model configuration
        self.available_models = DEFAULT_MODELS.copy()
        self.models = {}
        self.predictions = {}
        self.scores = {}
        
        # Experiment tracking
        self.experiment_id = None
        self.experiment_start_time = None
        self.dataset_hash = None
        self.preprocessing_steps = []
        
        # Dataset info
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.is_binary_classification = False
        self.dataset_analysis = None
        
        # Set random seed
        np.random.seed(self.random_state)
 
    def main_menu(self):
        """Enhanced main menu with v3.1 features."""
        while True:
            self.menu_manager.show_main_menu_options()
            
            try:
                choice = int(self.ui.input("\nEnter your choice (1-11)", default="11"))
                
                if choice == 1:
                    self.menu_manager.handle_data_loading_menu()
                elif choice == 2:
                    self.menu_manager.handle_exploration_menu()
                elif choice == 3:
                    self.menu_manager.handle_preprocessing_menu()
                elif choice == 4:
                    self.menu_manager.handle_model_management_menu()
                elif choice == 5:
                    self._run_pipeline_menu()
                elif choice == 6:
                    self._view_results_menu()
                elif choice == 7:
                    self._generate_submission_menu()
                elif choice == 8:
                    self._help_menu()
                elif choice == 9:
                    self._view_database_menu()
                elif choice == 10:
                    self._show_about_menu()
                elif choice == 11:
                    self.ui.print(f"\n👋 Thank you for using Enhanced ML Pipeline v3.1!")
                    self.ui.print("🎯 Your experiments have been saved to the database.")
                    self.ui.print("✨ New in v3.1: Smart data tools, enhanced exploration, and distribution fixing!")
                    break
                else:
                    self.ui.print("❌ Invalid choice. Please try again.")
                    input("Press Enter to continue...")
            except (ValueError, KeyboardInterrupt):
                self.ui.print("❌ Please enter a valid number.")
                input("Press Enter to continue...")
    # Simplified methods - complex logic moved to managers
    def load_data(self, train_path: str, test_path: str = None, target_column: str = None) -> bool:
        """Load data using data operations manager."""
        return self.data_ops.load_data(train_path, test_path, target_column, self)

    def _run_pipeline_menu(self):
        """Run complete ML pipeline."""
        if self.X_train is None:
            self.ui.print("❌ No training data. Please load and preprocess data first.")
            input("Press Enter to continue...")
            return
        
        enabled_models = [name for name, enabled in self.available_models.items() if enabled]
        
        if not enabled_models:
            self.ui.print("❌ No models enabled. Please enable models first.")
            input("Press Enter to continue...")
            return
        
        self.ui.print(f"\n🚀 RUNNING PIPELINE")
        self.ui.print("="*30)
        self.ui.print(f"Training {len(enabled_models)} models...")
        self.ui.print(f"Training samples: {len(self.X_train):,}")
        self.ui.print(f"Validation samples: {len(self.X_test):,}")
        self.ui.print(f"Features: {self.X_train.shape[1]}")
        
        # Train models
        results = self.model_trainer.train_multiple_models(enabled_models, self)
        
        # Store results
        self.models = {}
        self.scores = {}
        
        for name, result in results.items():
            if 'error' not in result:
                self.models[name] = result.get('model')
                metrics = result.get('metrics', {})
                if 'training_time' in result:
                    metrics['Training_Time'] = result['training_time']
                self.scores[name] = metrics
        
        if self.scores:
            self._show_enhanced_results_summary()
        
        input("\nPress Enter to continue...")

    def _view_results_menu(self):
        """View results menu - simplified."""
        if not self.scores:
            self.ui.print("❌ No results available. Train models first.")
            input("Press Enter to continue...")
            return
        
        self.ui.print("\n📊 RESULTS & ANALYSIS")
        self.ui.print("="*30)
        self.ui.print("1. Performance summary")
        self.ui.print("2. Best model analysis")
        self.ui.print("0. Go Back")
        
        choice = self.ui.input("Enter choice (0-2)", default="0")
        
        if choice == "1":
            self._show_enhanced_results_summary()
        elif choice == "2":
            self._show_best_model_analysis()
        
        input("\nPress Enter to continue...")

    def _generate_submission_menu(self):
        """Generate submission file - simplified."""
        if not self.models or self.X_submission is None:
            self.ui.print("❌ Need trained models and test data for submission.")
            input("Press Enter to continue...")
            return
        
        # Get best model
        sorted_models = self._sort_models_by_performance()
        if not sorted_models:
            self.ui.print("❌ No trained models available.")
            input("Press Enter to continue...")
            return
        
        best_model_name, best_scores = sorted_models[0]
        
        self.ui.print("\n💾 GENERATE SUBMISSION")
        self.ui.print("="*30)
        self.ui.print(f"Using best model: {best_model_name}")
        
        primary_metric = self._get_primary_metric()
        primary_score = best_scores.get(primary_metric, 0)
        self.ui.print(f"{primary_metric}: {primary_score:.4f}")
        
        if self.ui.confirm("Generate submission file?", default=True):
            self._generate_submission_file(best_model_name)

    def _help_menu(self):
        """Help menu - simplified."""
        self.ui.print("\n📚 HELP & DOCUMENTATION")
        self.ui.print("="*30)
        self.ui.print("1. Getting started guide")
        self.ui.print("2. Model explanations")
        self.ui.print("3. Troubleshooting")
        self.ui.print("0. Go Back")
        
        choice = self.ui.input("Enter choice (0-3)", default="0")
        
        if choice == "1":
            self._show_getting_started()
        elif choice == "2":
            self._show_model_explanations()
        elif choice == "3":
            self._show_troubleshooting()
        
        input("\nPress Enter to continue...")

    def _view_database_menu(self):
        """Database menu - simplified."""
        if not self.db:
            self.ui.print("❌ Database not available.")
            input("Press Enter to continue...")
            return
        
        self.ui.print("\n🗄️ EXPERIMENT DATABASE")
        self.ui.print("="*30)
        
        stats = self.db.get_database_stats()
        self.ui.print(f"Total experiments: {stats.get('total_experiments', 0)}")
        self.ui.print(f"Total model results: {stats.get('total_model_results', 0)}")
        
        self.ui.print("\n1. View recent experiments")
        self.ui.print("0. Go Back")
        
        choice = self.ui.input("Enter choice (0-1)", default="0")
        
        if choice == "1":
            self._show_recent_experiments()
        
        input("\nPress Enter to continue...")

    def _show_about_menu(self):
        """Show about information - simplified."""
        self.ui.print("\n" + "="*60)
        self.ui.print("ℹ️ ENHANCED ML PIPELINE v3.0 - ABOUT")
        self.ui.print("="*60)
        
        self.ui.print("🚀 ENHANCED MACHINE LEARNING PIPELINE v3.0")
        self.ui.print("📅 Release Date: 2024")
        self.ui.print("👥 Authors: ML Pipeline Framework Team")
        self.ui.print("📄 License: MIT License")
        
        self.ui.print(f"\n🔥 NEW IN VERSION 3.0:")
        self.ui.print("• 🏗️ Completely refactored modular architecture")
        self.ui.print("• 📦 Separated concerns into specialized managers")
        self.ui.print("• 🔧 Enhanced preprocessing pipeline")
        self.ui.print("• 📊 Improved data exploration engine")
        self.ui.print("• 🎛️ Better menu system organization")
        self.ui.print("• 📈 Advanced metrics management")
        
        input("\nPress Enter to continue...")

    # Utility methods
    def _get_primary_metric(self) -> str:
        """Get the primary metric for comparison."""
        if not self.scores:
            return "Score"
        
        first_model_metrics = list(self.scores.values())[0]
        
        if self.problem_type == 'classification':
            return 'Accuracy' if 'Accuracy' in first_model_metrics else list(first_model_metrics.keys())[0]
        else:
            return 'R²' if 'R²' in first_model_metrics else 'RMSE' if 'RMSE' in first_model_metrics else list(first_model_metrics.keys())[0]

    def _sort_models_by_performance(self):
        """Sort models by performance based on primary metric."""
        primary_metric = self._get_primary_metric()
        reverse_sort = primary_metric != 'RMSE'
        
        return sorted(
            self.scores.items(),
            key=lambda x: x[1].get(primary_metric, -float('inf') if reverse_sort else float('inf')),
            reverse=reverse_sort
        )

    def _show_enhanced_results_summary(self):
        """Show enhanced results summary."""
        self.ui.print("\n📊 RESULTS SUMMARY")
        self.ui.print("="*50)
        
        primary_metric = self._get_primary_metric()
        self.ui.print(f"Primary Metric: {primary_metric}")
        
        sorted_models = self._sort_models_by_performance()
        
        self.ui.print(f"\n{'Rank':<4} {'Model':<20} {primary_metric:<10} {'Time (s)':<10}")
        self.ui.print("-" * 50)
        
        for rank, (model_name, scores) in enumerate(sorted_models, 1):
            primary_score = scores.get(primary_metric, 0)
            training_time = scores.get('Training_Time', 0)
            
            rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            
            self.ui.print(f"{rank_emoji}{rank:<3} {model_name:<20} {primary_score:<10.4f} {training_time:<10.2f}")
        
        if sorted_models:
            best_model, best_scores = sorted_models[0]
            self.ui.print(f"\n🏆 Best Model: {best_model}")

    def _show_best_model_analysis(self):
        """Show analysis of the best performing model."""
        sorted_models = self._sort_models_by_performance()
        if not sorted_models:
            self.ui.print("❌ No model results available.")
            return
        
        best_model_name, best_scores = sorted_models[0]
        
        self.ui.print(f"\n🏆 BEST MODEL ANALYSIS: {best_model_name}")
        self.ui.print("="*50)
        
        for metric, value in best_scores.items():
            if metric != 'Training_Time':
                self.ui.print(f"  {metric}: {value:.4f}")

    def _generate_submission_file(self, model_name: str):
        """Generate submission file with the specified model."""
        try:
            predictions = self.model_trainer.generate_predictions(model_name, self.X_submission)
            
            submission = pd.DataFrame()
            
            if hasattr(self, 'test_data') and self.test_data is not None:
                if 'Id' in self.test_data.columns:
                    submission['Id'] = self.test_data['Id']
                elif 'id' in self.test_data.columns:
                    submission['Id'] = self.test_data['id']
                else:
                    submission['Id'] = range(len(predictions))
            else:
                submission['Id'] = range(len(predictions))
            
            submission[self.target_column] = predictions
            
            filename = self.ui.input("Enter submission filename", default="submission.csv")
            submission.to_csv(filename, index=False)
            
            self.ui.print(f"✅ Submission saved to {filename}")
            self.ui.print(f"Shape: {submission.shape}")
            self.ui.print(f"Model used: {model_name}")
            
        except Exception as e:
            self.ui.print(f"❌ Error generating submission: {str(e)}")

    def _show_getting_started(self):
        """Show getting started guide."""
        self.ui.print("\n🚀 GETTING STARTED GUIDE")
        self.ui.print("="*30)
        self.ui.print("1. Load Data: Start by loading your training dataset")
        self.ui.print("2. Explore: Analyze data quality and characteristics")
        self.ui.print("3. Preprocess: Clean and prepare your data")
        self.ui.print("4. Configure Models: Select which algorithms to use")
        self.ui.print("5. Train: Run the complete ML pipeline")
        self.ui.print("6. Evaluate: Compare model performance")
        self.ui.print("7. Submit: Generate predictions for test data")

    def _show_model_explanations(self):
        """Show model explanations."""
        self.ui.print("\n🤖 MODEL EXPLANATIONS")
        self.ui.print("="*30)
        
        explanations = {
            "Random Forest": "Ensemble of decision trees, reduces overfitting",
            "XGBoost": "Gradient boosting, excellent for structured data",
            "Decision Tree": "Tree-like model, highly interpretable",
            "KNN": "Instance-based learning, uses nearest neighbors",
            "Linear Model": "Linear relationships, fast baseline model"
        }
        
        for model, explanation in explanations.items():
            self.ui.print(f"\n• {model}:")
            self.ui.print(f"  {explanation}")

    def _show_troubleshooting(self):
        """Show troubleshooting guide."""
        self.ui.print("\n🔧 TROUBLESHOOTING")
        self.ui.print("="*30)
        self.ui.print("Common Issues:")
        self.ui.print("• File not found: Check file path and extension")
        self.ui.print("• Target column missing: Verify column name spelling")
        self.ui.print("• Poor performance: Try different models or preprocessing")
        self.ui.print("• Memory errors: Reduce dataset size or model complexity")
        self.ui.print("• Import errors: Install missing packages with pip")

    def _show_recent_experiments(self):
        """Show recent experiments from database."""
        experiments = self.db.get_experiment_history(10)
        
        if not experiments:
            self.ui.print("📋 No experiments found in database.")
            return
        
        self.ui.print("\n📋 RECENT EXPERIMENTS")
        self.ui.print("="*50)
        
        for i, exp in enumerate(experiments, 1):
            self.ui.print(f"\n{i}. {exp['experiment_name']}")
            self.ui.print(f"   Dataset: {exp['dataset_name']}")
            self.ui.print(f"   Type: {exp['problem_type']}")
            self.ui.print(f"   Samples: {exp.get('n_samples', 'Unknown')}")
            self.ui.print(f"   Created: {exp['created_at']}")