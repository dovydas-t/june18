# ui/menus.py - Menu Systems
"""
Menu systems for different parts of the ML pipeline.
"""

from typing import List, Dict, Any


class BaseMenu:
    """Base class for all menu systems."""
    
    def __init__(self, pipeline):
        """
        Initialize base menu.
        
        Args:
            pipeline: Main pipeline object
        """
        self.pipeline = pipeline
        self.ui = pipeline.ui
    
    def show(self):
        """Show the menu - to be implemented by subclasses."""
        raise NotImplementedError


class DataExplorationMenu(BaseMenu):
    """Menu for data exploration functionality."""
    
    def show(self):
        """Show data exploration menu."""
        if self.pipeline.train_data is None:
            self.ui.print("[red]❌ No data loaded. Please load data first.[/red]")
            input("Press Enter to continue...")
            return
        
        while True:
            self.ui.show_header("🔍 Data Exploration", "Analyze your dataset comprehensively")
            
            options = [
                "📊 Dataset Overview",
                "📈 Column Analysis", 
                "🔍 Data Quality Assessment",
                "📋 Missing Values Report",
                "🎯 Target Variable Analysis",
                "📊 Correlation Analysis",
                "🔙 Back to Main Menu"
            ]
            
            self.ui.show_menu("Select exploration option:", options)
            
            choice = self.ui.input("Enter your choice (1-7)", default="7")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self._show_dataset_overview()
                elif choice == 2:
                    self._show_column_analysis()
                elif choice == 3:
                    self._show_data_quality()
                elif choice == 4:
                    self._show_missing_values()
                elif choice == 5:
                    self._show_target_analysis()
                elif choice == 6:
                    self._show_correlation_analysis()
                elif choice == 7:
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number.[/red]")
            
            if choice != 7:
                input("\nPress Enter to continue...")
    
    def _show_dataset_overview(self):
        """Show basic dataset overview."""
        df = self.pipeline.train_data
        
        self.ui.print("[bold blue]📊 Dataset Overview[/bold blue]")
        self.ui.print(f"Shape: {df.shape}")
        self.ui.print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show first few rows
        self.ui.show_data_table(df.head(), title="First 5 rows")
        
        # Show data types
        dtype_info = df.dtypes.value_counts()
        self.ui.print(f"\nData types: {dict(dtype_info)}")
    
    def _show_column_analysis(self):
        """Show detailed column analysis."""
        df = self.pipeline.train_data
        
        # Let user select a column to analyze
        columns = list(df.columns)
        self.ui.print(f"Available columns: {', '.join(columns)}")
        
        col_name = self.ui.input("Enter column name to analyze")
        
        if col_name not in columns:
            self.ui.print(f"[red]❌ Column '{col_name}' not found[/red]")
            return
        
        # Analyze the column
        analysis = self.pipeline.analyzer.analyze_single_column(df, col_name)
        
        self.ui.print(f"\n[bold blue]📈 Analysis for '{col_name}'[/bold blue]")
        self.ui.print(f"Data type: {analysis['dtype']}")
        self.ui.print(f"Missing values: {analysis['missing_count']} ({analysis['missing_percentage']:.1f}%)")
        self.ui.print(f"Unique values: {analysis['unique_count']} ({analysis['unique_percentage']:.1f}%)")
        
        if 'statistics' in analysis:
            stats = analysis['statistics']
            self.ui.print(f"\nStatistics:")
            for key, value in stats.items():
                self.ui.print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        if analysis['recommendations']:
            self.ui.print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                self.ui.print(f"  • {rec}")
    
    def _show_data_quality(self):
        """Show data quality assessment."""
        if hasattr(self.pipeline, 'dataset_analysis') and self.pipeline.dataset_analysis:
            quality = self.pipeline.dataset_analysis['data_quality']
            
            self.ui.print(f"[bold blue]🔍 Data Quality Assessment[/bold blue]")
            self.ui.print(f"Overall Quality Score: {quality['quality_score']:.1f}/100")
            
            if quality['issues']:
                self.ui.print(f"\n❌ Issues found:")
                for issue in quality['issues']:
                    self.ui.print(f"  • {issue}")
            
            if 'recommendations' in quality:
                self.ui.print(f"\n💡 Recommendations:")
                for rec in quality['recommendations']:
                    self.ui.print(f"  • {rec}")
        else:
            self.ui.print("[yellow]⚠️ No quality analysis available. Run dataset analysis first.[/yellow]")
    
    def _show_missing_values(self):
        """Show missing values report."""
        df = self.pipeline.train_data
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            self.ui.print("[green]✅ No missing values found![/green]")
            return
        
        self.ui.print(f"[bold blue]📋 Missing Values Report[/bold blue]")
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            self.ui.print(f"{col}: {count} ({pct:.1f}%)")
    
    def _show_target_analysis(self):
        """Show target variable analysis."""
        if not self.pipeline.target_column:
            self.ui.print("[red]❌ No target column specified[/red]")
            return
        
        target = self.pipeline.train_data[self.pipeline.target_column]
        
        self.ui.print(f"[bold blue]🎯 Target Variable Analysis: {self.pipeline.target_column}[/bold blue]")
        self.ui.print(f"Problem type: {self.pipeline.problem_type}")
        self.ui.print(f"Missing values: {target.isnull().sum()}")
        self.ui.print(f"Unique values: {target.nunique()}")
        
        if self.pipeline.problem_type == 'classification':
            value_counts = target.value_counts()
            self.ui.print(f"\nClass distribution:")
            for cls, count in value_counts.items():
                pct = (count / len(target)) * 100
                self.ui.print(f"  {cls}: {count} ({pct:.1f}%)")
        else:
            self.ui.print(f"\nStatistics:")
            self.ui.print(f"  Mean: {target.mean():.4f}")
            self.ui.print(f"  Median: {target.median():.4f}")
            self.ui.print(f"  Std: {target.std():.4f}")
            self.ui.print(f"  Min: {target.min():.4f}")
            self.ui.print(f"  Max: {target.max():.4f}")
    
    def _show_correlation_analysis(self):
        """Show correlation analysis."""
        df = self.pipeline.train_data
        numerical_cols = df.select_dtypes(include=['number']).columns
        
        if len(numerical_cols) < 2:
            self.ui.print("[yellow]⚠️ Need at least 2 numerical columns for correlation analysis[/yellow]")
            return
        
        # Calculate correlations with target if available
        if self.pipeline.target_column and self.pipeline.target_column in numerical_cols:
            target_corr = df[numerical_cols].corr()[self.pipeline.target_column].sort_values(ascending=False)
            
            self.ui.print(f"[bold blue]📊 Correlations with target ({self.pipeline.target_column})[/bold blue]")
            for col, corr in target_corr.items():
                if col != self.pipeline.target_column:
                    self.ui.print(f"{col}: {corr:.4f}")


class PreprocessingMenu(BaseMenu):
    """Menu for preprocessing configuration."""
    
    def show(self):
        """Show preprocessing menu."""
        if self.pipeline.train_data is None:
            self.ui.print("[red]❌ No data loaded. Please load data first.[/red]")
            input("Press Enter to continue...")
            return
        
        while True:
            self.ui.show_header("🔧 Preprocessing Configuration", "Configure data preprocessing steps")
            
            options = [
                "🔧 Run Preprocessing Pipeline",
                "📊 View Current Configuration",
                "🎯 Remove Outliers",
                "📋 Preprocessing Help",
                "🔙 Back to Main Menu"
            ]
            
            self.ui.show_menu("Select preprocessing option:", options)
            
            choice = self.ui.input("Enter your choice (1-5)", default="5")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self._run_preprocessing()
                elif choice == 2:
                    self._show_current_config()
                elif choice == 3:
                    self._remove_outliers()
                elif choice == 4:
                    self._show_preprocessing_help()
                elif choice == 5:
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number.[/red]")
            
            if choice != 5:
                input("\nPress Enter to continue...")
    
    def _run_preprocessing(self):
        """Run the preprocessing pipeline."""
        self.ui.print("[blue]🔧 Running preprocessing pipeline...[/blue]")
        
        success = self.pipeline.preprocessor.preprocess_data(self.pipeline)
        
        if success:
            self.ui.print("[green]✅ Preprocessing completed successfully![/green]")
            self.ui.print(f"Training data shape: {self.pipeline.X_train.shape}")
            self.ui.print(f"Validation data shape: {self.pipeline.X_test.shape}")
        else:
            self.ui.print("[red]❌ Preprocessing failed![/red]")
    
    def _show_current_config(self):
        """Show current preprocessing configuration."""
        self.ui.print("[bold blue]📊 Current Preprocessing Configuration[/bold blue]")
        self.ui.print(f"Test size: {self.pipeline.test_size}")
        self.ui.print(f"CV folds: {self.pipeline.cv_folds}")
        self.ui.print(f"Remove outliers: {self.pipeline.remove_outliers}")
        self.ui.print(f"Random state: {self.pipeline.random_state}")
        
        if self.pipeline.preprocessing_steps:
            self.ui.print(f"\nCompleted steps:")
            for step in self.pipeline.preprocessing_steps:
                self.ui.print(f"  ✅ {step}")
    
    def _remove_outliers(self):
        """Remove outliers from training data."""
        if self.pipeline.X_train is None:
            self.ui.print("[red]❌ No preprocessed data available. Run preprocessing first.[/red]")
            return
        
        method = self.ui.input("Outlier detection method (iqr/zscore)", default="iqr")
        threshold = float(self.ui.input("Threshold (1.5 for IQR, 3.0 for Z-score)", default="1.5" if method == "iqr" else "3.0"))
        
        outliers_removed = self.pipeline.preprocessor.remove_outliers(self.pipeline, method, threshold)
        
        if outliers_removed > 0:
            self.ui.print(f"[green]✅ Removed {outliers_removed} outliers[/green]")
        else:
            self.ui.print("[blue]ℹ️ No outliers detected or removed[/blue]")
    
    def _show_preprocessing_help(self):
        """Show preprocessing help."""
        self.ui.show_preprocessing_help("Missing Values")


class ModelManagementMenu(BaseMenu):
    """Menu for model management."""
    
    def show(self):
        """Show model management menu."""
        while True:
            self.ui.show_header("🤖 Model Management", "Configure and manage ML models")
            
            options = [
                "📋 View Available Models",
                "⚙️ Enable/Disable Models",
                "📚 Model Documentation",
                "🎛️ Hyperparameter Tuning",
                "🔙 Back to Main Menu"
            ]
            
            self.ui.show_menu("Select model option:", options)
            
            choice = self.ui.input("Enter your choice (1-5)", default="5")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self._show_available_models()
                elif choice == 2:
                    self._configure_models()
                elif choice == 3:
                    self._show_model_docs()
                elif choice == 4:
                    self._hyperparameter_tuning()
                elif choice == 5:
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number.[/red]")
            
            if choice != 5:
                input("\nPress Enter to continue...")
    
    def _show_available_models(self):
        """Show current model status."""
        self.ui.show_model_status_table(self.pipeline.available_models)
    
    def _configure_models(self):
        """Configure which models to enable/disable."""
        self.ui.print("[bold blue]⚙️ Model Configuration[/bold blue]")
        
        for model_name in self.pipeline.available_models.keys():
            current_status = "enabled" if self.pipeline.available_models[model_name] else "disabled"
            new_status = self.ui.confirm(f"Enable {model_name}? (currently {current_status})", 
                                       default=self.pipeline.available_models[model_name])
            self.pipeline.available_models[model_name] = new_status
        
        enabled_count = sum(self.pipeline.available_models.values())
        self.ui.print(f"[green]✅ Configuration updated! {enabled_count} models enabled.[/green]")
    
    def _show_model_docs(self):
        """Show model documentation."""
        models = list(self.pipeline.available_models.keys())
        
        self.ui.print("Available models:")
        for i, model in enumerate(models, 1):
            self.ui.print(f"[cyan]{i}.[/cyan] {model}")
        
        try:
            choice = int(self.ui.input(f"Select model (1-{len(models)})", default="1"))
            if 1 <= choice <= len(models):
                model_name = models[choice - 1]
                self.ui.show_model_documentation(model_name)
        except ValueError:
            self.ui.print("[red]❌ Invalid selection.[/red]")
    
    def _hyperparameter_tuning(self):
        """Hyperparameter tuning interface."""
        if self.pipeline.X_train is None:
            self.ui.print("[red]❌ No training data available. Load and preprocess data first.[/red]")
            return
        
        enabled_models = [name for name, enabled in self.pipeline.available_models.items() if enabled]
        
        if not enabled_models:
            self.ui.print("[red]❌ No models enabled for tuning.[/red]")
            return
        
        self.ui.print("Enabled models:")
        for i, model in enumerate(enabled_models, 1):
            self.ui.print(f"[cyan]{i}.[/cyan] {model}")
        
        try:
            choice = int(self.ui.input(f"Select model to tune (1-{len(enabled_models)})", default="1"))
            if 1 <= choice <= len(enabled_models):
                model_name = enabled_models[choice - 1]
                
                cv_folds = int(self.ui.input("CV folds for tuning", default="3"))
                
                self.ui.print(f"[blue]🎛️ Starting hyperparameter tuning for {model_name}...[/blue]")
                results = self.pipeline.model_trainer.tune_hyperparameters(model_name, self.pipeline, cv_folds)
                
                if 'error' not in results:
                    self.ui.print(f"[green]✅ Tuning completed![/green]")
                    self.ui.print(f"Best parameters: {results['best_params']}")
                    self.ui.print(f"Best CV score: {results['best_cv_score']:.4f}")
        except ValueError:
            self.ui.print("[red]❌ Invalid selection.[/red]")


class ResultsMenu(BaseMenu):
    """Menu for viewing results."""
    
    def show(self):
        """Show results menu."""
        if not self.pipeline.scores:
            self.ui.print("[red]❌ No model results available. Train models first.[/red]")
            input("Press Enter to continue...")
            return
        
        while True:
            self.ui.show_header("📊 Results & Analysis", "View and analyze model performance")
            
            options = [
                "📊 Performance Summary",
                "🏆 Best Model Details",
                "📈 Feature Importance",
                "📋 Detailed Metrics",
                "🔙 Back to Main Menu"
            ]
            
            self.ui.show_menu("Select results option:", options)
            
            choice = self.ui.input("Enter your choice (1-5)", default="5")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self._show_performance_summary()
                elif choice == 2:
                    self._show_best_model()
                elif choice == 3:
                    self._show_feature_importance()
                elif choice == 4:
                    self._show_detailed_metrics()
                elif choice == 5:
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number.[/red]")
            
            if choice != 5:
                input("\nPress Enter to continue...")
    
    def _show_performance_summary(self):
        """Show performance summary table."""
        import pandas as pd
        
        summary_data = []
        for model_name, scores in self.pipeline.scores.items():
            row = {'Model': model_name}
            row.update(scores)
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            self.ui.show_data_table(summary_df, title="Model Performance Summary")
    
    def _show_best_model(self):
        """Show details of the best performing model."""
        best_model_name, best_model = self.pipeline.model_trainer.get_best_model()
        
        if best_model_name:
            self.ui.print(f"[bold green]🏆 Best Model: {best_model_name}[/bold green]")
            
            scores = self.pipeline.scores[best_model_name]
            for metric, value in scores.items():
                self.ui.print(f"{metric}: {value:.4f}")
        else:
            self.ui.print("[yellow]⚠️ No best model found.[/yellow]")
    
    def _show_feature_importance(self):
        """Show feature importance for models that support it."""
        models_with_importance = []
        
        for model_name in self.pipeline.models.keys():
            importance = self.pipeline.model_trainer.get_feature_importance(model_name)
            if importance:
                models_with_importance.append(model_name)
        
        if not models_with_importance:
            self.ui.print("[yellow]⚠️ No models with feature importance available.[/yellow]")
            return
        
        self.ui.print("Models with feature importance:")
        for i, model in enumerate(models_with_importance, 1):
            self.ui.print(f"[cyan]{i}.[/cyan] {model}")
        
        try:
            choice = int(self.ui.input(f"Select model (1-{len(models_with_importance)})", default="1"))
            if 1 <= choice <= len(models_with_importance):
                model_name = models_with_importance[choice - 1]
                importance = self.pipeline.model_trainer.get_feature_importance(model_name)
                
                # Sort by importance
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                self.ui.print(f"\n[bold blue]📈 Feature Importance for {model_name}[/bold blue]")
                for i, (feature, imp) in enumerate(sorted_importance[:10], 1):  # Top 10
                    feature_name = self.pipeline.X_train.columns[feature] if hasattr(self.pipeline.X_train, 'columns') else f"Feature_{feature}"
                    self.ui.print(f"{i:2d}. {feature_name}: {imp:.4f}")
        except ValueError:
            self.ui.print("[red]❌ Invalid selection.[/red]")
    
    def _show_detailed_metrics(self):
        """Show detailed metrics for all models."""
        for model_name, scores in self.pipeline.scores.items():
            self.ui.print(f"\n[bold blue]📋 {model_name} Detailed Metrics[/bold blue]")
            
            for metric, value in scores.items():
                self.ui.print(f"  {metric}: {value:.4f}")
            
            # Show training time if available
            if model_name in self.pipeline.model_trainer.training_times:
                time_taken = self.pipeline.model_trainer.training_times[model_name]
                self.ui.print(f"  Training Time: {time_taken:.2f} seconds")


class DatabaseMenu(BaseMenu):
    """Menu for database operations."""
    
    def show(self):
        """Show database menu."""
        while True:
            self.ui.show_header("🗄️ Experiment Database", "View and manage experiment history")
            
            options = [
                "📋 View Experiment History",
                "🏆 Best Models",
                "📊 Model Comparison",
                "📁 Export Results",
                "📊 Database Statistics",
                "🔙 Back to Main Menu"
            ]
            
            self.ui.show_menu("Select database option:", options)
            
            choice = self.ui.input("Enter your choice (1-6)", default="6")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self._show_experiment_history()
                elif choice == 2:
                    self._show_best_models()
                elif choice == 3:
                    self._show_model_comparison()
                elif choice == 4:
                    self._export_results()
                elif choice == 5:
                    self._show_database_stats()
                elif choice == 6:
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number.[/red]")
            
            if choice != 6:
                input("\nPress Enter to continue...")
    
    def _show_experiment_history(self):
        """Show experiment history."""
        experiments = self.pipeline.db.get_experiment_history()
        
        if not experiments:
            self.ui.print("[yellow]⚠️ No experiments found in database.[/yellow]")
            return
        
        self.ui.print("[bold blue]📋 Experiment History[/bold blue]")
        for exp in experiments:
            self.ui.print(f"\nExperiment: {exp['experiment_name']}")
            self.ui.print(f"  Dataset: {exp['dataset_name']}")
            self.ui.print(f"  Problem: {exp['problem_type']}")
            self.ui.print(f"  Samples: {exp['n_samples']}")
            self.ui.print(f"  Features: {exp['n_features']}")
            self.ui.print(f"  Created: {exp['created_at']}")
    
    def _show_best_models(self):
        """Show best models across all experiments."""
        best_models = self.pipeline.db.get_best_models()
        
        if not best_models:
            self.ui.print("[yellow]⚠️ No model results found in database.[/yellow]")
            return
        
        self.ui.print("[bold blue]🏆 Best Models[/bold blue]")
        for model in best_models:
            self.ui.print(f"\nModel: {model['model_name']}")
            self.ui.print(f"  Experiment: {model['experiment_name']}")
            self.ui.print(f"  CV Score: {model['mean_cv_score']:.4f}")
            self.ui.print(f"  Training Time: {model['training_time']:.2f}s")
    
    def _show_model_comparison(self):
        """Show model comparison from database."""
        comparison_df = self.pipeline.db.get_model_comparison()
        
        if comparison_df.empty:
            self.ui.print("[yellow]⚠️ No model results found for comparison.[/yellow]")
            return
        
        self.ui.show_data_table(comparison_df, title="Model Comparison from Database")
    
    def _export_results(self):
        """Export results to CSV."""
        filename = self.ui.input("Enter output filename", default="experiment_results.csv")
        
        success = self.pipeline.db.export_results(filename)
        
        if success:
            self.ui.print(f"[green]✅ Results exported to {filename}[/green]")
        else:
            self.ui.print("[red]❌ Export failed.[/red]")
    
    def _show_database_stats(self):
        """Show database statistics."""
        stats = self.pipeline.db.get_database_stats()
        
        if stats:
            self.ui.print("[bold blue]📊 Database Statistics[/bold blue]")
            self.ui.print(f"Total Experiments: {stats['total_experiments']}")
            self.ui.print(f"Total Model Results: {stats['total_model_results']}")
            self.ui.print(f"Total Metrics: {stats['total_metrics']}")
        else:
            self.ui.print("[yellow]⚠️ Could not retrieve database statistics.[/yellow]")