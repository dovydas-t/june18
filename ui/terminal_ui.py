# ui/terminal_ui.py - Enhanced Terminal UI
"""
Enhanced Terminal UI with comprehensive legends, tooltips, and help system.
Provides user-friendly interface with detailed explanations for all functionality.
"""

import time
from typing import List, Dict, Any
from pathlib import Path

from config import RICH_AVAILABLE, TABLE_WIDTH_LIMITS
from documentation.parameter_docs import ParameterDocumentation

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.text import Text
    from rich.align import Align
    from rich import box


class EnhancedTerminalUI:
    """
    Enhanced Terminal UI with comprehensive legends, tooltips, and help system.
    """
    
    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        # Initialize parameter documentation
        self.param_docs = ParameterDocumentation()
        self.model_docs = self.param_docs.get_model_docs()
        self.preprocessing_docs = self.param_docs.get_preprocessing_docs()
        self.metrics_docs = self.param_docs.get_evaluation_metrics_docs()
    
    def print(self, *args, **kwargs):
        """Enhanced print with Rich formatting."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args)
    
    def input(self, prompt, **kwargs):
        """Enhanced input with Rich formatting."""
        if self.console:
            return Prompt.ask(prompt, **kwargs)
        else:
            default = kwargs.get('default', '')
            user_input = input(f"{prompt} [{default}]: " if default else f"{prompt}: ")
            return user_input if user_input else default
    
    def confirm(self, prompt, default=True):
        """Enhanced confirmation dialog."""
        if self.console:
            return Confirm.ask(prompt, default=default)
        else:
            response = input(f"{prompt} (y/n): ").lower()
            return response in ['y', 'yes'] if response else default
    
    def show_header(self, title, subtitle=None):
        """Show enhanced header with optional subtitle."""
        if self.console:
            header_text = title
            if subtitle:
                header_text += f"\n[dim]{subtitle}[/dim]"
            
            self.console.print(Panel(
                Align.center(header_text),
                style="bold blue",
                box=box.DOUBLE,
                padding=(1, 2)
            ))
        else:
            print(f"\n{'='*60}\n{title}\n{'='*60}")
            if subtitle:
                print(f"{subtitle}\n")
    
    def show_legend(self, legend_type: str):
        """Show comprehensive legends for different UI elements."""
        legends = {
            "main_menu": {
                "title": "🏠 Main Menu Legend",
                "items": [
                    ("📁 Load Data", "Import training and test datasets, configure target column"),
                    ("🔍 Explore Data", "Analyze data quality, visualize distributions, get preprocessing recommendations"),
                    ("🔧 Configure Preprocessing", "Set up data cleaning, scaling, and feature engineering"),
                    ("🤖 Model Management", "Select models, view parameter explanations, configure hyperparameters"),
                    ("🚀 Run Pipeline", "Execute complete ML workflow with selected models"),
                    ("📊 View Results", "Compare model performance, analyze metrics, view visualizations"),
                    ("💾 Generate Submission", "Create predictions with best model for competition submission"),
                    ("📚 Help & Documentation", "Access comprehensive guides and parameter explanations")
                ]
            },
            
            "model_parameters": {
                "title": "🤖 Model Parameters Legend",
                "items": [
                    ("Regularization", "Controls overfitting by penalizing model complexity"),
                    ("Learning Rate", "Step size for gradient descent optimization"),
                    ("Tree Depth", "Maximum levels in decision tree (deeper = more complex)"),
                    ("Cross-Validation", "Technique to estimate model performance on unseen data"),
                    ("Grid Search", "Systematic search through hyperparameter combinations"),
                    ("Feature Importance", "Measure of how much each feature contributes to predictions")
                ]
            },
            
            "metrics": {
                "title": "📊 Evaluation Metrics Legend",
                "items": [
                    ("Accuracy", "Proportion of correct predictions (good for balanced data)"),
                    ("Precision", "Of positive predictions, how many were correct?"),
                    ("Recall", "Of actual positives, how many were found?"),
                    ("F1-Score", "Harmonic mean of precision and recall"),
                    ("RMSE", "Root Mean Square Error (lower is better for regression)"),
                    ("R²", "Coefficient of determination (higher is better, max = 1.0)")
                ]
            },
            
            "preprocessing": {
                "title": "🔧 Preprocessing Legend",
                "items": [
                    ("Missing Values", "Handle gaps in data (imputation strategies)"),
                    ("Feature Scaling", "Normalize feature ranges (StandardScaler, MinMaxScaler)"),
                    ("Outlier Detection", "Identify and handle extreme values"),
                    ("Feature Engineering", "Create new features from existing ones"),
                    ("Encoding", "Convert categorical variables to numerical format"),
                    ("Data Splitting", "Divide data into training and validation sets")
                ]
            }
        }
        
        if legend_type in legends:
            legend = legends[legend_type]
            
            if self.console:
                table = Table(title=legend["title"], box=box.ROUNDED, show_header=False)
                table.add_column("Term", style="cyan", width=20)
                table.add_column("Explanation", style="white")
                
                for term, explanation in legend["items"]:
                    table.add_row(term, explanation)
                
                self.console.print(table)
            else:
                print(f"\n{legend['title']}")
                print("-" * 50)
                for term, explanation in legend["items"]:
                    print(f"{term}: {explanation}")
    
    def show_model_documentation(self, model_name: str):
        """Show comprehensive documentation for a specific model."""
        if model_name not in self.model_docs:
            self.print(f"[red]Documentation not available for {model_name}[/red]")
            return
        
        doc = self.model_docs[model_name]
        
        if self.console:
            # Main information panel
            self.console.print(Panel(
                f"[bold]{doc['name']}[/bold]\n\n"
                f"{doc['description']}\n\n"
                f"[bold cyan]Mathematical Foundation:[/bold cyan]\n{doc['mathematical_foundation']}\n\n"
                f"[bold green]Pros:[/bold green] {', '.join(doc['pros'])}\n"
                f"[bold red]Cons:[/bold red] {', '.join(doc['cons'])}",
                title=f"📖 {model_name} Documentation",
                border_style="blue"
            ))
            
            # Parameters table
            param_table = Table(title="🔧 Parameters", box=box.ROUNDED)
            param_table.add_column("Parameter", style="cyan", width=15)
            param_table.add_column("Description", style="white", width=30)
            param_table.add_column("Range/Options", style="yellow", width=20)
            param_table.add_column("Effect", style="green", width=25)
            
            for param_name, param_info in doc['parameters'].items():
                param_table.add_row(
                    param_name,
                    param_info['description'],
                    param_info.get('range', str(param_info.get('options', 'N/A'))),
                    param_info['effect']
                )
            
            self.console.print(param_table)
            
            # Additional info
            self.console.print(Panel(
                f"[bold cyan]Use Cases:[/bold cyan] {', '.join(doc['use_cases'])}\n\n"
                f"[bold yellow]Preprocessing Requirements:[/bold yellow]\n" +
                '\n'.join([f"• {req}" for req in doc['preprocessing_requirements']]) + "\n\n"
                f"[bold magenta]Computational Complexity:[/bold magenta] {doc['complexity']}",
                title="📋 Additional Information",
                border_style="green"
            ))
        else:
            print(f"\n{doc['name']} Documentation")
            print("=" * 50)
            print(f"Description: {doc['description']}")
            print(f"Mathematical Foundation: {doc['mathematical_foundation']}")
            print(f"Use Cases: {', '.join(doc['use_cases'])}")
            print(f"Pros: {', '.join(doc['pros'])}")
            print(f"Cons: {', '.join(doc['cons'])}")
            print(f"Complexity: {doc['complexity']}")
    
    def show_preprocessing_help(self, topic: str):
        """Show help for preprocessing topics."""
        if topic not in self.preprocessing_docs:
            self.print(f"[red]Documentation not available for {topic}[/red]")
            return
        
        doc = self.preprocessing_docs[topic]
        
        if self.console:
            content = f"[bold]{doc['description']}[/bold]\n\n"
            
            if 'methods' in doc:
                content += "[bold cyan]Methods:[/bold cyan]\n"
                for method, description in doc['methods'].items():
                    content += f"• [yellow]{method}:[/yellow] {description}\n"
            
            if 'considerations' in doc:
                content += "\n[bold red]Important Considerations:[/bold red]\n"
                for consideration in doc['considerations']:
                    content += f"• {consideration}\n"
            
            if 'when_to_use' in doc:
                content += "\n[bold green]When to Use:[/bold green]\n"
                for method, usage in doc['when_to_use'].items():
                    content += f"• [yellow]{method}:[/yellow] {usage}\n"
            
            self.console.print(Panel(content, title=f"📚 {topic} Guide", border_style="cyan"))
        else:
            print(f"\n{topic} Documentation")
            print("=" * 40)
            print(doc['description'])
    
    def show_metrics_explanation(self, problem_type: str):
        """Show detailed explanation of evaluation metrics."""
        metrics_type = "Classification Metrics" if problem_type == "classification" else "Regression Metrics"
        
        if metrics_type not in self.metrics_docs:
            return
        
        metrics = self.metrics_docs[metrics_type]
        
        if self.console:
            table = Table(title=f"📊 {metrics_type} Explanation", box=box.ROUNDED)
            table.add_column("Metric", style="cyan", width=15)
            table.add_column("Formula", style="yellow", width=25)
            table.add_column("Interpretation", style="white", width=30)
            table.add_column("Best For", style="green", width=20)
            
            for metric_name, metric_info in metrics.items():
                table.add_row(
                    metric_name,
                    metric_info['formula'],
                    metric_info['interpretation'],
                    metric_info['best_for']
                )
            
            self.console.print(table)
        else:
            print(f"\n{metrics_type}")
            print("=" * 40)
            for metric_name, metric_info in metrics.items():
                print(f"{metric_name}: {metric_info['interpretation']}")
    
    def show_progress_with_steps(self, steps: List[str]):
        """Show progress indicator with step descriptions."""
        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                for i, step in enumerate(steps):
                    task = progress.add_task(step, total=100)
                    time.sleep(0.1)  # Simulate work
                    progress.update(task, completed=100)
                    if i < len(steps) - 1:
                        progress.remove_task(task)
        else:
            for step in steps:
                print(f"Processing: {step}")
    
    def show_interactive_help(self):
        """Show comprehensive interactive help system."""
        while True:
            if self.console:
                self.console.print(Panel(
                    "[bold blue]📚 Interactive Help System[/bold blue]\n\n"
                    "Select a topic to learn more about:",
                    title="Help & Documentation",
                    border_style="blue"
                ))
            
            options = [
                "🤖 Machine Learning Models",
                "🔧 Preprocessing Techniques", 
                "📊 Evaluation Metrics",
                "💡 Best Practices",
                "🎯 Parameter Tuning Guide",
                "📈 Interpreting Results",
                "⬅️ Back to Main Menu"
            ]
            
            self.show_menu("Choose help topic:", options)
            choice = self.input("Enter your choice (1-7)", default="7")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self._show_models_help()
                elif choice == 2:
                    self._show_preprocessing_help_menu()
                elif choice == 3:
                    self._show_metrics_help_menu()
                elif choice == 4:
                    self._show_best_practices()
                elif choice == 5:
                    self._show_tuning_guide()
                elif choice == 6:
                    self._show_results_interpretation()
                elif choice == 7:
                    break
                else:
                    self.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.print("[red]❌ Please enter a valid number.[/red]")
            
            input("\nPress Enter to continue...")
    
    def show_menu(self, title: str, options: List[str]):
        """Show enhanced menu with Rich formatting."""
        if self.console:
            self.console.print(f"\n[bold blue]{title}[/bold blue]")
            for i, option in enumerate(options, 1):
                self.console.print(f"[cyan]{i}.[/cyan] {option}")
        else:
            print(f"\n{title}")
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
    
    def show_data_table(self, df, title="Data Preview", max_rows=10, max_cols=8):
        """Show data in a formatted table."""
        if self.console:
            # Limit rows and columns for display
            display_df = df.head(max_rows)
            if len(df.columns) > max_cols:
                display_df = display_df.iloc[:, :max_cols]
                
            table = Table(title=title, box=box.ROUNDED)
            
            # Add columns
            for col in display_df.columns:
                table.add_column(str(col), style="white", width=TABLE_WIDTH_LIMITS['small'])
            
            # Add rows
            for _, row in display_df.iterrows():
                table.add_row(*[str(val)[:15] + "..." if len(str(val)) > 15 else str(val) for val in row])
            
            if len(df) > max_rows or len(df.columns) > max_cols:
                table.caption = f"Showing {min(max_rows, len(df))} of {len(df)} rows, {min(max_cols, len(df.columns))} of {len(df.columns)} columns"
            
            self.console.print(table)
        else:
            print(f"\n{title}")
            print(df.head(max_rows).to_string())
    
    def show_model_status_table(self, available_models: Dict[str, bool]):
        """Show current model status in a comprehensive table."""
        if self.console:
            status_table = Table(title="🤖 Current Model Configuration", box=box.ROUNDED)
            status_table.add_column("Model", style="cyan", width=20)
            status_table.add_column("Status", style="white", width=15)
            status_table.add_column("Type", style="yellow", width=15)
            status_table.add_column("Best For", style="green", width=30)
            status_table.add_column("Complexity", style="red", width=12)
            
            model_info = {
                'KNN': ('Instance-based', 'Small datasets, non-linear patterns', 'Low'),
                'SVM': ('Kernel-based', 'High-dimensional data, text classification', 'Medium'),
                'Decision Tree': ('Tree-based', 'Interpretable models, mixed data', 'Low'),
                'Linear Model': ('Linear', 'Baseline models, linear relationships', 'Low'),
                'Random Forest': ('Ensemble', 'General purpose, robust performance', 'Medium'),
                'Extra Trees': ('Ensemble', 'Fast training, good generalization', 'Medium'),
                'Bagging': ('Ensemble', 'Reduce overfitting, stable predictions', 'Medium'),
                'AdaBoost': ('Boosting', 'Sequential improvement, weighted samples', 'Medium'),
                'Hist Gradient Boosting': ('Boosting', 'Large datasets, memory efficient', 'High'),
                'XGBoost': ('Boosting', 'Competitions, structured data', 'High'),
                'LightGBM': ('Boosting', 'Fast training, high accuracy', 'High')
            }
            
            for model_name, enabled in available_models.items():
                if enabled:
                    status = "✅ Enabled"
                    status_style = "green"
                else:
                    status = "⏸️ Disabled"
                    status_style = "yellow"
                
                info = model_info.get(model_name, ('Unknown', 'General purpose', 'Medium'))
                
                status_table.add_row(
                    model_name,
                    f"[{status_style}]{status}[/{status_style}]",
                    info[0],
                    info[1],
                    info[2]
                )
            
            self.console.print(status_table)
        else:
            print("\nCurrent Model Status:")
            for model_name, enabled in available_models.items():
                status = "Enabled" if enabled else "Disabled"
                print(f"  {model_name}: {status}")
    
    def _show_models_help(self):
        """Show help for machine learning models."""
        model_names = list(self.model_docs.keys())
        
        self.print("\n[bold]Available Models:[/bold]")
        for i, model in enumerate(model_names, 1):
            self.print(f"[cyan]{i}.[/cyan] {model}")
        
        try:
            choice = int(self.input(f"Select model (1-{len(model_names)}, 0 for overview)", default="0"))
            if choice == 0:
                self._show_model_overview()
            elif 1 <= choice <= len(model_names):
                self.show_model_documentation(model_names[choice-1])
        except ValueError:
            self.print("[red]❌ Invalid selection.[/red]")
    
    def _show_model_overview(self):
        """Show overview of all models."""
        if self.console:
            table = Table(title="🤖 Model Overview", box=box.ROUNDED)
            table.add_column("Model", style="cyan", width=20)
            table.add_column("Type", style="yellow", width=15)
            table.add_column("Best For", style="green", width=30)
            table.add_column("Complexity", style="red", width=15)
            
            model_types = {
                "KNN": ("Instance-based", "Small datasets, Non-linear patterns", "Low"),
                "SVM": ("Kernel-based", "High-dimensional data, Text classification", "Medium"),
                "Decision Tree": ("Tree-based", "Interpretable models, Mixed data types", "Low"),
                "Random Forest": ("Ensemble", "General purpose, Feature importance", "Medium"),
                "XGBoost": ("Gradient Boosting", "Competitions, Structured data", "High")
            }
            
            for model, (type_desc, best_for, complexity) in model_types.items():
                table.add_row(model, type_desc, best_for, complexity)
            
            self.console.print(table)
        else:
            print("\nModel Overview:")
            print("KNN: Instance-based learning")
            print("SVM: Kernel-based classification/regression")
            print("Decision Tree: Tree-based interpretable model")
            print("Random Forest: Ensemble method")
            print("XGBoost: Gradient boosting")
    
    def _show_preprocessing_help_menu(self):
        """Show preprocessing help menu."""
        topics = list(self.preprocessing_docs.keys())
        
        self.print("\n[bold]Preprocessing Topics:[/bold]")
        for i, topic in enumerate(topics, 1):
            self.print(f"[cyan]{i}.[/cyan] {topic}")
        
        try:
            choice = int(self.input(f"Select topic (1-{len(topics)})", default="1"))
            if 1 <= choice <= len(topics):
                self.show_preprocessing_help(topics[choice-1])
        except ValueError:
            self.print("[red]❌ Invalid selection.[/red]")
    
    def _show_metrics_help_menu(self):
        """Show metrics help menu."""
        self.print("\n[bold]Evaluation Metrics:[/bold]")
        self.print("[cyan]1.[/cyan] Classification Metrics")
        self.print("[cyan]2.[/cyan] Regression Metrics")
        
        try:
            choice = int(self.input("Select metrics type (1-2)", default="1"))
            if choice == 1:
                self.show_metrics_explanation("classification")
            elif choice == 2:
                self.show_metrics_explanation("regression")
        except ValueError:
            self.print("[red]❌ Invalid selection.[/red]")
    
    def _show_best_practices(self):
        """Show ML best practices."""
        practices = [
            "🎯 Always establish a baseline model first (simple linear/tree model)",
            "📊 Understand your data before modeling (EDA is crucial)",
            "🔧 Preprocess data consistently across train/validation/test sets",
            "⚖️ Use appropriate evaluation metrics for your problem type",
            "🔄 Always use cross-validation for model selection",
            "📈 Monitor for overfitting (train vs validation performance)",
            "🎛️ Start with default parameters, then tune systematically",
            "📋 Keep detailed logs of experiments and results",
            "🧪 Test preprocessing steps on validation data",
            "🔍 Interpret model results and feature importance"
        ]
        
        if self.console:
            self.console.print(Panel(
                "\n".join(practices),
                title="💡 Machine Learning Best Practices",
                border_style="green"
            ))
        else:
            print("\nMachine Learning Best Practices:")
            for practice in practices:
                print(practice)
    
    def _show_tuning_guide(self):
        """Show parameter tuning guide."""
        if self.console:
            guide_text = """[bold cyan]🎯 Parameter Tuning Strategy[/bold cyan]

[bold yellow]1. Start Simple:[/bold yellow]
• Begin with default parameters
• Establish baseline performance
• Understand data characteristics

[bold yellow]2. Systematic Approach:[/bold yellow]
• Tune one parameter at a time initially
• Use cross-validation for reliable estimates
• Focus on most impactful parameters first

[bold yellow]3. Common Parameter Priorities:[/bold yellow]
• Tree models: max_depth, min_samples_split
• Ensemble: n_estimators, max_features
• SVM: C, gamma, kernel
• Neural networks: learning_rate, hidden_units

[bold yellow]4. Avoid Overfitting:[/bold yellow]
• Use separate validation set
• Monitor train vs validation performance
• Apply regularization when needed

[bold yellow]5. Computational Efficiency:[/bold yellow]
• Start with coarse grid, then fine-tune
• Use random search for high-dimensional spaces
• Consider Bayesian optimization for expensive models"""
            
            self.console.print(Panel(guide_text, title="🎛️ Parameter Tuning Guide", border_style="yellow"))
        else:
            print("\nParameter Tuning Guide:")
            print("1. Start with default parameters")
            print("2. Tune systematically using cross-validation")
            print("3. Focus on most impactful parameters first")
            print("4. Monitor for overfitting")
            print("5. Use computational resources efficiently")
    
    def _show_results_interpretation(self):
        """Show results interpretation guide."""
        if self.console:
            interpretation_text = """[bold cyan]📈 Interpreting ML Results[/bold cyan]

[bold yellow]Performance Metrics:[/bold yellow]
• Don't rely on a single metric
• Consider domain-specific requirements
• Compare against baseline models

[bold yellow]Model Comparison:[/bold yellow]
• Statistical significance of differences
• Computational cost vs performance trade-offs
• Robustness across different data splits

[bold yellow]Feature Importance:[/bold yellow]
• Identify key predictive features
• Check for feature leakage
• Validate importance with domain knowledge

[bold yellow]Error Analysis:[/bold yellow]
• Analyze prediction errors patterns
• Identify challenging cases
• Look for systematic biases

[bold yellow]Generalization:[/bold yellow]
• Cross-validation consistency
• Performance on holdout test set
• Robustness to data distribution shifts"""
            
            self.console.print(Panel(interpretation_text, title="📊 Results Interpretation", border_style="blue"))
        else:
            print("\nResults Interpretation Guide:")
            print("• Use multiple metrics for evaluation")
            print("• Compare models statistically")
            print("• Analyze feature importance")
            print("• Examine prediction errors")
            print("• Test generalization ability")

    def _show_visualization_menu(self, pipeline):
        """Show visualization options for data exploration."""
        if not hasattr(pipeline, 'analyzer') or not hasattr(pipeline.analyzer, 'visualizer'):
            # Initialize visualizer if not present
            try:
                from analysis.visualizer import DataVisualizer
                pipeline.analyzer.visualizer = DataVisualizer(self)
            except ImportError:
                self.print("[red]❌ Visualization module not available[/red]")
                return
        
        visualizer = pipeline.analyzer.visualizer
        
        while True:
            self.print("\n📊 DATA VISUALIZATION")
            self.print("="*30)
            
            options = [
                "📈 Feature Distributions",
                "❓ Missing Values Pattern", 
                "🔗 Correlation Matrix",
                "🤖 Model Performance Comparison",
                "🔙 Back to Exploration"
            ]
            
            self.show_menu("Select visualization:", options)
            choice = self.input("Enter choice (1-5)", default="5")
            
            try:
                choice = int(choice)
                if choice == 1:
                    visualizer.plot_data_distribution(pipeline.train_data, pipeline.target_column)
                elif choice == 2:
                    visualizer.plot_missing_values(pipeline.train_data)
                elif choice == 3:
                    visualizer.plot_correlation_matrix(pipeline.train_data, pipeline.target_column)
                elif choice == 4:
                    if pipeline.scores:
                        primary_metric = pipeline._get_primary_metric()
                        visualizer.plot_model_comparison(pipeline.scores, primary_metric)
                    else:
                        self.print("[yellow]⚠️ No model results available[/yellow]")
                elif choice == 5:
                    break
                else:
                    self.print("[red]❌ Invalid choice[/red]")
            except ValueError:
                self.print("[red]❌ Please enter a valid number[/red]")
            except Exception as e:
                self.print(f"[red]❌ Visualization error: {str(e)}[/red]")
            
            if choice != 5:
                input("\nPress Enter to continue...")