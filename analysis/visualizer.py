"""
Data visualization module for comprehensive data exploration and model analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataVisualizer:
    """
    Comprehensive data visualization for ML pipeline.
    """
    
    def __init__(self, ui):
        """Initialize visualizer with UI reference."""
        self.ui = ui
        self.figure_size = (12, 8)
        self.style_set = False
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for consistent plots."""
        try:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            self.style_set = True
        except:
            # Fallback to default style
            plt.rcParams['figure.figsize'] = self.figure_size
    
    def plot_data_distribution(self, df: pd.DataFrame, target_column: str = None, 
                             columns: List[str] = None, max_plots: int = 12):
        """
        Plot distribution of numerical features.
        
        Args:
            df: DataFrame to visualize
            target_column: Target column for coloring
            columns: Specific columns to plot
            max_plots: Maximum number of plots to show
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        if columns:
            numerical_cols = [col for col in columns if col in numerical_cols]
        
        numerical_cols = numerical_cols[:max_plots]
        
        if not numerical_cols:
            self.ui.print("📊 No numerical columns to visualize")
            return
        
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            ax = axes[i] if len(numerical_cols) > 1 else axes
            
            # Plot histogram
            df[col].hist(bins=30, alpha=0.7, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
            # Add statistics text
            stats_text = f'Mean: {df[col].mean():.2f}\nStd: {df[col].std():.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        self.ui.print(f"📊 Distribution plots for {len(numerical_cols)} numerical features")
    
    def plot_missing_values(self, df: pd.DataFrame):
        """
        Visualize missing values pattern in the dataset.
        
        Args:
            df: DataFrame to analyze
        """
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) == 0:
            self.ui.print("✅ No missing values to visualize!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of missing values
        missing_data.plot(kind='bar', ax=ax1, color='coral')
        ax1.set_title('Missing Values by Column')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Number of Missing Values')
        ax1.tick_params(axis='x', rotation=45)
        
        # Heatmap of missing values pattern
        if len(df.columns) <= 50:  # Only for manageable number of columns
            missing_matrix = df.isnull()
            sns.heatmap(missing_matrix, cbar=True, ax=ax2, cmap='viridis')
            ax2.set_title('Missing Values Pattern')
            ax2.set_xlabel('Columns')
            ax2.set_ylabel('Rows')
        else:
            ax2.text(0.5, 0.5, f'Too many columns ({len(df.columns)}) for heatmap', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Missing Values Heatmap (Skipped)')
        
        plt.tight_layout()
        plt.show()
        self.ui.print(f"📊 Missing values visualization: {len(missing_data)} columns with missing data")
    
    def plot_correlation_matrix(self, df: pd.DataFrame, target_column: str = None, 
                              method: str = 'pearson', threshold: float = 0.1):
        """
        Plot correlation matrix for numerical features.
        
        Args:
            df: DataFrame to analyze
            target_column: Target column to highlight
            method: Correlation method
            threshold: Minimum correlation to display
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            self.ui.print("📊 Need at least 2 numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr(method=method)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(f'Correlation Matrix ({method.title()})')
        plt.tight_layout()
        plt.show()
        
        # Show high correlations with target if available
        if target_column and target_column in numerical_cols:
            target_corr = corr_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)
            high_corr = target_corr[target_corr > threshold]
            
            if len(high_corr) > 0:
                self.ui.print(f"\n🎯 High correlations with {target_column} (|r| > {threshold}):")
                for feature, corr in high_corr.head(10).items():
                    self.ui.print(f"  {feature}: {corr:.3f}")
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], 
                            primary_metric: str = 'Accuracy'):
        """
        Compare model performance visually.
        
        Args:
            results: Dictionary of model results
            primary_metric: Primary metric for comparison
        """
        if not results:
            self.ui.print("📊 No model results to visualize")
            return
        
        # Prepare data
        models = list(results.keys())
        scores = [results[model].get(primary_metric, 0) for model in models]
        times = [results[model].get('Training_Time', 0) for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison
        bars = ax1.bar(models, scores, color='skyblue', alpha=0.7)
        ax1.set_title(f'Model Performance Comparison ({primary_metric})')
        ax1.set_ylabel(primary_metric)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Training time comparison
        bars2 = ax2.bar(models, times, color='lightcoral', alpha=0.7)
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        self.ui.print(f"📊 Model comparison for {len(models)} models")