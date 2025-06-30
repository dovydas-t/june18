"""
File operation utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any


class SubmissionGenerator:
    """Generate submission files for competitions."""
    
    def __init__(self, pipeline):
        """Initialize with pipeline object."""
        self.pipeline = pipeline
        self.ui = pipeline.ui
    
    def generate(self):
        """Generate submission file with proper ID column handling."""
        if not self.pipeline.models:
            self.ui.print("[red]❌ No trained models available.[/red]")
            return
        
        if self.pipeline.X_submission is None:
            self.ui.print("[red]❌ No test data available for submission.[/red]")
            return
        
        # Get best model
        best_model_name, best_model = self.pipeline.model_trainer.get_best_model()
        
        if not best_model:
            self.ui.print("[red]❌ No best model found.[/red]")
            return
        
        self.ui.print(f"[blue]📁 Generating submission with {best_model_name}...[/blue]")
        
        # Generate predictions
        try:
            predictions = self.pipeline.model_trainer.generate_predictions(
                best_model_name, self.pipeline.X_submission
            )
            
            # Create submission DataFrame
            submission_df = self._prepare_submission_data(predictions)
            
            # Save to file
            filename = self.ui.input("Enter submission filename", default="submission.csv")
            submission_df.to_csv(filename, index=False)
            
            self.ui.print(f"[green]✅ Submission saved to {filename}[/green]")
            self.ui.print(f"Shape: {submission_df.shape}")
            
            # Show preview
            self._show_submission_preview(submission_df)
            
        except Exception as e:
            self.ui.print(f"[red]❌ Error generating submission: {str(e)}[/red]")

    def _prepare_submission_data(self, predictions):
        """
        Prepare submission data with proper ID column handling.
        
        Args:
            predictions: Model predictions array
            
        Returns:
            DataFrame: Properly formatted submission data
        """
        submission_df = pd.DataFrame()
        
        # Handle ID columns
        if hasattr(self.pipeline, 'submission_ids') and self.pipeline.submission_ids is not None:
            # Use preserved ID columns from test data
            for id_col in self.pipeline.submission_ids.columns:
                submission_df[id_col] = self.pipeline.submission_ids[id_col].values
        elif hasattr(self.pipeline, 'id_columns') and self.pipeline.id_columns:
            # Try to get ID from test data
            if self.pipeline.test_data is not None:
                for id_col in self.pipeline.id_columns:
                    if id_col in self.pipeline.test_data.columns:
                        submission_df[id_col] = self.pipeline.test_data[id_col].values
                        break
        
        # If no ID column found, create default
        if submission_df.empty:
            submission_df['Id'] = range(len(predictions))
        
        # Add predictions
        submission_df[self.pipeline.target_column] = predictions
        
        return submission_df

    def _show_submission_preview(self, submission_df):
        """Show preview of submission file."""
        self.ui.print(f"\n📋 Submission Preview:")
        self.ui.print(f"Columns: {list(submission_df.columns)}")
        self.ui.print(f"Shape: {submission_df.shape}")
        
        # Show first few rows
        if len(submission_df) > 0:
            self.ui.print(f"\nFirst 5 rows:")
            for i, row in submission_df.head().iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                self.ui.print(f"  {i}: {row_str}")
        
        # Basic validation
        if submission_df[self.pipeline.target_column].isnull().any():
            self.ui.print("[yellow]⚠️ Warning: Some predictions are null[/yellow]")
        
        self.ui.print(f"[green]✅ Submission validation complete[/green]")