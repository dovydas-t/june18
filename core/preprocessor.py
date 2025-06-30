import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from preprocessing.data_splitter import DataSplitter
from preprocessing.missing_values_handler import MissingValuesHandler
from preprocessing.feature_scaler import FeatureScaler
from preprocessing.outlier_detector import OutlierDetector
from preprocessing.encoding_manager import EncodingManager

class DataPreprocessor:
    """
    Simplified data preprocessing coordinator using modular components.
    Reduced from 800+ lines to ~100 lines by delegating to specialized modules.
    """
    
    def __init__(self, ui):
        """Initialize data preprocessor with modular components."""
        self.ui = ui
        
        # Initialize specialized preprocessing components
        self.data_splitter = DataSplitter(ui)
        self.missing_handler = MissingValuesHandler(ui)
        self.feature_scaler = FeatureScaler(ui)
        self.outlier_detector = OutlierDetector(ui)
        self.encoding_manager = EncodingManager(ui)
        
        # Track preprocessing steps
        self.preprocessing_steps = []

    def preprocess_data(self, pipeline) -> bool:
        """
        Enhanced preprocessing pipeline using modular components.
        
        Args:
            pipeline: Reference to main pipeline object
            
        Returns:
            bool: True if preprocessing successful
        """
        try:
            self.ui.print("\n🔧 STARTING DATA PREPROCESSING v3.0")
            self.ui.print("=" * 45)
            
            # Step 1: Split data
            self.ui.print("📊 Step 1: Splitting data...")
            self.data_splitter.split_data(pipeline)
            self.preprocessing_steps.append("Data splitting completed")
            
            # Step 2: Handle missing values
            self.ui.print("\n📋 Step 2: Handling missing values...")
            self.missing_handler.handle_missing_values(pipeline)
            self.preprocessing_steps.append("Missing values handled")
            
            # Step 3: Encode categorical variables
            self.ui.print("\n🏷️ Step 3: Encoding categorical variables...")
            self.encoding_manager.encode_categorical(pipeline)
            self.preprocessing_steps.append("Categorical encoding completed")
            
            # Step 4: Scale numerical features
            self.ui.print("\n📏 Step 4: Scaling numerical features...")
            self.feature_scaler.scale_features(pipeline)
            self.preprocessing_steps.append("Feature scaling completed")
            
            # Step 5: Handle outliers (if configured)
            if hasattr(pipeline, 'outlier_method') and pipeline.outlier_method:
                self.ui.print("\n🎯 Step 5: Removing outliers...")
                self.outlier_detector.remove_outliers(pipeline)
                self.preprocessing_steps.append("Outlier removal completed")
            
            # Update pipeline with preprocessing steps
            pipeline.preprocessing_steps = self.preprocessing_steps
            
            self.ui.print("\n✅ Data preprocessing v3.0 completed successfully!")
            self._show_preprocessing_summary(pipeline)
            return True
            
        except Exception as e:
            self.ui.print(f"❌ Preprocessing failed: {str(e)}")
            return False
    
    def _show_preprocessing_summary(self, pipeline):
        """Show comprehensive preprocessing summary."""
        self.ui.print("\n📋 PREPROCESSING SUMMARY v3.0")
        self.ui.print("=" * 35)
        self.ui.print(f"✅ Final training shape: {pipeline.X_train.shape}")
        self.ui.print(f"✅ Final validation shape: {pipeline.X_test.shape}")
        
        if pipeline.X_submission is not None:
            self.ui.print(f"✅ Submission data shape: {pipeline.X_submission.shape}")
        
        # Show data types
        numeric_count = len(pipeline.X_train.select_dtypes(include=[np.number]).columns)
        categorical_count = len(pipeline.X_train.select_dtypes(include=['object']).columns)
        
        self.ui.print(f"📊 Features: {numeric_count} numerical, {categorical_count} categorical")
        
        # Show any remaining missing values
        missing_after = pipeline.X_train.isnull().sum().sum()
        if missing_after > 0:
            self.ui.print(f"⚠️ Remaining missing values: {missing_after}")
        else:
            self.ui.print("✅ No missing values remaining")
        
        # Show completed steps
        self.ui.print(f"\n🔧 Completed preprocessing steps:")
        for i, step in enumerate(self.preprocessing_steps, 1):
            self.ui.print(f"  {i}. {step}")
    
    def get_preprocessing_info(self, pipeline) -> Dict[str, Any]:
        """Get comprehensive preprocessing information."""
        info = {
            'train_shape': pipeline.X_train.shape if hasattr(pipeline, 'X_train') else None,
            'test_shape': pipeline.X_test.shape if hasattr(pipeline, 'X_test') else None,
            'feature_columns': getattr(pipeline, 'feature_columns', []),
            'id_columns': getattr(pipeline, 'id_columns', []),
            'scaling_method': getattr(pipeline, 'scaling_method', None),
            'missing_strategy': getattr(pipeline, 'missing_strategy', None),
            'outlier_method': getattr(pipeline, 'outlier_method', None),
            'encoders_used': len(self.encoding_manager.encoders),
            'scalers_used': len(self.feature_scaler.scalers),
            'preprocessing_steps': self.preprocessing_steps
        }
        return info