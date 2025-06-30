import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class FeatureScaler:
    """Handles feature scaling with various methods."""
    
    def __init__(self, ui):
        self.ui = ui
        self.scalers = {}
    
    def scale_features(self, pipeline):
        """Enhanced feature scaling with user configuration."""
        numerical_cols = pipeline.X_train.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.ui.print("✅ No numerical features to scale")
            return
        
        scaling_method = getattr(pipeline, 'scaling_method', 'standard')
        
        if scaling_method == 'minmax':
            scaler = MinMaxScaler()
            scaler_name = "MinMaxScaler (0-1 range)"
        elif scaling_method == 'robust':
            scaler = RobustScaler()
            scaler_name = "RobustScaler (median-based)"
        else:
            scaler = StandardScaler()
            scaler_name = "StandardScaler (mean=0, std=1)"
        
        self.ui.print(f"📏 Scaling {len(numerical_cols)} features using {scaler_name}")
        
        pipeline.X_train[numerical_cols] = scaler.fit_transform(pipeline.X_train[numerical_cols])
        pipeline.X_test[numerical_cols] = scaler.transform(pipeline.X_test[numerical_cols])
        
        if pipeline.X_submission is not None:
            available_num_cols = [col for col in numerical_cols if col in pipeline.X_submission.columns]
            if available_num_cols:
                pipeline.X_submission[available_num_cols] = scaler.transform(pipeline.X_submission[available_num_cols])
        
        self.scalers['numerical'] = scaler
