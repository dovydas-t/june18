import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class MissingValuesHandler:
    """Handles missing values with various strategies."""
    
    def __init__(self, ui):
        self.ui = ui
        self.imputers = {}
    
    def handle_missing_values(self, pipeline):
        """Enhanced missing values handling with user configuration."""
        train_missing = pipeline.X_train.isnull().sum()
        missing_cols = train_missing[train_missing > 0]
        
        if len(missing_cols) == 0:
            self.ui.print("✅ No missing values to handle")
            return
        
        self.ui.print(f"🔧 Handling missing values in {len(missing_cols)} columns")
        
        if hasattr(pipeline, 'missing_config') and pipeline.missing_config:
            self._apply_manual_missing_config(pipeline, missing_cols)
        else:
            missing_strategy = getattr(pipeline, 'missing_strategy', 'auto')
            if missing_strategy == 'auto':
                self._handle_missing_values_auto(pipeline, missing_cols)
            else:
                self.ui.print(f"Using missing values strategy: {missing_strategy}")
    
    def _apply_manual_missing_config(self, pipeline, missing_cols):
        """Apply manually configured missing values strategies."""
        for col in missing_cols.index:
            if col in pipeline.missing_config:
                strategy = pipeline.missing_config[col]
                self.ui.print(f"  {col}: Using {strategy}")
                
                if strategy == 'drop':
                    pipeline.X_train = pipeline.X_train.drop(columns=[col])
                    pipeline.X_test = pipeline.X_test.drop(columns=[col])
                    if pipeline.X_submission is not None:
                        pipeline.X_submission = pipeline.X_submission.drop(columns=[col])
                    continue
                
                # Create appropriate imputer
                if strategy in ['mean', 'median']:
                    imputer = SimpleImputer(strategy=strategy)
                elif strategy == 'most_frequent':
                    imputer = SimpleImputer(strategy='most_frequent')
                elif strategy == 'constant':
                    fill_value = 0 if pipeline.X_train[col].dtype in ['int64', 'float64'] else 'Unknown'
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                else:
                    if pipeline.X_train[col].dtype in ['int64', 'float64']:
                        imputer = SimpleImputer(strategy='median')
                    else:
                        imputer = SimpleImputer(strategy='most_frequent')
                
                self._apply_imputer(pipeline, col, imputer)
    
    def _handle_missing_values_auto(self, pipeline, missing_cols):
        """Automatic missing values handling."""
        numerical_cols = pipeline.X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = pipeline.X_train.select_dtypes(include=['object']).columns
        
        numerical_missing = [col for col in missing_cols.index if col in numerical_cols]
        if numerical_missing:
            self.ui.print(f"  📊 Imputing {len(numerical_missing)} numerical columns with median")
            imputer = SimpleImputer(strategy='median')
            self._apply_imputer_multiple(pipeline, numerical_missing, imputer)
        
        categorical_missing = [col for col in missing_cols.index if col in categorical_cols]
        if categorical_missing:
            self.ui.print(f"  🏷️ Imputing {len(categorical_missing)} categorical columns with mode")
            imputer = SimpleImputer(strategy='most_frequent')
            self._apply_imputer_multiple(pipeline, categorical_missing, imputer)
    
    def _apply_imputer(self, pipeline, col, imputer):
        """Apply imputer to a single column."""
        train_reshaped = pipeline.X_train[[col]]
        test_reshaped = pipeline.X_test[[col]]
        
        pipeline.X_train[col] = imputer.fit_transform(train_reshaped).flatten()
        pipeline.X_test[col] = imputer.transform(test_reshaped).flatten()
        
        if pipeline.X_submission is not None and col in pipeline.X_submission.columns:
            sub_reshaped = pipeline.X_submission[[col]]
            pipeline.X_submission[col] = imputer.transform(sub_reshaped).flatten()
    
    def _apply_imputer_multiple(self, pipeline, columns, imputer):
        """Apply imputer to multiple columns."""
        pipeline.X_train[columns] = imputer.fit_transform(pipeline.X_train[columns])
        pipeline.X_test[columns] = imputer.transform(pipeline.X_test[columns])
        
        if pipeline.X_submission is not None:
            available_cols = [col for col in columns if col in pipeline.X_submission.columns]
            if available_cols:
                pipeline.X_submission[available_cols] = imputer.transform(pipeline.X_submission[available_cols])
