import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataSplitter:
    """Handles train/validation data splitting."""
    
    def __init__(self, ui):
        self.ui = ui
    
    def split_data(self, pipeline):
        """Enhanced data splitting with better ID column handling."""
        # Handle ID columns first
        self._handle_id_columns(pipeline)
        
        # Get feature columns (exclude target and ID columns)
        exclude_cols = [pipeline.target_column] + getattr(pipeline, 'id_columns', [])
        feature_columns = [col for col in pipeline.train_data.columns if col not in exclude_cols]
        
        X = pipeline.train_data[feature_columns].copy()
        y = pipeline.train_data[pipeline.target_column].copy()
        
        self.ui.print(f"📊 Splitting data with {len(feature_columns)} features")
        
        # Split the data with proper stratification
        try:
            stratify = None
            if pipeline.problem_type == 'classification' and y.nunique() > 1:
                min_class_count = y.value_counts().min()
                if min_class_count >= 2:
                    stratify = y
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=pipeline.test_size,
                random_state=pipeline.random_state,
                stratify=stratify
            )
        except ValueError as e:
            self.ui.print(f"⚠️ Stratification failed: {e}")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=pipeline.test_size,
                random_state=pipeline.random_state
            )
        
        # Store splits
        pipeline.X_train = X_train
        pipeline.X_test = X_val
        pipeline.y_train = y_train
        pipeline.y_test = y_val
        pipeline.feature_columns = feature_columns
        
        # Handle submission data if available
        if pipeline.test_data is not None:
            available_features = [col for col in feature_columns if col in pipeline.test_data.columns]
            if len(available_features) != len(feature_columns):
                missing_features = set(feature_columns) - set(available_features)
                self.ui.print(f"⚠️ Missing features in test data: {missing_features}")
            
            pipeline.X_submission = pipeline.test_data[available_features].copy()
            
            if hasattr(pipeline, 'id_columns') and pipeline.id_columns:
                available_ids = [col for col in pipeline.id_columns if col in pipeline.test_data.columns]
                if available_ids:
                    pipeline.submission_ids = pipeline.test_data[available_ids].copy()
        
        self.ui.print(f"✅ Train: {len(X_train)}, Validation: {len(X_val)}")

    def _show_column_analysis(self, df, col, confidence, target_col):
        """Show detailed column analysis for ID detection."""
        self.ui.print(f"\n🔍 Analyzing column: {col}")
        self.ui.print(f"   Confidence as ID: {confidence:.1f}%")
        self.ui.print(f"   Data type: {df[col].dtype}")
        self.ui.print(f"   Unique values: {df[col].nunique()}/{len(df)} ({df[col].nunique()/len(df)*100:.1f}%)")
        
        # Show sample values
        sample_values = df[col].dropna().head(5).tolist()
        self.ui.print(f"   Sample values: {sample_values}")
        
        # Recommendation
        if confidence > 70:
            self.ui.print("   💡 Recommendation: Likely ID column - exclude from modeling")
        elif confidence > 40:
            self.ui.print("   ⚠️ Recommendation: Possibly ID column - consider excluding")
        else:
            self.ui.print("   ✅ Recommendation: Keep for modeling")
    
    def _configure_columns_interactive(self, df, potential_ids, target_col):
        """Interactive column configuration."""
        exclude_cols = []
        
        self.ui.print(f"\n🔧 COLUMN CONFIGURATION")
        self.ui.print("=" * 30)
        
        for col, confidence in potential_ids:
            if col == target_col:
                continue
                
            self.ui.print(f"\nColumn: {col} (ID confidence: {confidence:.1f}%)")
            
            # Auto-exclude high confidence ID columns
            if confidence > 80:
                exclude_cols.append(col)
                self.ui.print(f"   ✅ Auto-excluded (high ID confidence)")
            else:
                # Ask user
                exclude = self.ui.confirm(f"Exclude '{col}' from modeling?", 
                                        default=confidence > 50)
                if exclude:
                    exclude_cols.append(col)
        
        return {'exclude': exclude_cols}

    def _handle_id_columns(self, pipeline):
        """Enhanced ID column detection and interactive selection."""
        pipeline.id_columns = []
        potential_ids = []
        
        for col in pipeline.train_data.columns:
            if col == pipeline.target_column:
                continue
                
            confidence_score = self._calculate_id_confidence(pipeline.train_data, col)
            
            if confidence_score > 30:
                potential_ids.append((col, confidence_score))
        
        if not potential_ids:
            self.ui.print("🔑 No potential ID columns detected")
            return
        
        potential_ids.sort(key=lambda x: x[1], reverse=True)
        
        self.ui.print(f"\n🔑 ANALYZING COLUMNS FOR ID DETECTION...")
        self.ui.print("="*50)
        
        for col, confidence in potential_ids:
            self._show_column_analysis(pipeline.train_data, col, confidence, pipeline.target_column)
        
        configured_columns = self._configure_columns_interactive(pipeline.train_data, potential_ids, pipeline.target_column)
        
        pipeline.id_columns = configured_columns['exclude']
        
        if pipeline.id_columns:
            self.ui.print(f"\n✅ Final ID Column Configuration:")
            self.ui.print(f"  • Excluded from modeling: {', '.join(pipeline.id_columns)}")
            
        feature_count = len(pipeline.train_data.columns) - len(pipeline.id_columns) - 1
        self.ui.print(f"  • Features for modeling: {feature_count}")

    def _calculate_id_confidence(self, df, column):
        """Calculate confidence score that a column is an ID column."""
        confidence = 0
        
        # Uniqueness ratio
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio == 1.0:
            confidence += 70
        elif unique_ratio > 0.95:
            confidence += 50
        elif unique_ratio > 0.8:
            confidence += 20
        
        # Naming patterns
        id_patterns = ['id', 'ID', 'Id', 'key', 'KEY', 'index', 'INDEX', 'number', 'num', 'code']
        column_lower = column.lower()
        
        for pattern in id_patterns:
            if pattern.lower() in column_lower:
                if column_lower == pattern.lower():
                    confidence += 30
                elif column_lower.endswith(pattern.lower()) or column_lower.startswith(pattern.lower()):
                    confidence += 25
                else:
                    confidence += 15
                break
        
        # Data type analysis
        if df[column].dtype in ['int64', 'int32', 'object']:
            confidence += 10
        
        # Sequential pattern detection
        if df[column].dtype in ['int64', 'int32']:
            try:
                sorted_values = df[column].dropna().sort_values()
                if len(sorted_values) > 1:
                    differences = sorted_values.diff().dropna()
                    if (differences == 1).all():
                        confidence += 20
                    elif differences.std() < differences.mean() * 0.1:
                        confidence += 10
            except:
                pass
        
        return min(100, confidence)