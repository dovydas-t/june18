import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split

class DataOperations:
    """Centralized data operations and validation."""
    
    def __init__(self, ui, db=None):
        self.ui = ui
        self.db = db
        self.dataset_cache = {}  # Cache for dataset paths

    def get_existing_datasets(self) -> List[Dict]:
        """Get existing datasets from database with path cleaning."""
        if not self.db:
            return []
        
        try:
            experiments = self.db.get_experiment_history(20)
            datasets = []
            seen_datasets = set()
            
            for exp in experiments:
                dataset_key = f"{exp['dataset_name']}_{exp['problem_type']}"
                if dataset_key not in seen_datasets:
                    # Clean dataset name from path
                    clean_name = self._clean_dataset_name(exp['dataset_name'])
                    datasets.append({
                        'name': clean_name,
                        'original_name': exp['dataset_name'],
                        'experiment_name': exp['experiment_name'],
                        'problem_type': exp['problem_type'],
                        'n_samples': exp.get('n_samples', 'Unknown'),
                        'dataset_hash': exp.get('dataset_hash', '')
                    })
                    seen_datasets.add(dataset_key)
            
            return datasets
        except Exception:
            return []
    
    def _clean_dataset_name(self, dataset_name: str) -> str:
        """Clean dataset name from folder paths."""
        from pathlib import Path
        return Path(dataset_name).stem
    
    def clear_dataset_cache(self):
        """Clear previous dataset choices."""
        self.dataset_cache.clear()
        self.ui.print("✅ Previous dataset choices cleared")
    
    def load_data(self, train_path: str, test_path: str = None, target_column: str = None, pipeline=None) -> bool:
        """Enhanced data loading with validation and hash generation."""
        try:
            # Check if file exists
            if not Path(train_path).exists():
                self.ui.print(f"❌ Training file not found: {train_path}")
                return False
            
            # Load training data
            train_data = self._load_file(train_path)
            if train_data is None:
                return False
            
            self.ui.print(f"✅ Training data loaded: {train_data.shape}")
            
            # Generate dataset hash for tracking
            dataset_hash = self._generate_dataset_hash(train_data)
            
            # Load test data if provided
            test_data = None
            if test_path and Path(test_path).exists():
                test_data = self._load_file(test_path)
                if test_data is not None:
                    self.ui.print(f"✅ Test data loaded: {test_data.shape}")
            
            # Validate target column
            if target_column and target_column not in train_data.columns:
                self.ui.print(f"❌ Target column '{target_column}' not found!")
                self.ui.print(f"Available columns: {list(train_data.columns)}")
                return False
            
            # Update pipeline with loaded data
            if pipeline:
                pipeline.train_data = train_data
                pipeline.test_data = test_data
                pipeline.dataset_hash = dataset_hash
                if target_column:
                    pipeline.target_column = target_column
                
                # Auto-detect problem type if not specified
                if pipeline.problem_type == 'auto':
                    self.detect_problem_type(pipeline)
                
                # Identify feature types
                self.identify_feature_types(pipeline)
            
            return True
            
        except Exception as e:
            self.ui.print(f"❌ Error loading data: {str(e)}")
            return False
    
    def _load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load file based on extension."""
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                self.ui.print(f"❌ Unsupported file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            self.ui.print(f"❌ Error loading {file_path}: {str(e)}")
            return None
    
    def _generate_dataset_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash for the dataset for tracking purposes."""
        try:
            data_signature = f"{df.shape}_{list(df.columns)}_{df.dtypes.to_string()}"
            return hashlib.md5(data_signature.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.md5(str(df.shape).encode()).hexdigest()[:16]
    
    def show_dataset_overview(self, df: pd.DataFrame, target_column: str = None, test_data: pd.DataFrame = None):
        """Show comprehensive dataset overview."""
        self.ui.print("\n📊 DATASET OVERVIEW")
        self.ui.print("-" * 30)
        self.ui.print(f"Shape: {df.shape}")
        if target_column:
            self.ui.print(f"Target Column: {target_column}")
        self.ui.print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        self.ui.print(f"Missing Values: {df.isnull().sum().sum():,}")
        self.ui.print(f"Duplicate Rows: {df.duplicated().sum():,}")
        
        if test_data is not None:
            self.ui.print(f"Test Samples: {len(test_data):,}")
    
    def detect_problem_type(self, pipeline):
        """Enhanced problem type detection with manual override option."""
        if not pipeline.target_column or pipeline.target_column not in pipeline.train_data.columns:
            self.ui.print(f"❌ Target column '{pipeline.target_column}' not found")
            
            # Show available columns and let user choose
            self.ui.print("Available columns:")
            for i, col in enumerate(pipeline.train_data.columns, 1):
                self.ui.print(f"  {i}. {col}")
            
            try:
                choice = int(self.ui.input(f"Select target column (1-{len(pipeline.train_data.columns)})", default="1"))
                if 1 <= choice <= len(pipeline.train_data.columns):
                    pipeline.target_column = pipeline.train_data.columns[choice-1]
                    self.ui.print(f"✅ Target column set to: {pipeline.target_column}")
                else:
                    self.ui.print("❌ Invalid choice")
                    return
            except ValueError:
                self.ui.print("❌ Invalid input")
                return
        
        target = pipeline.train_data[pipeline.target_column]
        target_clean = target.dropna()
        
        if len(target_clean) == 0:
            self.ui.print("❌ Target column contains only missing values")
            return
        
        # Auto-detect problem type
        if pd.api.types.is_numeric_dtype(target_clean):
            unique_values = target_clean.nunique()
            total_values = len(target_clean)
            unique_ratio = unique_values / total_values
            
            if unique_values <= 2:
                detected_type = 'classification'
                pipeline.is_binary_classification = True
                explanation = f"Binary classification detected (2 unique values)"
            elif unique_values <= 20 and unique_ratio < 0.05:
                detected_type = 'classification'
                explanation = f"Multi-class classification detected ({unique_values} classes)"
            elif all(target_clean == target_clean.astype(int)):
                if unique_values <= 50:
                    detected_type = 'classification'
                    explanation = f"Classification detected ({unique_values} integer classes)"
                else:
                    detected_type = 'regression'
                    explanation = f"Regression detected (many integer values)"
            else:
                detected_type = 'regression'
                explanation = f"Regression detected (continuous values)"
        else:
            detected_type = 'classification'
            unique_values = target_clean.nunique()
            if unique_values == 2:
                pipeline.is_binary_classification = True
                explanation = f"Binary classification detected (2 unique classes)"
            else:
                explanation = f"Multi-class classification detected ({unique_values} classes)"
        
        # Show detection result and ask for confirmation
        self.ui.print(f"🎯 {explanation}")
        
        if pipeline.problem_type == 'auto':
            confirm = self.ui.confirm(f"Accept detected problem type: {detected_type}?", default=True)
            if confirm:
                pipeline.problem_type = detected_type
            else:
                # Manual selection
                self.ui.print("Manual problem type selection:")
                self.ui.print("1. Classification")
                self.ui.print("2. Regression")
                
                choice = self.ui.input("Select problem type (1-2)", default="1")
                if choice == "2":
                    pipeline.problem_type = 'regression'
                else:
                    pipeline.problem_type = 'classification'
            
            self.ui.print(f"✅ Problem type set to: {pipeline.problem_type}")

    def identify_feature_types(self, pipeline):
        """Identify and categorize feature types."""
        exclude_cols = ['Id', 'id', 'ID', pipeline.target_column]
        all_columns = [col for col in pipeline.train_data.columns if col not in exclude_cols]
        
        pipeline.numerical_columns = []
        pipeline.categorical_columns = []
        
        for col in all_columns:
            if pd.api.types.is_numeric_dtype(pipeline.train_data[col]):
                pipeline.numerical_columns.append(col)
            else:
                pipeline.categorical_columns.append(col)
        
        pipeline.feature_columns = all_columns
        
        self.ui.print(f"📊 Identified {len(pipeline.numerical_columns)} numerical and {len(pipeline.categorical_columns)} categorical features")

    def get_existing_datasets(self) -> List[Dict]:
        """Get existing datasets from database."""
        if not self.db:
            return []
        
        try:
            experiments = self.db.get_experiment_history(20)
            datasets = []
            seen_datasets = set()
            
            for exp in experiments:
                dataset_key = f"{exp['dataset_name']}_{exp['problem_type']}"
                if dataset_key not in seen_datasets:
                    datasets.append({
                        'name': exp['dataset_name'],
                        'experiment_name': exp['experiment_name'],
                        'problem_type': exp['problem_type'],
                        'n_samples': exp.get('n_samples', 'Unknown'),
                        'dataset_hash': exp.get('dataset_hash', '')
                    })
                    seen_datasets.add(dataset_key)
            
            return datasets
        except Exception:
            return []