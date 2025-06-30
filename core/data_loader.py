"""
Simplified data loading module
Handles CSV and Excel files with basic validation
"""

import pandas as pd
import hashlib
from pathlib import Path
from typing import Optional


class DataLoader:
    """
    Simplified data loader with basic validation.
    """
    
    def __init__(self, ui):
        """
        Initialize data loader.
        
        Args:
            ui: Terminal UI instance
        """
        self.ui = ui
    
    def load_data(self, train_path: str, test_path: Optional[str] = None, 
                target_column: Optional[str] = None, pipeline=None) -> bool:
        """
        Load data with enhanced ID column detection and validation.
        
        Args:
            train_path: Path to training data file
            test_path: Path to test data file (optional)
            target_column: Name of target column
            pipeline: Reference to main pipeline object
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
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
            
            # Detect ID columns
            id_columns = self._detect_id_columns(train_data)
            if id_columns:
                self.ui.print(f"🔑 Detected potential ID columns: {id_columns}")
            
            # Generate dataset hash for tracking
            dataset_hash = self._generate_dataset_hash(train_data)
            
            # Load test data if provided
            test_data = None
            if test_path and Path(test_path).exists():
                test_data = self._load_file(test_path)
                if test_data is not None:
                    self.ui.print(f"✅ Test data loaded: {test_data.shape}")
                    
                    # Verify ID columns match between train and test
                    test_id_columns = self._detect_id_columns(test_data)
                    if set(id_columns) != set(test_id_columns):
                        self.ui.print("⚠️ Warning: ID columns differ between train and test data")
            
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
                pipeline.id_columns = id_columns
                if target_column:
                    pipeline.target_column = target_column
                
                # Auto-detect problem type if not specified
                if pipeline.problem_type == 'auto':
                    pipeline._detect_problem_type()
                
                # Identify feature types
                pipeline._identify_feature_types()
            
            return True
            
        except Exception as e:
            self.ui.print(f"❌ Error loading data: {str(e)}")
            return False   
        
    def _load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame if successful, None otherwise
        """
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
        """
        Generate a hash for the dataset for tracking purposes.
        
        Args:
            df: DataFrame to hash
            
        Returns:
            str: Hexadecimal hash string
        """
        try:
            # Create a string representation of the data structure
            data_signature = f"{df.shape}_{list(df.columns)}_{df.dtypes.to_string()}"
            
            # Generate hash
            return hashlib.md5(data_signature.encode()).hexdigest()[:16]
            
        except Exception:
            # Fallback to simple hash
            return hashlib.md5(str(df.shape).encode()).hexdigest()[:16]
        
    def _detect_id_columns(self, df):
        """
        Automatically detect ID-like columns in the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of detected ID column names
        """
        id_columns = []
        
        for col in df.columns:
            # Check for common ID naming patterns
            if col.lower() in ['id', 'index', 'key']:
                id_columns.append(col)
            elif col.lower().endswith('_id') or col.lower().endswith('id'):
                id_columns.append(col)
            elif df[col].nunique() == len(df) and df[col].dtype in ['int64', 'object']:
                # Likely unique identifier
                id_columns.append(col)
        
        return id_columns