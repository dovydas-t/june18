# Updated config.py - Enhanced configuration with manual preprocessing settings

"""
Enhanced Configuration file for the ML Pipeline v2.1
Contains all constants, settings, and global configurations
"""

import warnings
warnings.filterwarnings('ignore')

# Version Information
VERSION = "3.1"
APP_NAME = "Enhanced Machine Learning Pipeline"

# Global Configuration
RANDOM_STATE = 42
DATABASE_PATH = "ml_experiments.db"
DEFAULT_RANDOM_STATE = RANDOM_STATE

# Default Pipeline Settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_REMOVE_OUTLIERS = True

# Manual Preprocessing Settings
PREPROCESSING_OPTIONS = {
    'scaling_methods': {
        'standard': 'StandardScaler (mean=0, std=1)',
        'minmax': 'MinMaxScaler (0-1 range)', 
        'robust': 'RobustScaler (median-based)',
        'none': 'No scaling'
    },
    'missing_strategies': {
        'auto': 'Automatic (median/mode)',
        'mean': 'Mean imputation',
        'median': 'Median imputation',
        'most_frequent': 'Mode imputation',
        'constant': 'Constant value',
        'drop': 'Drop columns/rows'
    },
    'outlier_methods': {
        'iqr': 'Interquartile Range',
        'zscore': 'Z-score method',
        'isolation': 'Isolation Forest',
        'none': 'No outlier removal'
    }
}

# Outlier Detection Thresholds
OUTLIER_THRESHOLDS = {
    'iqr_default': 1.5,
    'zscore_default': 3.0,
    'isolation_contamination': 0.1
}

# Optional Library Availability
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.layout import Layout
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    from rich.tree import Tree
    from rich.markup import escape
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available. Enhanced UI features disabled.")
    print("Install with: pip install rich")

# Model Categories with Enhanced Descriptions
MODEL_CATEGORIES = {
    'instance_based': {
        'models': ['KNN'],
        'description': 'Memory-based learning using similarity',
        'best_for': ['Small datasets', 'Non-linear patterns', 'Recommendation systems']
    },
    'kernel_based': {
        'models': ['SVM'],
        'description': 'Support vector machines with kernel tricks',
        'best_for': ['High-dimensional data', 'Text classification', 'Complex boundaries']
    },
    'tree_based': {
        'models': ['Decision Tree'],
        'description': 'Rule-based decision making',
        'best_for': ['Interpretable models', 'Mixed data types', 'Feature selection']
    },
    'linear': {
        'models': ['Linear Model'],
        'description': 'Linear relationships and fast baselines',
        'best_for': ['Linear relationships', 'Fast inference', 'Feature importance']
    },
    'ensemble': {
        'models': ['Random Forest', 'Extra Trees', 'Bagging'],
        'description': 'Multiple models for robust predictions',
        'best_for': ['General purpose', 'Reduced overfitting', 'Feature importance']
    },
    'boosting': {
        'models': ['AdaBoost', 'Hist Gradient Boosting', 'XGBoost', 'LightGBM'],
        'description': 'Sequential learning with error correction',
        'best_for': ['High performance', 'Structured data', 'Competitions']
    }
}

# Enhanced Model Configuration
DEFAULT_MODELS = {
    'KNN': True,
    'SVM': True,
    'Decision Tree': True,
    'Linear Model': True,
    'Random Forest': True,
    'Extra Trees': True,
    'Bagging': True,
    'AdaBoost': True,
    'Hist Gradient Boosting': True,
    'XGBoost': XGBOOST_AVAILABLE,
    'LightGBM': LIGHTGBM_AVAILABLE
}

# Model Performance Expectations
MODEL_PERFORMANCE_GUIDE = {
    'fast_training': ['Linear Model', 'KNN', 'Decision Tree'],
    'high_accuracy': ['Random Forest', 'XGBoost', 'LightGBM'],
    'interpretable': ['Decision Tree', 'Linear Model'],
    'handles_mixed_data': ['Random Forest', 'XGBoost', 'Decision Tree'],
    'good_for_small_data': ['KNN', 'Decision Tree', 'SVM'],
    'good_for_large_data': ['Hist Gradient Boosting', 'LightGBM']
}

# File Patterns and Support
SUPPORTED_FILE_FORMATS = ['.csv', '.xlsx', '.xls', '.tsv']
BACKUP_FILE_PATTERN = "ml_experiments_backup_%Y%m%d_%H%M%S.db"

# ID Column Detection Patterns
ID_COLUMN_PATTERNS = [
    'id', 'ID', 'Id', 'iD',
    'index', 'INDEX', 'Index',
    'key', 'KEY', 'Key',
    'student_id', 'studentid', 'StudentID',
    'user_id', 'userid', 'UserID',
    'customer_id', 'customerid', 'CustomerID'
]

# UI Settings
MAX_DISPLAY_ROWS = 20
MAX_DISPLAY_COLS = 10
TABLE_WIDTH_LIMITS = {
    'small': 15,
    'medium': 25,
    'large': 40,
    'extra_large': 60
}

# Enhanced Performance Thresholds
QUALITY_THRESHOLDS = {
    'excellent': 90,
    'good': 75,
    'fair': 60,
    'poor': 40,
    'critical': 20
}

# Performance Benchmarks by Problem Type
PERFORMANCE_BENCHMARKS = {
    'classification': {
        'accuracy': {'excellent': 0.95, 'good': 0.85, 'fair': 0.75, 'poor': 0.65},
        'f1_score': {'excellent': 0.90, 'good': 0.80, 'fair': 0.70, 'poor': 0.60}
    },
    'regression': {
        'r2': {'excellent': 0.90, 'good': 0.75, 'fair': 0.60, 'poor': 0.45},
        'rmse': 'lower_is_better'  # Relative to target range
    }
}

# Memory Limits (in MB)
MEMORY_LIMITS = {
    'small_dataset': 100,
    'medium_dataset': 500,
    'large_dataset': 2000,
    'warning_threshold': 1000
}

# Enhanced Cross-validation Settings
CV_SETTINGS = {
    'min_folds': 3,
    'max_folds': 10,
    'default_folds': 5,
    'stratify_threshold': 10  # Minimum samples per class for stratification
}

# Enhanced Hyperparameter Tuning Settings
TUNING_SETTINGS = {
    'max_iter': 100,
    'n_jobs': -1,
    'verbose': 0,
    'cv_folds_for_tuning': 3,
    'scoring_timeout': 300,  # 5 minutes max per model
    'param_combinations_limit': 50
}

# Enhanced Data Quality Thresholds
DATA_QUALITY = {
    'high_missing_threshold': 0.5,     # 50% missing = major issue
    'moderate_missing_threshold': 0.05, # 5% missing = minor issue
    'high_cardinality_threshold': 50,   # >50 unique values = high cardinality
    'outlier_threshold': 0.05,          # >5% outliers = needs attention
    'duplicate_threshold': 0.1,         # >10% duplicates = issue
    'constant_threshold': 0.95,         # >95% same value = nearly constant
    'correlation_threshold': 0.95       # >95% correlation = highly correlated
}

# Enhanced Visualization Settings
VIZ_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'color_palette': 'Set2',
    'max_categories': 20,
    'correlation_heatmap_size': (10, 8),
    'feature_importance_top_n': 15
}

# Database Settings with Enhanced Features
DB_SETTINGS = {
    'timeout': 30.0,
    'check_same_thread': False,
    'isolation_level': None,
    'auto_backup': True,
    'backup_frequency': 'daily',
    'max_backup_files': 7
}

# Enhanced Logging Settings
LOG_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'ml_pipeline.log',
    'max_file_size': 10485760,  # 10MB
    'backup_count': 3
}

# Export Settings with More Options
EXPORT_SETTINGS = {
    'default_submission_filename': 'submission.csv',
    'default_results_filename': 'ml_experiment_results.csv',
    'include_metadata': True,
    'include_preprocessing_info': True,
    'include_hyperparameters': True,
    'datetime_format': '%Y%m%d_%H%M%S'
}

# User Experience Settings
UX_SETTINGS = {
    'confirmation_timeout': 30,  # seconds
    'auto_save_interval': 300,   # 5 minutes
    'progress_update_frequency': 1,  # seconds
    'max_menu_items': 10,
    'default_page_size': 20
}

# Feature Engineering Options
FEATURE_ENGINEERING = {
    'polynomial_degree_max': 3,
    'interaction_features': True,
    'log_transform_skew_threshold': 2.0,
    'binning_methods': ['equal_width', 'equal_frequency', 'kmeans'],
    'text_vectorization_max_features': 1000
}

# Model Complexity Guidelines
MODEL_COMPLEXITY = {
    'simple': ['Linear Model', 'KNN', 'Decision Tree'],
    'moderate': ['Random Forest', 'SVM', 'Extra Trees', 'Bagging'],
    'complex': ['XGBoost', 'LightGBM', 'AdaBoost', 'Hist Gradient Boosting']
}

# Resource Usage Guidelines
RESOURCE_GUIDELINES = {
    'memory_efficient': ['Linear Model', 'KNN', 'Decision Tree'],
    'cpu_intensive': ['Random Forest', 'Extra Trees', 'Bagging'],
    'gpu_capable': ['XGBoost', 'LightGBM'],
    'parallel_training': ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM']
}

# Error Messages and Help
ERROR_MESSAGES = {
    'file_not_found': "File not found. Please check the file path and try again.",
    'invalid_target': "Target column not found. Please verify the column name.",
    'insufficient_data': "Dataset too small for reliable training. Consider getting more data.",
    'memory_error': "Not enough memory. Try reducing dataset size or using simpler models.",
    'missing_dependencies': "Required packages not installed. Run: pip install -r requirements.txt"
}