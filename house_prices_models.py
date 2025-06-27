"""
🏠 Advanced House Price Prediction ML Pipeline
===============================================
Models Used:
1. K-Nearest Neighbors (KNN)
2. Support Vector Regression (SVR)
3. Decision Tree Regression
4. Linear Regression

Evaluation Metric: RMSE between log(predicted) and log(actual) prices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class HousePricePredictor:
    """
    Comprehensive House Price Prediction Pipeline
    
    This class handles data loading, preprocessing, model training,
    evaluation, and visualization for house price prediction.
    """
    
    def __init__(self, remove_outliers=True):
        """
        Initialize the predictor with empty containers for data and models
        
        Args:
            remove_outliers (bool): Whether to remove outliers from training data
        """
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_submission = None
        self.scaler = None
        self.label_encoders = {}
        self.models = {}
        self.predictions = {}
        self.scores = {}
        self.remove_outliers = remove_outliers
        
    def load_data(self, train_path='houses.csv', test_path='test.csv'):
        """
        Load training and test datasets
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data
        """
        print("🔄 Loading datasets...")
        try:
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            print(f"✅ Training data loaded: {self.train_data.shape}")
            print(f"✅ Test data loaded: {self.test_data.shape}")
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return False
        return True
    
    def analyze_missing_values(self, df, dataset_name):
        """
        Analyze and visualize missing values in dataset
        
        Args:
            df (DataFrame): Dataset to analyze
            dataset_name (str): Name for the dataset (for plotting)
        """
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        missing_percent = (missing_data / len(df)) * 100
        
        if len(missing_data) > 0:
            print(f"\n📊 Missing values in {dataset_name}:")
            missing_df = pd.DataFrame({
                'Missing Count': missing_data,
                'Missing Percentage': missing_percent.round(2)  # Round to 2 decimal places
            })
            
            # Format the output nicely
            print(f"{'Column':<15} {'Missing Count':<15} {'Missing %':<12}")
            print("-" * 45)
            for idx, (column, row) in enumerate(missing_df.head(10).iterrows()):
                print(f"{column:<15} {row['Missing Count']:<15} {row['Missing Percentage']:<11.2f}%")
        else:
            print(f"✅ No missing values in {dataset_name}")
            
        return missing_data
    
    def handle_missing_values(self, df):
        """
        Handle missing values using domain knowledge and statistical methods
        
        Args:
            df (DataFrame): Dataset to process
            
        Returns:
            DataFrame: Processed dataset
        """
        df = df.copy()
        
        # Categorical features where NaN means "None" or "No feature"
        categorical_none_features = [
            'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
        ]
        
        # Fill categorical "None" features
        for feature in categorical_none_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna('None')
        
        # Numerical features where NaN means 0
        numerical_zero_features = [
            'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
        ]
        
        # Fill numerical zero features
        for feature in numerical_zero_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)
        
        # Handle LotFrontage (use median by neighborhood)
        if 'LotFrontage' in df.columns:
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Handle GarageYrBlt (use YearBuilt when missing)
        if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        
        # Handle remaining categorical variables with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_value)
        
        # Handle remaining numerical variables with median
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_features(self, df):
        """
        Create new features based on domain knowledge
        
        Args:
            df (DataFrame): Dataset to enhance
            
        Returns:
            DataFrame: Enhanced dataset
        """
        df = df.copy()
        
        # Total square footage
        df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
        
        # Total bathrooms
        df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
        
        # House age
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
        
        # Porch area
        df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
        
        # Has garage, basement, fireplace, pool
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        
        # Quality scores
        df['OverallScore'] = df['OverallQual'] * df['OverallCond']
        
        # Price per square foot (only for training data)
        if 'SalePrice' in df.columns:
            df['PricePerSF'] = df['SalePrice'] / df['TotalSF']
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
        """
        Encode categorical features using Label Encoding
        
        Args:
            df (DataFrame): Dataset to encode
            is_training (bool): Whether this is training data
            
        Returns:
            DataFrame: Encoded dataset
        """
        df = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = categorical_columns.drop(['Id'], errors='ignore')
        
        for col in categorical_columns:
            if is_training:
                # Fit encoder on training data
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Transform using fitted encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    unique_values = set(le.classes_)
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(lambda x: x if x in unique_values else le.classes_[0])
                    df[col] = le.transform(df[col])
        
        return df
    
    def preprocess_data(self):
        """
        Main preprocessing pipeline for both training and test data
        """
        print("\n🔧 Starting data preprocessing...")
        
        # Analyze missing values
        self.analyze_missing_values(self.train_data, "Training Data")
        self.analyze_missing_values(self.test_data, "Test Data")
        
        # Handle missing values
        print("🔄 Handling missing values...")
        self.train_data = self.handle_missing_values(self.train_data)
        self.test_data = self.handle_missing_values(self.test_data)
        
        # Create new features
        print("🔄 Creating new features...")
        self.train_data = self.create_features(self.train_data)
        self.test_data = self.create_features(self.test_data)
        
        # Remove outliers from training data (optional)
        if self.remove_outliers and 'SalePrice' in self.train_data.columns:
            Q1 = self.train_data['SalePrice'].quantile(0.25)
            Q3 = self.train_data['SalePrice'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            original_size = len(self.train_data)
            self.train_data = self.train_data[
                (self.train_data['SalePrice'] >= lower_bound) & 
                (self.train_data['SalePrice'] <= upper_bound)
            ]
            print(f"🧹 Removed {original_size - len(self.train_data)} outliers from training data")
        elif not self.remove_outliers:
            print("⚠️  Keeping all data points (including outliers)")
        else:
            print("ℹ️  No outlier removal needed (SalePrice not found)")  
        
        # Prepare features and target - ensure consistency between train and test
        # Get common columns between train and test (excluding Id and SalePrice)
        train_cols = set(self.train_data.columns) - {'Id', 'SalePrice', 'PricePerSF'}
        test_cols = set(self.test_data.columns) - {'Id'}
        feature_columns = list(train_cols.intersection(test_cols))
        
        print(f"🔄 Using {len(feature_columns)} common features between train and test sets")
        
        # Encode categorical features
        print("🔄 Encoding categorical features...")
        train_encoded = self.encode_categorical_features(
            self.train_data[feature_columns + ['SalePrice']], is_training=True
        )
        test_encoded = self.encode_categorical_features(
            self.test_data[feature_columns], is_training=False
        )
        
        # Split features and target
        self.X_train = train_encoded[feature_columns]
        self.y_train = train_encoded['SalePrice']
        self.X_submission = test_encoded[feature_columns]
        
        # Create train/validation split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train, self.y_train, 
            test_size=0.2, 
            random_state=RANDOM_STATE,
            stratify=pd.qcut(self.y_train, q=5, duplicates='drop')  # Stratify by price quintiles
        )
        
        # Scale features
        print("🔄 Scaling features...")
        self.scaler = StandardScaler()  # Standard normalization (mean=0, std=1)
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        self.X_submission = pd.DataFrame(
            self.scaler.transform(self.X_submission),
            columns=self.X_submission.columns,
            index=self.X_submission.index
        )
        
        print(f"✅ Preprocessing complete!")
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Validation set: {self.X_test.shape}")
        print(f"   Submission set: {self.X_submission.shape}")
    
    def calculate_rmse_log(self, y_true, y_pred):
        """
        Calculate RMSE between log of actual and predicted values
        
        Args:
            y_true (array): True values
            y_pred (array): Predicted values
            
        Returns:
            float: RMSE of log values
        """
        # Ensure positive values for log transformation
        y_true = np.maximum(y_true, 1)
        y_pred = np.maximum(y_pred, 1)
        
        return np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))
    
    def train_knn_model(self):
        """
        🎯 K-NEAREST NEIGHBORS (KNN) REGRESSOR
        =====================================
        
        How it works:
        - Finds K most similar houses to the target house
        - Predicts price as average/weighted average of K neighbors
        - "Tell me who your neighbors are, and I'll tell you your price"
        
        Hyperparameters explained:
        - n_neighbors: How many similar houses to consider (3-20)
                      • Low values (3-5): More sensitive to local patterns, can overfit
                      • High values (15-20): Smoother predictions, less sensitive to noise
        - weights: How to weight neighbor influences
                  • 'uniform': All neighbors equally important
                  • 'distance': Closer neighbors have more influence (usually better)
        - metric: How to measure "similarity" between houses
                 • 'euclidean': Straight-line distance √((x₁-x₂)² + (y₁-y₂)²)
                 • 'manhattan': City-block distance |x₁-x₂| + |y₁-y₂| 
                 • 'minkowski': Generalized distance (p=1→manhattan, p=2→euclidean)
        - p: Power parameter for minkowski metric
            • p=1: Manhattan distance
            • p=2: Euclidean distance
        
        Best for: Local patterns, irregular price distributions, non-linear relationships
        Struggles with: High dimensions, irrelevant features, sparse data
        
        Returns:
            tuple: (best_model, best_params)
        """
        print("🔄 Training KNN Regressor...")
        
        # Define hyperparameter grid
        knn_params = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],      # Number of neighbors to consider
            'weights': ['uniform', 'distance'],            # Equal vs distance-based weighting  
            'metric': ['euclidean', 'manhattan', 'minkowski'], # Distance calculation method
            'p': [1, 2]                                   # Power for minkowski (1=manhattan, 2=euclidean)
        }
        
        # Create model and grid search
        knn = KNeighborsRegressor()
        knn_grid = GridSearchCV(
            knn, knn_params, 
            cv=5,                              # 5-fold cross-validation
            scoring='neg_mean_squared_error',  # Minimize MSE
            n_jobs=-1,                         # Use all CPU cores
            verbose=0                          # Suppress detailed output
        )
        
        # Fit model
        knn_grid.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Best KNN params: {knn_grid.best_params_}")
        print(f"   ✅ Best KNN CV score: {-knn_grid.best_score_:.2f}")
        
        return knn_grid.best_estimator_, knn_grid.best_params_
    
    def train_svr_model(self):
        """
        🎯 SUPPORT VECTOR REGRESSION (SVR)
        ==================================
        
        How it works:
        - Creates a "tube" around the best-fit line/curve
        - Points inside the tube don't contribute to loss (epsilon-insensitive)
        - Finds optimal balance between fitting data and model simplicity
        - Uses kernel trick to capture non-linear relationships
        
        Hyperparameters explained:
        - C: Regularization strength (0.1-1000)
            • Low C (0.1-1): More regularization, simpler model, may underfit
            • High C (100-1000): Less regularization, complex model, may overfit
            • Sweet spot often around 10-100
        - kernel: How to transform features for non-linear relationships
                 • 'linear': No transformation, finds best straight line
                 • 'rbf': Radial Basis Function, good for non-linear patterns
                 • 'poly': Polynomial features, captures interactions
        - gamma: Kernel coefficient (for rbf/poly kernels)
                • 'scale': 1/(n_features * X.var()) - auto-adjusted
                • 'auto': 1/n_features - simpler auto-adjustment  
                • Low values (0.001): Broad influence, smoother decision boundary
                • High values (1): Narrow influence, more complex boundary
        - epsilon: Size of epsilon-tube (0.01-0.2)
                  • Small epsilon: Tight fit, more support vectors
                  • Large epsilon: Loose fit, fewer support vectors
        
        Best for: High-dimensional data, non-linear relationships, robust to outliers
        Struggles with: Very large datasets, choosing right kernel
        
        Returns:
            tuple: (best_model, best_params)
        """
        print("🔄 Training SVR...")
        
        # Define hyperparameter grid
        svr_params = {
            'C': [0.1, 1, 10, 100, 1000],                    # Regularization strength
            'kernel': ['rbf', 'linear', 'poly'],             # Kernel type for non-linearity
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], # Kernel coefficient
            'epsilon': [0.01, 0.1, 0.2]                      # Epsilon-tube width
        }
        
        # Create model and grid search
        svr = SVR()
        svr_grid = GridSearchCV(
            svr, svr_params, 
            cv=3,                              # Reduced CV for SVR (computationally expensive)
            scoring='neg_mean_squared_error',  # Minimize MSE
            n_jobs=-1,                         # Use all CPU cores
            verbose=0                          # Suppress detailed output
        )
        
        # Fit model
        svr_grid.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Best SVR params: {svr_grid.best_params_}")
        print(f"   ✅ Best SVR CV score: {-svr_grid.best_score_:.2f}")
        
        return svr_grid.best_estimator_, svr_grid.best_params_
    
    def train_decision_tree_model(self):
        """
        🎯 DECISION TREE REGRESSION (Basic)
        ===================================
        
        How it works:
        - Creates a tree of if/else questions to predict house prices
        - Each node splits data based on one feature
        - Leaves contain final price predictions
        - Example: "If GrLivArea > 2000 AND Neighborhood = 'NridgHt' → $350k"
        
        Hyperparameters explained:
        - max_depth: Maximum tree depth (None, 5-30)
                    • None: Tree grows until pure leaves (can overfit)
                    • Low values (5-10): Simple tree, may underfit
                    • High values (20-30): Complex tree, may overfit
        - min_samples_split: Minimum samples to split a node (2-20)
                           • Low values (2-5): More splits, complex tree
                           • High values (10-20): Fewer splits, simpler tree
        - min_samples_leaf: Minimum samples in each leaf (1-8)
                          • Low values (1-2): Detailed predictions, may overfit
                          • High values (4-8): Smoother predictions
        - max_features: Features to consider for each split
                       • 'sqrt': √(n_features) - good for reducing overfitting
                       • 'log2': log₂(n_features) - even more conservative
                       • None: All features - more expressive but may overfit
                       • 0.5-0.7: Fraction of features - balanced approach
        - criterion: How to measure split quality
                    • 'squared_error': Standard MSE-based splits
                    • 'friedman_mse': Friedman's improvement (often better)
        
        Best for: Interpretable models, capturing feature interactions, handling mixed data types
        Struggles with: Overfitting, instability (small data changes → different tree)
        
        Returns:
            tuple: (best_model, best_params)
        """
        print("🔄 Training Decision Tree...")
        
        # Define hyperparameter grid
        dt_params = {
            'max_depth': [None, 5, 10, 15, 20, 25, 30],       # Tree depth limit
            'min_samples_split': [2, 5, 10, 15, 20],          # Min samples to split node
            'min_samples_leaf': [1, 2, 4, 6, 8],              # Min samples in leaf
            'max_features': ['sqrt', 'log2', None, 0.5, 0.7], # Features per split
            'criterion': ['squared_error', 'friedman_mse']     # Split quality measure
        }
        
        # Create model and grid search
        dt = DecisionTreeRegressor(random_state=RANDOM_STATE)
        dt_grid = GridSearchCV(
            dt, dt_params, 
            cv=5,                              # 5-fold cross-validation
            scoring='neg_mean_squared_error',  # Minimize MSE
            n_jobs=-1,                         # Use all CPU cores
            verbose=0                          # Suppress detailed output
        )
        
        # Fit model
        dt_grid.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Best DT params: {dt_grid.best_params_}")
        print(f"   ✅ Best DT CV score: {-dt_grid.best_score_:.2f}")
        
        return dt_grid.best_estimator_, dt_grid.best_params_
    
    def train_improved_decision_tree_model(self):
        """
        🎯 IMPROVED DECISION TREE (Random Forest + Extra Trees)
        ======================================================
        
        RANDOM FOREST:
        How it works:
        - Trains multiple decision trees on different data subsets
        - Each tree uses random feature subsets (bootstrap aggregating)
        - Final prediction = average of all tree predictions
        - Reduces overfitting through ensemble wisdom
        
        Hyperparameters explained:
        - n_estimators: Number of trees in forest (50-200)
                       • More trees = better performance but slower training
                       • Usually 100-200 is good balance
        - max_depth: Individual tree depth (10-30, None)
        - min_samples_split: Minimum samples to split (2-10)
        - min_samples_leaf: Minimum samples in leaf (1-4)  
        - max_features: Features per tree ('sqrt', 'log2', None)
                       • 'sqrt': Good default for regression
                       • Reduces correlation between trees
        - bootstrap: Whether to use bootstrap sampling (True/False)
                    • True: Standard Random Forest
                    • False: Uses all data for each tree
        
        EXTRA TREES (Extremely Randomized Trees):
        - Like Random Forest but with extra randomization
        - Splits are completely random (not optimized)
        - Faster training, sometimes better generalization
        - Less prone to overfitting than standard Random Forest
        
        Best for: Robust predictions, handling overfitting, feature importance
        Struggles with: Interpretability (black box), memory usage with many trees
        
        Returns:
            dict: Dictionary with both Random Forest and Extra Trees models
        """
        print("🔄 Training Improved Decision Trees (Random Forest + Extra Trees)...")
        
        # Random Forest hyperparameters
        rf_params = {
            'n_estimators': [50, 100, 150, 200],              # Number of trees
            'max_depth': [10, 15, 20, 25, None],              # Tree depth
            'min_samples_split': [2, 5, 10],                  # Min samples to split
            'min_samples_leaf': [1, 2, 4],                    # Min samples in leaf
            'max_features': ['sqrt', 'log2', None],           # Features per tree
            'bootstrap': [True, False]                         # Bootstrap sampling
        }
        
        # Extra Trees hyperparameters (similar but typically less tuning needed)
        et_params = {
            'n_estimators': [50, 100, 150],                   # Fewer options for speed
            'max_depth': [15, 20, None],                      # 
            'min_samples_split': [2, 5],                      #
            'min_samples_leaf': [1, 2],                       #
            'max_features': ['sqrt', 'log2'],                 # 
            'bootstrap': [False]                               # Extra Trees typically don't use bootstrap
        }
        
        results = {}
        
        # Train Random Forest
        print("   🌲 Training Random Forest...")
        rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        rf_grid = GridSearchCV(
            rf, rf_params,
            cv=3,                              # Reduced CV for ensemble methods (slow)
            scoring='neg_mean_squared_error',
            n_jobs=1,                          # RF already uses n_jobs, avoid nested parallelism
            verbose=0
        )
        rf_grid.fit(self.X_train, self.y_train)
        results['Random Forest'] = (rf_grid.best_estimator_, rf_grid.best_params_)
        print(f"   ✅ Best RF params: {rf_grid.best_params_}")
        print(f"   ✅ Best RF CV score: {-rf_grid.best_score_:.2f}")
        
        # Train Extra Trees
        print("   🌳 Training Extra Trees...")
        et = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        et_grid = GridSearchCV(
            et, et_params,
            cv=3,                              # Reduced CV for ensemble methods (slow)
            scoring='neg_mean_squared_error',
            n_jobs=1,                          # ET already uses n_jobs
            verbose=0
        )
        et_grid.fit(self.X_train, self.y_train)
        results['Extra Trees'] = (et_grid.best_estimator_, et_grid.best_params_)
        print(f"   ✅ Best ET params: {et_grid.best_params_}")
        print(f"   ✅ Best ET CV score: {-et_grid.best_score_:.2f}")
        
        return results
    
    def train_linear_regression_model(self):
        """
        🎯 LINEAR REGRESSION (Ordinary Least Squares)
        =============================================
        
        How it works:
        - Finds the best straight line through the data
        - Minimizes sum of squared residuals (actual - predicted)²
        - Mathematical formula: Price = β₀ + β₁×Feature₁ + β₂×Feature₂ + ... + ε
        - Each β coefficient shows how much price changes per unit of that feature
        
        Key characteristics:
        - No hyperparameters to tune (except regularization variants)
        - Fast training and prediction
        - Interpretable coefficients
        - Assumes linear relationships between features and target
        - Sensitive to outliers and multicollinearity
        
        Assumptions:
        1. Linear relationship between features and target
        2. Independence of residuals  
        3. Homoscedasticity (constant variance of residuals)
        4. Normal distribution of residuals
        5. No multicollinearity between features
        
        Best for: Baseline model, interpretable results, linear relationships
        Struggles with: Non-linear patterns, outliers, correlated features
        
        Note: This is Ordinary Least Squares (OLS) - no regularization
              For regularized versions, consider Ridge/Lasso regression
        
        Returns:
            tuple: (model, params_info)
        """
        print("🔄 Training Linear Regression...")
        
        # Create and fit model (no hyperparameters for basic OLS)
        lr = LinearRegression(
            fit_intercept=True,    # Include y-intercept (β₀)
            copy_X=True,          # Don't modify original data
            n_jobs=None           # Use single core (fast enough for linear regression)
        )
        lr.fit(self.X_train, self.y_train)
        
        # Calculate training score for consistency with other models
        train_score = lr.score(self.X_train, self.y_train)
        
        # Get some model insights
        n_features = len(lr.coef_)
        max_coef = np.max(np.abs(lr.coef_))
        min_coef = np.min(np.abs(lr.coef_))
        
        print(f"   ✅ Linear Regression R² score: {train_score:.4f}")
        print(f"   ✅ Number of features: {n_features}")
        print(f"   ✅ Coefficient range: {min_coef:.4f} to {max_coef:.4f}")
        print(f"   ✅ Intercept: {lr.intercept_:.2f}")
        
        params_info = {
            "model": "LinearRegression - OLS", 
            "n_features": n_features,
            "train_r2": train_score,
            "intercept": lr.intercept_
        }
        
        return lr, params_info
    
    def train_models(self, use_improved_trees=True):
        """
        Train all machine learning models using separate functions
        
        Args:
            use_improved_trees (bool): Whether to use Random Forest/Extra Trees instead of basic Decision Tree
        """
        print("\n🤖 Training machine learning models...")
        print(f"🌲 Using improved trees: {use_improved_trees}")
        
        # Train each model using dedicated functions
        try:
            # 1. K-Nearest Neighbors
            knn_model, knn_params = self.train_knn_model()
            self.models['KNN'] = knn_model
            
            # 2. Support Vector Regression
            svr_model, svr_params = self.train_svr_model()
            self.models['SVR'] = svr_model
            
            # 3. Decision Tree variants
            if use_improved_trees:
                # Train improved tree models (Random Forest + Extra Trees)
                tree_results = self.train_improved_decision_tree_model()
                self.models['Random Forest'] = tree_results['Random Forest'][0]
                self.models['Extra Trees'] = tree_results['Extra Trees'][0]
                
                # Also train basic decision tree for comparison
                dt_model, dt_params = self.train_decision_tree_model()
                self.models['Decision Tree'] = dt_model
            else:
                # Train only basic decision tree
                dt_model, dt_params = self.train_decision_tree_model()
                self.models['Decision Tree'] = dt_model
            
            # 4. Linear Regression
            lr_model, lr_params = self.train_linear_regression_model()
            self.models['Linear Regression'] = lr_model
            
            print(f"\n✅ All models trained successfully!")
            print(f"📊 Total models trained: {len(self.models)}")
            print(f"🎯 Models: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"❌ Error during model training: {e}")
            raise
    
    def evaluate_models(self):
        """
        Evaluate all trained models on validation set
        """
        print("\n📊 Evaluating models on validation set...")
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            rmse_log = self.calculate_rmse_log(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Store results
            self.scores[name] = {
                'RMSE_Log': rmse_log,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            self.predictions[name] = y_pred
            
            print(f"\n{name} Results:")
            print(f"   RMSE (Log): {rmse_log:.4f}")
            print(f"   RMSE: {rmse:,.2f}")
            print(f"   MAE: {mae:,.2f}")
            print(f"   R²: {r2:.4f}")
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations for model comparison and analysis
        """
        print("\n📈 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Comparison - RMSE Log (Main Metric)
        ax1 = plt.subplot(2, 3, 1)
        models = list(self.scores.keys())
        rmse_logs = [self.scores[model]['RMSE_Log'] for model in models]
        
        bars = ax1.bar(models, rmse_logs, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Model Comparison - RMSE (Log Scale)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSE (Log)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_logs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. R² Score Comparison
        ax2 = plt.subplot(2, 3, 2)
        r2_scores = [self.scores[model]['R2'] for model in models]
        
        bars = ax2.bar(models, r2_scores,
                       color=['#FFD93D', '#6BCF7F', '#4D96FF', '#9B59B6'],
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, r2_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Actual vs Predicted - Best Model
        best_model = min(self.scores.keys(), key=lambda x: self.scores[x]['RMSE_Log'])
        ax3 = plt.subplot(2, 3, 3)
        
        y_pred_best = self.predictions[best_model]
        ax3.scatter(self.y_test, y_pred_best, alpha=0.6, c='#2E86AB', s=30)
        
        # Perfect prediction line
        min_val = min(min(self.y_test), min(y_pred_best))
        max_val = max(max(self.y_test), max(y_pred_best))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax3.set_xlabel('Actual Price ($)', fontsize=12)
        ax3.set_ylabel('Predicted Price ($)', fontsize=12)
        ax3.set_title(f'Actual vs Predicted - {best_model}', fontsize=14, fontweight='bold')
        ax3.legend()
        
        # Format axis labels
        ax3.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        
        # 4. Residuals Plot - Best Model
        ax4 = plt.subplot(2, 3, 4)
        residuals = self.y_test - y_pred_best
        
        ax4.scatter(y_pred_best, residuals, alpha=0.6, c='#F18F01', s=30)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Predicted Price ($)', fontsize=12)
        ax4.set_ylabel('Residuals ($)', fontsize=12)
        ax4.set_title(f'Residuals Plot - {best_model}', fontsize=14, fontweight='bold')
        ax4.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        
        # 5. Feature Importance (for Decision Tree)
        if 'Decision Tree' in self.models:
            ax5 = plt.subplot(2, 3, 5)
            dt_model = self.models['Decision Tree']
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': dt_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            bars = ax5.barh(range(len(feature_importance)), feature_importance['importance'],
                           color='#C44569', alpha=0.8, edgecolor='black', linewidth=1)
            ax5.set_yticks(range(len(feature_importance)))
            ax5.set_yticklabels(feature_importance['feature'])
            ax5.set_xlabel('Importance', fontsize=12)
            ax5.set_title('Top 10 Feature Importance - Decision Tree', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, feature_importance['importance'])):
                ax5.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        # 6. Model Performance Summary Table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create table data
        table_data = []
        for model in models:
            row = [
                model,
                f"{self.scores[model]['RMSE_Log']:.4f}",
                f"{self.scores[model]['R2']:.3f}",
                f"{self.scores[model]['MAE']:,.0f}"
            ]
            table_data.append(row)
        
        # Sort by RMSE_Log (lower is better)
        table_data.sort(key=lambda x: float(x[1]))
        
        table = ax6.table(cellText=table_data,
                         colLabels=['Model', 'RMSE (Log)', 'R²', 'MAE'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#ecf0f1')
                    else:
                        cell.set_facecolor('#bdc3c7')
        
        ax6.set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Distribution of predictions vs actual
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot distributions for each model
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ax.hist(self.y_test, bins=30, alpha=0.7, label='Actual', color='black', density=True)
        
        for i, (name, pred) in enumerate(self.predictions.items()):
            ax.hist(pred, bins=30, alpha=0.6, label=f'{name} Predicted', 
                   color=colors[i], density=True)
        
        ax.set_xlabel('House Price ($)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution Comparison: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        ax.legend()
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations created successfully!")
    
    def generate_submission(self, filename='house_price_predictions.csv'):
        """
        Generate predictions for the test set using the best model
        
        Args:
            filename (str): Output filename for predictions
        """
        print(f"\n📝 Generating submission file: {filename}")
        
        # Find best model based on RMSE_Log
        best_model_name = min(self.scores.keys(), key=lambda x: self.scores[x]['RMSE_Log'])
        best_model = self.models[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (RMSE Log: {self.scores[best_model_name]['RMSE_Log']:.4f})")
        
        # Make predictions on submission set
        submission_predictions = best_model.predict(self.X_submission)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'Id': self.test_data['Id'],
            'SalePrice': submission_predictions
        })
        
        # Save to CSV
        submission_df.to_csv(filename, index=False)
        print(f"✅ Submission saved to {filename}")
        
        return submission_df
    
    def run_complete_pipeline(self, use_improved_trees=True):
        """
        Execute the complete machine learning pipeline
        
        Args:
            use_improved_trees (bool): Whether to use Random Forest/Extra Trees
        """
        print("🚀 Starting Complete House Price Prediction Pipeline")
        print("=" * 60)
        print(f"⚙️  Configuration:")
        print(f"   - Remove outliers: {self.remove_outliers}")
        print(f"   - Use improved trees: {use_improved_trees}")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_models(use_improved_trees=use_improved_trees)
        
        # Evaluate models
        self.evaluate_models()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate submission
        submission = self.generate_submission()
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 60)
        
        # Print final summary
        best_model = min(self.scores.keys(), key=lambda x: self.scores[x]['RMSE_Log'])
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Best Model: {best_model}")
        print(f"   RMSE (Log): {self.scores[best_model]['RMSE_Log']:.4f}")
        print(f"   R² Score: {self.scores[best_model]['R2']:.4f}")
        print(f"   MAE: ${self.scores[best_model]['MAE']:,.2f}")
        
        return submission

# Main execution - Configurable pipeline
if __name__ == "__main__":
    
    print("🎯 HOUSE PRICE PREDICTION - ENHANCED PIPELINE")
    print("=" * 60)
    
    # Choose configuration (modify these as needed)
    REMOVE_OUTLIERS = True      # 🔧 Set to False to keep outliers
    USE_IMPROVED_TREES = False    # 🔧 Set to True for Random Forest + Extra Trees
    
    print(f"⚙️  Configuration:")
    print(f"   - Remove outliers: {REMOVE_OUTLIERS}")
    print(f"   - Use improved trees: {USE_IMPROVED_TREES}")
    print("=" * 60)
    
    # Create and run the predictor with chosen configuration
    predictor = HousePricePredictor(remove_outliers=REMOVE_OUTLIERS)
    submission = predictor.run_complete_pipeline(use_improved_trees=USE_IMPROVED_TREES)
    
    # Display sample predictions
    if submission is not None:
        print(f"\n📋 Sample predictions:")
        print(submission.head(10))
        print(f"\nPrediction statistics:")
        print(f"   Mean: ${submission['SalePrice'].mean():,.2f}")
        print(f"   Median: ${submission['SalePrice'].median():,.2f}")
        print(f"   Min: ${submission['SalePrice'].min():,.2f}")
        print(f"   Max: ${submission['SalePrice'].max():,.2f}")
    
    print("\n" + "="*60)
    print("✨ PIPELINE COMPLETE!")
    print("="*60)
    
    # Optional: Quick comparison function
    def run_comparison():
        """
        Run a quick comparison between different configurations
        (Uncomment and call this function if you want to compare multiple setups)
        """
        print("\n🔬 Running Configuration Comparison...")
        
        configs = [
            ("Remove Outliers + Improved Trees", True, True),
            ("Keep Outliers + Improved Trees", False, True),
            ("Remove Outliers + Basic Models", True, False)
        ]
        
        results = []
        for name, remove_out, use_improved in configs:
            print(f"\n🚀 Testing: {name}")
            pred = HousePricePredictor(remove_outliers=remove_out)
            pred.load_data()
            pred.preprocess_data()
            pred.train_models(use_improved_trees=use_improved)
            pred.evaluate_models()
            
            best_model = min(pred.scores.keys(), key=lambda x: pred.scores[x]['RMSE_Log'])
            results.append((name, best_model, pred.scores[best_model]['RMSE_Log']))
            
        print(f"\n🏆 COMPARISON RESULTS:")
        for name, model, score in sorted(results, key=lambda x: x[2]):
            print(f"   {name}: {model} (RMSE Log: {score:.4f})")
    
    # Uncomment the next line to run comparison (warning: takes longer!)
    # run_comparison()