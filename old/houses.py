import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def comprehensive_data_exploration(df, max_unique_values=20, show_plots=False):
    """
    Comprehensive data exploration combining multiple analysis approaches
    """
    print("🧠 DataFrame Overview:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n📌 Column Information:")
    print(df.dtypes)
    
    print("\n🕳️ Missing Values Analysis:")
    missing_summary = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_summary.append({
            'Column': col,
            'Missing_Count': missing_count,
            'Missing_Percent': round(missing_pct, 2)
        })
    
    missing_df = pd.DataFrame(missing_summary)
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    if not missing_df.empty:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")

    # Separate continuous and categorical features
    continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\n📊 Feature Types:")
    print(f"Continuous features ({len(continuous_cols)}): {continuous_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Detailed analysis for continuous features
    if continuous_cols:
        print("\n📊 Continuous Features Summary:")
        cont_summary = []
        for col in continuous_cols:
            series = df[col]
            count = series.count()
            pct_missing = 100 * (1 - count / len(series))
            cardinality = series.nunique(dropna=True)
            
            if count > 0:  # Only calculate stats if we have data
                summary = {
                    "Feature": col,
                    "Count": count,
                    "% Missing": round(pct_missing, 2),
                    "Cardinality": cardinality,
                    "Min": round(series.min(), 3),
                    "Q1": round(series.quantile(0.25), 3),
                    "Mean": round(series.mean(), 3),
                    "Median": round(series.median(), 3),
                    "Q3": round(series.quantile(0.75), 3),
                    "Max": round(series.max(), 3),
                    "Std": round(series.std(), 3),
                }
            else:
                summary = {
                    "Feature": col,
                    "Count": count,
                    "% Missing": round(pct_missing, 2),
                    "Cardinality": cardinality,
                    "Min": "N/A", "Q1": "N/A", "Mean": "N/A", "Median": "N/A",
                    "Q3": "N/A", "Max": "N/A", "Std": "N/A"
                }
            cont_summary.append(summary)
        
        cont_df = pd.DataFrame(cont_summary)
        print(cont_df.to_string(index=False))

    # Detailed analysis for categorical features
    if categorical_cols:
        print("\n📊 Categorical Features Summary:")
        cat_summary = []
        for col in categorical_cols:
            series = df[col]
            count = series.count()
            pct_missing = 100 * (1 - count / len(series))
            cardinality = series.nunique(dropna=True)
            
            if count > 0:
                mode_freqs = series.value_counts(dropna=True)
                mode = mode_freqs.index[0] if not mode_freqs.empty else "N/A"
                mode_freq = mode_freqs.iloc[0] if not mode_freqs.empty else 0
                mode_pct = 100 * mode_freq / count if count else 0

                second_mode = mode_freqs.index[1] if len(mode_freqs) > 1 else "N/A"
                second_mode_freq = mode_freqs.iloc[1] if len(mode_freqs) > 1 else 0
                second_mode_pct = 100 * second_mode_freq / count if count else 0
            else:
                mode = mode_freq = mode_pct = "N/A"
                second_mode = second_mode_freq = second_mode_pct = "N/A"

            summary = {
                "Feature": col,
                "Count": count,
                "% Missing": round(pct_missing, 2),
                "Cardinality": cardinality,
                "Mode": mode,
                "Mode_Freq": mode_freq,
                "Mode_%": round(mode_pct, 2) if isinstance(mode_pct, (int, float)) else mode_pct,
                "2nd_Mode": second_mode,
                "2nd_Freq": second_mode_freq,
                "2nd_%": round(second_mode_pct, 2) if isinstance(second_mode_pct, (int, float)) else second_mode_pct
            }
            cat_summary.append(summary)

        cat_df = pd.DataFrame(cat_summary)
        print(cat_df.to_string(index=False))

    # Value counts for categorical/low-cardinality columns
    print("\n📦 Value Counts for Categorical/Low-Cardinality Features:")
    for col in df.columns:
        if col in categorical_cols or df[col].nunique() <= max_unique_values:
            print(f"\n'{col}':")
            print(df[col].value_counts(dropna=False).head(10))

    # Optional plotting
    if show_plots and (continuous_cols or categorical_cols):
        plot_data_overview(df, continuous_cols, categorical_cols, max_unique_values)

    print("\n👀 Sample Data:")
    print("First 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

    return {
        'continuous_cols': continuous_cols,
        'categorical_cols': categorical_cols,
        'missing_summary': missing_df if not missing_df.empty else None
    }


def plot_data_overview(df, continuous_cols, categorical_cols, max_unique_values=20):
    """Create overview plots for the dataset"""
    # Calculate subplot dimensions
    plot_cols = continuous_cols + [col for col in categorical_cols if df[col].nunique() <= max_unique_values]
    n_plots = len(plot_cols)
    
    if n_plots == 0:
        print("No suitable columns for plotting.")
        return
    
    n_cols = min(3, n_plots)
    n_rows = (n_plots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(plot_cols):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        
        if col in continuous_cols:
            df[col].hist(bins=30, ax=ax, alpha=0.7)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
        else:
            vc = df[col].value_counts().head(max_unique_values)
            vc.plot(kind='bar', ax=ax)
            ax.set_title(f"Top Values in {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()


def intelligent_data_cleaning(df, missing_threshold=0.5, verbose=True):
    """
    Intelligent data cleaning with configurable thresholds
    """
    if verbose:
        print("🧹 Intelligent Data Cleaning:")
        print(f"Starting shape: {df.shape}")
    
    df_clean = df.copy()
    
    # 1. Drop columns with too many missing values
    missing_cols = []
    for col in df_clean.columns:
        missing_pct = df_clean[col].isnull().sum() / len(df_clean)
        if missing_pct > missing_threshold:
            missing_cols.append(col)
    
    if missing_cols:
        if verbose:
            print(f"Dropping columns with >{missing_threshold*100}% missing: {missing_cols}")
        df_clean = df_clean.drop(columns=missing_cols)
    
    # 2. Handle remaining missing values
    rows_with_missing = df_clean.isnull().any(axis=1).sum()
    total_missing = df_clean.isnull().sum().sum()
    
    if verbose:
        print(f"Rows with missing values: {rows_with_missing}")
        print(f"Total missing values: {total_missing}")
    
    if total_missing > 0:
        # Strategy based on amount of missing data
        if rows_with_missing < len(df_clean) * 0.1:  # Less than 10% of rows affected
            if verbose:
                print("Strategy: Dropping rows with missing values (< 10% affected)")
            df_clean = df_clean.dropna()
        else:
            if verbose:
                print("Strategy: Imputing missing values")
            # Impute missing values
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if df_clean[col].dtype in ['object', 'category']:
                        # Categorical: use mode or 'Unknown'
                        mode_val = df_clean[col].mode()
                        fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                        df_clean[col] = df_clean[col].fillna(fill_val)
                        if verbose:
                            print(f"   {col}: filled with '{fill_val}'")
                    else:
                        # Numerical: use median
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                        if verbose:
                            print(f"   {col}: filled with {median_val}")
    
    if verbose:
        print(f"Final shape: {df_clean.shape}")
        print(f"Remaining missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """Remove outliers using IQR method with configurable multiplier"""
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_info = {}
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Count outliers
            outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            outlier_info[col] = outlier_count
            
            if outlier_count > 0:
                print(f"🚨 Removing {outlier_count} outliers from {col}")
                df_clean = df_clean[~outliers_mask]
    
    return df_clean, outlier_info


class DecisionNode:
    """Node class for Decision Tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Prediction value (for leaf nodes)


class DecisionTreeFromScratch:
    """
    Decision Tree Regressor implementation from scratch
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.feature_names = None
    
    def fit(self, X, y):
        """Train the decision tree"""
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _mse(self, y):
        """Calculate Mean Squared Error"""
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)
    
    def _best_split(self, X, y):
        """Find the best split for the data"""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted MSE
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)
                
                weighted_mse = (n_left / n_total) * left_mse + (n_right / n_total) * right_mse
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return DecisionNode(value=np.mean(y))
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return DecisionNode(value=np.mean(y))
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def _predict_sample(self, x, node):
        """Predict a single sample"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict for multiple samples"""
        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def get_tree_rules(self, node=None, depth=0, rules=None):
        """Extract rules from the decision tree"""
        if rules is None:
            rules = []
        if node is None:
            node = self.root
        
        if node.value is not None:  # Leaf node
            rules.append(f"{'  ' * depth}Predict: {node.value:.4f}")
        else:
            feature_name = self.feature_names[node.feature] if self.feature_names else f"feature_{node.feature}"
            rules.append(f"{'  ' * depth}If {feature_name} <= {node.threshold:.4f}:")
            self.get_tree_rules(node.left, depth + 1, rules)
            rules.append(f"{'  ' * depth}Else:")
            self.get_tree_rules(node.right, depth + 1, rules)
        
        return rules


class KNNFromScratch:
    """
    K-Nearest Neighbors implementation from scratch
    """
    
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _calculate_distance(self, x1, x2):
        """Calculate distance between two points"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _get_neighbors(self, x):
        """Get k nearest neighbors for a point"""
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x, x_train)
            distances.append((dist, i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        return neighbors
    
    def predict(self, X):
        """Predict for multiple samples (regression)"""
        X = np.array(X)
        predictions = []
        for x in X:
            neighbors = self._get_neighbors(x)
            neighbor_indices = [idx for _, idx in neighbors]
            neighbor_values = self.y_train[neighbor_indices]
            # Return mean of neighbor values for regression
            predictions.append(np.mean(neighbor_values))
        return np.array(predictions)


def find_optimal_k(X, y, k_range=range(1, 21), cv=5, scoring='r2'):
    """
    Find optimal K value for KNN with comprehensive evaluation
    """
    print(f"\n🔍 Finding Optimal K (range: {min(k_range)}-{max(k_range)}):")
    
    results = {}
    best_k = None
    best_score = -np.inf if scoring in ['r2'] else np.inf
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        
        # Cross-validation scores
        if scoring == 'r2':
            scores = cross_val_score(knn, X, y, cv=kf, scoring='r2')
            is_better = scores.mean() > best_score
        elif scoring == 'mse':
            scores = cross_val_score(knn, X, y, cv=kf, scoring='neg_mean_squared_error')
            scores = -scores  # Convert to positive
            is_better = scores.mean() < best_score
        else:
            raise ValueError(f"Unsupported scoring: {scoring}")
        
        results[k] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"   K={k:2d}: {scoring.upper()}={scores.mean():.4f} (±{scores.std():.4f})")
        
        if is_better:
            best_score = scores.mean()
            best_k = k
    
    print(f"\n🏆 Best K: {best_k} with {scoring.upper()} score: {best_score:.4f}")
    return results, best_k


def optimize_decision_tree_advanced(X, y, cv=5, n_iter=50):
    """
    Advanced Decision Tree optimization with better parameter search
    """
    print(f"\n🌳 Advanced Decision Tree Optimization:")
    
    # Expanded parameter grid
    param_distributions = {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20, 50],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.9],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'splitter': ['best', 'random'],
        'min_impurity_decrease': [0.0, 0.01, 0.02, 0.05]
    }
    
    dt = DecisionTreeRegressor(random_state=42)
    
    # Use RandomizedSearchCV for efficiency
    random_search = RandomizedSearchCV(
        dt, param_distributions, 
        n_iter=n_iter, cv=cv, 
        scoring='r2', random_state=42, 
        n_jobs=-1, verbose=0
    )
    
    random_search.fit(X, y)
    
    print(f"🏆 Best DT Parameters: {random_search.best_params_}")
    print(f"Best R² Score: {random_search.best_score_:.4f}")
    
    return random_search.best_params_, random_search.best_score_

def optimize_svr(X, y, cv=5, n_iter=30):
    """
    Optimize SVR hyperparameters
    """
    print(f"\n🔧 SVR Optimization:")
    
    # SVR parameter grid
    param_distributions = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    svr = SVR()
    
    random_search = RandomizedSearchCV(
        svr, param_distributions,
        n_iter=n_iter, cv=cv,
        scoring='r2', random_state=42,
        n_jobs=-1, verbose=0
    )
    
    random_search.fit(X, y)
    
    print(f"🏆 Best SVR Parameters: {random_search.best_params_}")
    print(f"Best R² Score: {random_search.best_score_:.4f}")
    
    return random_search.best_params_, random_search.best_score_

def optimize_random_forest(X, y, cv=5, n_iter=30):
    """
    Optimize Random Forest hyperparameters
    """
    print(f"\n🌲 Random Forest Optimization:")
    
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42)
    
    random_search = RandomizedSearchCV(
        rf, param_distributions,
        n_iter=n_iter, cv=cv,
        scoring='r2', random_state=42,
        n_jobs=-1, verbose=0
    )
    
    random_search.fit(X, y)
    
    print(f"🏆 Best RF Parameters: {random_search.best_params_}")
    print(f"Best R² Score: {random_search.best_score_:.4f}")
    
    return random_search.best_params_, random_search.best_score_


def plot_k_optimization(results, scoring='r2'):
    """Plot K optimization results"""
    k_values = list(results.keys())
    scores = [results[k]['mean'] for k in k_values]
    stds = [results[k]['std'] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, scores, yerr=stds, marker='o', capsize=5, capthick=2)
    plt.xlabel('K Value')
    plt.ylabel(f'{scoring.upper()} Score')
    plt.title(f'KNN Performance vs K Value ({scoring.upper()})')
    plt.grid(True, alpha=0.3)
    
    # Highlight best K
    best_k = max(results.keys(), key=lambda k: results[k]['mean']) if scoring == 'r2' else min(results.keys(), key=lambda k: results[k]['mean'])
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_decision_tree(model, feature_names, max_depth=3):
    """Visualize decision tree structure"""
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names,
              filled=True,
              rounded=True,
              fontsize=10,
              max_depth=max_depth)
    plt.title("Decision Tree Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()


def analyze_feature_importance(model, feature_names):
    """Analyze and plot feature importance from decision tree"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n🎯 Feature Importance Analysis:")
    print("-" * 40)
    for i in range(len(feature_names)):
        print(f"{i+1:2d}. {feature_names[indices[i]]:<20} {importances[indices[i]]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance - Decision Tree")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return dict(zip([feature_names[i] for i in indices], importances[indices]))

def evaluate_multiple_models(X, y, cv=5):
    """
    Wrapper to maintain compatibility - calls enhanced version
    """
    results, params = evaluate_multiple_models_enhanced(X, y, cv)
    return results

def evaluate_multiple_models_enhanced(X, y, cv=5):
    """
    Enhanced multi-model evaluation with SVR and ensemble methods
    """
    print("\n🏆 Enhanced Multi-Model Evaluation:")
    print("="*60)
    
    # Get optimized parameters first
    print("🔍 Optimizing hyperparameters...")
    
    # Optimize key models
    best_dt_params, _ = optimize_decision_tree_advanced(X, y, cv=cv, n_iter=30)
    best_svr_params, _ = optimize_svr(X, y, cv=cv, n_iter=20)
    best_rf_params, _ = optimize_random_forest(X, y, cv=cv, n_iter=20)
    
    # Define enhanced model set
    models = {
        'Linear Regression': LinearRegression(),
        'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
        'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
        'KNN (k=7)': KNeighborsRegressor(n_neighbors=7),
        'KNN (k=10)': KNeighborsRegressor(n_neighbors=10),
        'Decision Tree (Basic)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Decision Tree (Optimized)': DecisionTreeRegressor(**best_dt_params, random_state=42),
        'Random Forest (Basic)': RandomForestRegressor(n_estimators=100, random_state=42),
        'Random Forest (Optimized)': RandomForestRegressor(**best_rf_params, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR (RBF)': SVR(kernel='rbf'),
        'SVR (Optimized)': SVR(**best_svr_params),
        'SVR (Linear)': SVR(kernel='linear'),
        'SVR (Polynomial)': SVR(kernel='poly', degree=3)
    }
    
    results = {}
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n📊 Evaluating {name}...")
        
        try:
            # Cross-validation scores
            mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
            mae_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
            r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
            
            # Convert negative scores to positive
            mse_scores = -mse_scores
            mae_scores = -mae_scores
            
            # Store results
            results[name] = {
                'MSE_mean': mse_scores.mean(),
                'MSE_std': mse_scores.std(),
                'MAE_mean': mae_scores.mean(),
                'MAE_std': mae_scores.std(),
                'R2_mean': r2_scores.mean(),
                'R2_std': r2_scores.std(),
                'RMSE_mean': np.sqrt(mse_scores.mean())
            }
            
            print(f"   MSE: {mse_scores.mean():.4f} (±{mse_scores.std():.4f})")
            print(f"   MAE: {mae_scores.mean():.4f} (±{mae_scores.std():.4f})")
            print(f"   R²:  {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
            print(f"   RMSE: {np.sqrt(mse_scores.mean()):.4f}")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {name}: {e}")
            continue
    
    return results, {
        'best_dt_params': best_dt_params,
        'best_svr_params': best_svr_params,
        'best_rf_params': best_rf_params
    }

def plot_model_comparison(results):
    """Plot comprehensive model comparison including Decision Trees"""
    models = list(results.keys())
    metrics = ['MSE_mean', 'MAE_mean', 'R2_mean']
    metric_names = ['MSE', 'MAE', 'R²']
    colors = ['lightcoral', 'lightskyblue', 'lightgreen']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        values = [results[model][metric] for model in models]
        
        bars = axes[i].bar(models, values, color=color, alpha=0.7)
        axes[i].set_title(f'{name} Comparison')
        axes[i].set_ylabel(name)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def plot_learning_curves(X, y, models_dict=None):
    """Plot learning curves for different models"""
    if models_dict is None:
        models_dict = {
            'Linear Regression': LinearRegression(),
            'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
            'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
        }
    
    # Fixed: ensure train_sizes don't include 1.0 exactly
    train_sizes = np.linspace(0.1, 0.9, 9)  # Changed from 0.1 to 1.0 to 0.1 to 0.9
    
    fig, axes = plt.subplots(1, len(models_dict), figsize=(6*len(models_dict), 5))
    if len(models_dict) == 1:
        axes = [axes]
    elif len(models_dict) > 1:
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        elif axes.ndim == 1:
            pass  # axes is already correct
        else:
            axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models_dict.items()):
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            try:
                # Fixed: ensure we don't use exactly 1.0 as train_size
                effective_train_size = min(train_size, 0.8)  # Cap at 0.8 to ensure test data
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=effective_train_size, random_state=42
                )
                
                # Ensure we have enough data for training and testing
                if len(X_train) < 2 or len(X_test) < 1:
                    continue
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_test)
                
                # R² scores
                train_scores.append(r2_score(y_train, train_pred))
                val_scores.append(r2_score(y_test, val_pred))
                
            except Exception as e:
                print(f"Warning: Skipping train_size {train_size} for {name}: {e}")
                continue
        
        # Only plot if we have data
        if len(train_scores) > 0 and len(val_scores) > 0:
            # Use only the train_sizes that worked
            working_train_sizes = train_sizes[:len(train_scores)]
            
            # Plot
            axes[idx].plot(working_train_sizes, train_scores, 'o-', color='blue', label='Training Score')
            axes[idx].plot(working_train_sizes, val_scores, 'o-', color='red', label='Validation Score')
            axes[idx].set_title(f'Learning Curve - {name}')
            axes[idx].set_xlabel('Training Set Size')
            axes[idx].set_ylabel('R² Score')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        else:
            # If no data, show empty plot with message
            axes[idx].text(0.5, 0.5, f'No data available\nfor {name}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'Learning Curve - {name}')
    
    plt.tight_layout()
    plt.show()


def auto_find_target_column(df):
    """
    Automatically find the most likely target column
    """
    target_keywords = ['price', 'value', 'cost', 'salary', 'sales', 'revenue', 'target', 'y', 'label']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # First, look for obvious target column names
    for keyword in target_keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                print(f"🎯 Found target column by keyword: '{col}'")
                return col
    
    # If no obvious target, suggest the numeric column with highest variance
    if len(numeric_cols) > 0:
        variances = df[numeric_cols].var()
        suggested_target = variances.idxmax()
        print(f"🎯 Suggested target column (highest variance): '{suggested_target}'")
        return suggested_target
    
    print("❌ No suitable target column found!")
    return None

def advanced_feature_engineering(X, y, feature_names=None):
    """
    Advanced feature engineering to improve model performance
    """
    print("\n🔧 Advanced Feature Engineering:")
    
    X_enhanced = X.copy()
    new_features = []
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # 1. Polynomial features for top features
    # Find top 5 most correlated features with target
    correlations = []
    for i, col in enumerate(feature_names):
        corr = abs(np.corrcoef(X.iloc[:, i], y)[0, 1])
        if not np.isnan(corr):
            correlations.append((col, i, corr))
    
    correlations.sort(key=lambda x: x[2], reverse=True)
    top_features = correlations[:5]
    
    print(f"Top 5 correlated features: {[f[0] for f in top_features]}")
    
    # Create polynomial features for top features
    for i, (name, idx, corr) in enumerate(top_features[:3]):  # Top 3 only
        # Square
        X_enhanced[f'{name}_squared'] = X.iloc[:, idx] ** 2
        new_features.append(f'{name}_squared')
        
        # Cube (if not too extreme)
        if X.iloc[:, idx].max() < 100:  # Avoid extreme values
            X_enhanced[f'{name}_cubed'] = X.iloc[:, idx] ** 3
            new_features.append(f'{name}_cubed')
    
    # 2. Interaction features between top correlated features
    if len(top_features) >= 2:
        for i in range(min(3, len(top_features))):
            for j in range(i+1, min(3, len(top_features))):
                name1, idx1, _ = top_features[i]
                name2, idx2, _ = top_features[j]
                
                # Product
                X_enhanced[f'{name1}_x_{name2}'] = X.iloc[:, idx1] * X.iloc[:, idx2]
                new_features.append(f'{name1}_x_{name2}')
                
                # Ratio (avoid division by zero)
                denominator = X.iloc[:, idx2]
                denominator = np.where(denominator == 0, 1e-8, denominator)
                X_enhanced[f'{name1}_div_{name2}'] = X.iloc[:, idx1] / denominator
                new_features.append(f'{name1}_div_{name2}')
    
    # 3. Statistical features
    # Row-wise statistics
    X_enhanced['row_mean'] = X.mean(axis=1)
    X_enhanced['row_std'] = X.std(axis=1)
    X_enhanced['row_max'] = X.max(axis=1)
    X_enhanced['row_min'] = X.min(axis=1)
    new_features.extend(['row_mean', 'row_std', 'row_max', 'row_min'])
    
    print(f"Added {len(new_features)} new features")
    print(f"Enhanced dataset shape: {X_enhanced.shape}")
    
    return X_enhanced, new_features



def analyze_prediction_errors(y_true, y_pred, model_name="Model"):
    """Analyze prediction errors with visualizations"""
    residuals = y_true - y_pred
    
    print(f"\n📊 Prediction Error Analysis - {model_name}:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Actual vs Predicted - {model_name}')
    
    # Residuals plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'Residuals Plot - {model_name}')
    
    # Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Residuals Distribution - {model_name}')
    
    # Q-Q plot (simple approximation)
    try:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q Plot - {model_name}')
    except ImportError:
        # Simple normal plot if scipy not available
        residuals_sorted = np.sort(residuals)
        axes[1, 1].plot(residuals_sorted)
        axes[1, 1].set_title(f'Sorted Residuals - {model_name}')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'residuals': residuals
    }


def compare_knn_implementations(X, y, k=5, test_size=0.2):
    """
    Compare KNN from scratch vs sklearn implementation
    """
    print(f"\n🔄 Comparing KNN Implementations (K={k}):")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # From scratch implementation
    knn_scratch = KNNFromScratch(k=k)
    knn_scratch.fit(X_train, y_train)
    y_pred_scratch = knn_scratch.predict(X_test)
    
    # Sklearn implementation
    knn_sklearn = KNeighborsRegressor(n_neighbors=k)
    knn_sklearn.fit(X_train, y_train)
    y_pred_sklearn = knn_sklearn.predict(X_test)
    
    # Calculate metrics
    metrics_scratch = {
        'MSE': mean_squared_error(y_test, y_pred_scratch),
        'MAE': mean_absolute_error(y_test, y_pred_scratch),
        'R2': r2_score(y_test, y_pred_scratch),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_scratch))
    }
    
    metrics_sklearn = {
        'MSE': mean_squared_error(y_test, y_pred_sklearn),
        'MAE': mean_absolute_error(y_test, y_pred_sklearn),
        'R2': r2_score(y_test, y_pred_sklearn),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
    }
    
    print("📊 Results Comparison:")
    print(f"{'Metric':<10} {'Scratch':<12} {'Sklearn':<12} {'Difference':<12}")
    print("-" * 50)
    for metric in metrics_scratch:
        diff = abs(metrics_scratch[metric] - metrics_sklearn[metric])
        print(f"{metric:<10} {metrics_scratch[metric]:<12.4f} {metrics_sklearn[metric]:<12.4f} {diff:<12.4f}")
    
    return {
        'scratch': metrics_scratch,
        'sklearn': metrics_sklearn
    }

def compare_tree_algorithms(X, y, test_size=0.2):
    """
    Compare different tree-based algorithms
    """
    print(f"\n🌳 Comparing Tree-Based Algorithms:")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Define tree models
    tree_models = {
        'Decision Tree (Basic)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Decision Tree (Optimized)': DecisionTreeRegressor(
            max_depth=10, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', criterion='squared_error', random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
    }
    
    results = {}
    
    for name, model in tree_models.items():
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        print(f"\n{name}:")
        print(f"   R²: {results[name]['R2']:.4f}")
        print(f"   MSE: {results[name]['MSE']:.4f}")
        print(f"   MAE: {results[name]['MAE']:.4f}")
        print(f"   RMSE: {results[name]['RMSE']:.4f}")
    
    return results

def compare_tree_implementations(X, y, max_depth=5, test_size=0.2):
    """
    Compare Decision Tree from scratch vs sklearn implementation
    """
    print(f"\n🌳 Comparing Decision Tree Implementations (max_depth={max_depth}):")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # From scratch implementation
    dt_scratch = DecisionTreeFromScratch(max_depth=max_depth)
    dt_scratch.fit(X_train, y_train)
    y_pred_scratch = dt_scratch.predict(X_test)
    
    # Sklearn implementation
    dt_sklearn = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt_sklearn.fit(X_train, y_train)
    y_pred_sklearn = dt_sklearn.predict(X_test)
    
    # Calculate metrics
    metrics_scratch = {
        'MSE': mean_squared_error(y_test, y_pred_scratch),
        'MAE': mean_absolute_error(y_test, y_pred_scratch),
        'R2': r2_score(y_test, y_pred_scratch),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_scratch))
    }
    
    metrics_sklearn = {
        'MSE': mean_squared_error(y_test, y_pred_sklearn),
        'MAE': mean_absolute_error(y_test, y_pred_sklearn),
        'R2': r2_score(y_test, y_pred_sklearn),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
    }
    
    print("📊 Results Comparison:")
    print(f"{'Metric':<10} {'Scratch':<12} {'Sklearn':<12} {'Difference':<12}")
    print("-" * 50)
    for metric in metrics_scratch:
        diff = abs(metrics_scratch[metric] - metrics_sklearn[metric])
        print(f"{metric:<10} {metrics_scratch[metric]:<12.4f} {metrics_sklearn[metric]:<12.4f} {diff:<12.4f}")
    
    # Print tree rules from scratch implementation
    print(f"\n🌳 Decision Tree Rules (From Scratch - max_depth={max_depth}):")
    rules = dt_scratch.get_tree_rules()
    for rule in rules[:20]:  # Show first 20 rules
        print(rule)
    if len(rules) > 20:
        print(f"... ({len(rules) - 20} more rules)")
    
    return {
        'scratch': metrics_scratch,
        'sklearn': metrics_sklearn,
        'tree_rules': rules
    }
# ===============================
# MAIN EXECUTION PIPELINE
# ===============================

def run_complete_analysis(csv_file, target_col=None, remove_outliers=True, show_plots=False):
    """
    Complete analysis pipeline including Decision Trees
    """
    print("🚀 Starting Complete Data Analysis Pipeline with Decision Trees")
    print("="*70)
    
    # 1. Load data
    print(f"📂 Loading data from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Data loaded successfully: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # 2. Initial exploration
    print("\n🔍 Phase 1: Data Exploration")
    exploration_results = comprehensive_data_exploration(df, show_plots=show_plots)
    
    # 3. Data cleaning
    print("\n🧹 Phase 2: Data Cleaning")
    df_clean = intelligent_data_cleaning(df)
    
    if len(df_clean) == 0:
        print("❌ No data remaining after cleaning!")
        return None
    
    # 4. Target column identification
    print("\n🎯 Phase 3: Target Identification")
    if target_col is None:
        target_col = auto_find_target_column(df_clean)
    
    if target_col is None or target_col not in df_clean.columns:
        print("❌ No valid target column found!")
        return None
    
    # 5. Feature preparation
    print(f"\n🔧 Phase 4: Feature Preparation")
    print(f"Target column: {target_col}")
    
    # Encode categorical variables
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    if len(categorical_cols) > 0:
        print(f"Encoding categorical variables: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    # Prepare features and target
    y = df_clean[target_col].copy()
    exclude_cols = [target_col] + [col for col in df_clean.columns if 'id' in col.lower()]
    X = df_clean.drop(columns=exclude_cols, errors='ignore')
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(X.columns)}")
    
    # Remove outliers if requested
    if remove_outliers:
        print("\n🚨 Phase 5: Outlier Removal")
        X, outlier_info = remove_outliers_iqr(X)
        y = y.loc[X.index]  # Keep corresponding target values
        print(f"Shape after outlier removal: {X.shape}")
    
    # 6. Feature scaling
    print("\n⚖️ Phase 6: Feature Scaling")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # 7. KNN Analysis
    print("\n🤖 Phase 7: KNN Analysis")
    
    # Find optimal K
    k_results, best_k = find_optimal_k(X_scaled, y, k_range=range(1, 21))
    
    if show_plots:
        plot_k_optimization(k_results)
    
    # Compare KNN implementations
    knn_comparison = compare_knn_implementations(X_scaled, y, k=best_k)
    
    # 8. Decision Tree Analysis
    print("\n🌳 Phase 8: Decision Tree Analysis")
    
    # Optimize Decision Tree hyperparameters
    dt_results, best_dt_params = optimize_decision_tree(X_scaled, y)
    
    # Compare Decision Tree implementations
    dt_comparison = compare_tree_implementations(X_scaled, y, max_depth=best_dt_params['max_depth'])
    
    # Train best Decision Tree for feature importance analysis
    best_dt = DecisionTreeRegressor(**best_dt_params, random_state=42)
    best_dt.fit(X_scaled, y)
    
    # Feature importance analysis
    if show_plots:
        feature_importance = analyze_feature_importance(best_dt, list(X.columns))
        
        # Visualize decision tree (limited depth for readability)
        visualize_decision_tree(best_dt, list(X.columns), max_depth=3)
    
    # 9. Multi-model evaluation
    print("\n🏁 Phase 9: Multi-Model Evaluation")
    model_results = evaluate_multiple_models(X_scaled, y)
    
    if show_plots:
        plot_model_comparison(model_results)
        
        # Learning curves
        plot_learning_curves(X_scaled, y)
    
    # 10. Detailed error analysis for best models
    print("\n📊 Phase 10: Detailed Error Analysis")
    
    # Find top 3 models
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['R2_mean'], reverse=True)
    top_models = sorted_models[:3]
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    error_analyses = {}
    for model_name, _ in top_models:
        if 'Linear Regression' in model_name:
            model = LinearRegression()
        elif 'KNN' in model_name:
            k = int(model_name.split('k=')[1].split(')')[0])
            model = KNeighborsRegressor(n_neighbors=k)
        elif 'Decision Tree' in model_name:
            if 'depth=3' in model_name:
                model = DecisionTreeRegressor(max_depth=3, random_state=42)
            elif 'depth=5' in model_name:
                model = DecisionTreeRegressor(max_depth=5, random_state=42)
            elif 'depth=10' in model_name:
                model = DecisionTreeRegressor(max_depth=10, random_state=42)
            else:
                model = DecisionTreeRegressor(random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if show_plots:
            error_analyses[model_name] = analyze_prediction_errors(y_test, y_pred, model_name)
    
    # 11. Summary
    print("\n📋 Analysis Summary:")
    print("="*50)
    print(f"Dataset: {csv_file}")
    print(f"Final data shape: {X_scaled.shape}")
    print(f"Target column: {target_col}")
    print(f"Best K for KNN: {best_k}")
    print(f"Best Decision Tree params: {best_dt_params}")
    
    # Find best overall model
    best_model = max(model_results.keys(), key=lambda k: model_results[k]['R2_mean'])
    best_r2 = model_results[best_model]['R2_mean']
    print(f"Best performing model: {best_model} (R² = {best_r2:.4f})")
    
    # Top 5 models ranking
    print(f"\n🏆 Top 5 Models Ranking (by R² score):")
    for i, (model_name, results) in enumerate(sorted_models[:5], 1):
        print(f"{i}. {model_name:<25} R² = {results['R2_mean']:.4f} (±{results['R2_std']:.4f})")
    
    return {
        'df_original': df,
        'df_clean': df_clean,
        'X': X_scaled,
        'y': y,
        'target_col': target_col,
        'best_k': best_k,
        'best_dt_params': best_dt_params,
        'model_results': model_results,
        'knn_comparison': knn_comparison,
        'dt_comparison': dt_comparison,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
        'error_analyses': error_analyses if show_plots else None
    }

def run_complete_analysis_enhanced(csv_file, target_col=None, remove_outliers=True, 
                                 show_plots=False, feature_engineering=True):
    """
    Enhanced complete analysis pipeline with SVR and improved Decision Trees
    """
    print("🚀 Starting Enhanced Data Analysis Pipeline")
    print("="*70)
    
    # Phases 1-6 remain the same as original function
    # ... (keep existing code for data loading, exploration, cleaning, etc.)
    
    # Load data
    print(f"📂 Loading data from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Data loaded successfully: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Initial exploration
    print("\n🔍 Phase 1: Data Exploration")
    exploration_results = comprehensive_data_exploration(df, show_plots=show_plots)
    
    # Data cleaning
    print("\n🧹 Phase 2: Data Cleaning")
    df_clean = intelligent_data_cleaning(df)
    
    if len(df_clean) == 0:
        print("❌ No data remaining after cleaning!")
        return None
    
    # Target column identification
    print("\n🎯 Phase 3: Target Identification")
    if target_col is None:
        target_col = auto_find_target_column(df_clean)
    
    if target_col is None or target_col not in df_clean.columns:
        print("❌ No valid target column found!")
        return None
    
    # Feature preparation
    print(f"\n🔧 Phase 4: Feature Preparation")
    print(f"Target column: {target_col}")
    
    # Encode categorical variables
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    if len(categorical_cols) > 0:
        print(f"Encoding categorical variables: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    # Prepare features and target
    y = df_clean[target_col].copy()
    exclude_cols = [target_col] + [col for col in df_clean.columns if 'id' in col.lower()]
    X = df_clean.drop(columns=exclude_cols, errors='ignore')
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {list(X.columns)}")
    
    # Remove outliers if requested
    if remove_outliers:
        print("\n🚨 Phase 5: Outlier Removal")
        X, outlier_info = remove_outliers_iqr(X)
        y = y.loc[X.index]
        print(f"Shape after outlier removal: {X.shape}")
    
    # Feature scaling
    print("\n⚖️ Phase 6: Feature Scaling")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # NEW: Advanced feature engineering
    if feature_engineering and X_scaled.shape[1] > 2:
        print("\n🔬 Phase 6.5: Advanced Feature Engineering")
        X_enhanced, new_features = advanced_feature_engineering(X_scaled, y, list(X.columns))
        
        # Re-scale enhanced features
        scaler_enhanced = StandardScaler()
        X_final = scaler_enhanced.fit_transform(X_enhanced)
        X_final = pd.DataFrame(X_final, columns=X_enhanced.columns, index=X_enhanced.index)
        
        print(f"Using enhanced features: {X_final.shape}")
        X_analysis = X_final
    else:
        X_analysis = X_scaled
        new_features = []
    
    # KNN Analysis
    print("\n🤖 Phase 7: KNN Analysis")
    k_results, best_k = find_optimal_k(X_analysis, y, k_range=range(1, min(21, len(X_analysis)-1)))
    
    # Tree Algorithm Comparison
    print("\n🌳 Phase 8: Tree Algorithm Comparison")
    tree_comparison = compare_tree_algorithms(X_analysis, y)
    
    # Enhanced multi-model evaluation
    print("\n🏁 Phase 9: Enhanced Multi-Model Evaluation")
    model_results, optimization_params = evaluate_multiple_models_enhanced(X_analysis, y)
    
    if show_plots:
        plot_model_comparison(model_results)
    
    # Summary
    print("\n📋 Enhanced Analysis Summary:")
    print("="*50)
    print(f"Dataset: {csv_file}")
    print(f"Final data shape: {X_analysis.shape}")
    print(f"Target column: {target_col}")
    print(f"Feature engineering: {feature_engineering}")
    if feature_engineering:
        print(f"New features added: {len(new_features)}")
    print(f"Best K for KNN: {best_k}")
    
    # Find best overall model
    best_model = max(model_results.keys(), key=lambda k: model_results[k]['R2_mean'])
    best_r2 = model_results[best_model]['R2_mean']
    print(f"Best performing model: {best_model} (R² = {best_r2:.4f})")
    
    # Top 10 models ranking
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['R2_mean'], reverse=True)
    print(f"\n🏆 Top 10 Models Ranking (by R² score):")
    for i, (model_name, results) in enumerate(sorted_models[:10], 1):
        print(f"{i:2d}. {model_name:<30} R² = {results['R2_mean']:.4f} (±{results['R2_std']:.4f})")
    
    return {
        'df_original': df,
        'df_clean': df_clean,
        'X': X_analysis,
        'y': y,
        'target_col': target_col,
        'best_k': best_k,
        'model_results': model_results,
        'optimization_params': optimization_params,
        'tree_comparison': tree_comparison,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
        'new_features': new_features if feature_engineering else [],
        'feature_engineering': feature_engineering
    }

# Utility function to run quick Decision Tree analysis
def quick_decision_tree_analysis(X, y, feature_names=None, max_depth=5, show_plots=True):
    """
    Quick Decision Tree analysis function
    """
    print("🌳 Quick Decision Tree Analysis")
    print("="*40)
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt.predict(X_test)
    
    # Metrics
    print(f"Decision Tree Performance (max_depth={max_depth}):")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    
    if show_plots:
        # Feature importance
        feature_importance = analyze_feature_importance(dt, feature_names)
        
        # Tree visualization
        visualize_decision_tree(dt, feature_names, max_depth=3)
        
        # Error analysis
        analyze_prediction_errors(y_test, y_pred, f"Decision Tree (depth={max_depth})")
    
    return dt, {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'feature_importance': dt.feature_importances_
    }

# Example usage and demonstrations
if __name__ == "__main__":
    print("🎯 Machine Learning Analysis Program")
    print("="*50)
    
    # Option 1: Run with sample data if no CSV file exists
    try:
        # Try to run with a real CSV file
        # results = run_complete_analysis("houses.csv", show_plots=False)
        results = run_complete_analysis_enhanced("test.csv", feature_engineering=True, show_plots=False)

        
        if results:
            print("\n🎉 Analysis completed successfully!")
            print("Results available in the 'results' dictionary")
            
            # Display key findings
            print(f"\n📊 Key Findings:")
            print(f"- Dataset shape: {results['X'].shape}")
            print(f"- Target column: {results['target_col']}")
            print(f"- Best K for KNN: {results['best_k']}")
            print(f"- Best Decision Tree params: {results['best_dt_params']}")
            
            # Find best model
            best_model = max(results['model_results'].keys(), 
                           key=lambda k: results['model_results'][k]['R2_mean'])
            best_r2 = results['model_results'][best_model]['R2_mean']
            print(f"- Best model: {best_model} (R² = {best_r2:.4f})")
            
        else:
            print("\n❌ Analysis failed with the provided CSV file!")
            
    except FileNotFoundError:
        print("📄 CSV file not found...")
    
    # Additional demonstration functions
    print("\n🔧 Additional Analysis Functions Available:")
    print("- comprehensive_data_exploration(df)")
    print("- intelligent_data_cleaning(df)")
    print("- find_optimal_k(X, y)")
    print("- optimize_decision_tree(X, y)")
    print("- compare_knn_implementations(X, y)")
    print("- compare_tree_implementations(X, y)")
    print("- quick_decision_tree_analysis(X, y)")
    print("- evaluate_multiple_models(X, y)")
       
    print("\n🏁 Program execution completed!")
    print("💡 Tip: Modify the 'show_plots' parameter to control visualization output")
    print("📈 Tip: Try different CSV files by changing the filename in run_complete_analysis()")