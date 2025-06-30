import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def explore_dataframe(df, max_unique_values=20, show_plots=False):
    print("🧠 DataFrame Shape:")
    print(df.shape)
    
    print("\n📌 Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n🕳️ Missing Values per Column:")
    print(df.isnull().sum())
    
    print("\n🆔 Unique Values per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    print("\n📊 Summary Statistics (Numerical):")
    print(df.describe())
    
    if not df.select_dtypes(include='object').empty:
        print("\n🔤 Summary Statistics (Categorical):")
        print(df.describe(include='object'))
    else:
        print("\n🔤 Summary Statistics (Categorical):")
        print("No categorical columns in this DataFrame.")

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or df[col].nunique() <= max_unique_values:
            print(f"\n📦 Value Counts for '{col}':")
            print(df[col].value_counts(dropna=False))

    if show_plots:
        for col in df.columns:
            plt.figure(figsize=(8, 4))
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"Histogram of {col}")
            else:
                vc = df[col].value_counts().head(max_unique_values)
                sns.barplot(x=vc.index, y=vc.values)
                plt.title(f"Top {max_unique_values} Categories in {col}")
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    print("\n👀 First 5 Rows:")
    print(df.head())

    print("\n🔚 Last 5 Rows:")
    print(df.tail())

def merge_duplicate_games(df):
    # Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols + ['Name', 'Platform']]

    # Group by 'Name' and 'Platform'
    grouped_df = df.groupby(['Name', 'Platform'], as_index=False).agg({
        **{col: 'sum' for col in numeric_cols},
        **{col: 'first' for col in non_numeric_cols}
    })

    return grouped_df

def group_by_feature_type(df):
    continuous_cols = []
    categorical_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)
    
    df_continuous = df[continuous_cols].copy()
    df_categorical = df[categorical_cols].copy()
    
    return df_continuous, df_categorical

def summarize_features_v2(df_continuous, df_categorical):
    print("\n📊 Summary of Continuous Features:\n")
    cont_summary = []
    for col in df_continuous.columns:
        series = df_continuous[col]
        count = series.count()
        pct_missing = 100 * (1 - count / len(series))
        cardinality = series.nunique(dropna=True)
        summary = {
            "Feature": col,
            "Count": count,
            "% Miss": round(pct_missing, 2),
            "Card.": cardinality,
            "min": round(series.min(), 3),
            "Q1": round(series.quantile(0.25), 3),
            "Mean": round(series.mean(), 3),
            "Median": round(series.median(), 3),
            "Q3": round(series.quantile(0.75), 3),
            "Max": round(series.max(), 3),
            "Std. Dev.": round(series.std(), 3),
        }
        cont_summary.append(summary)
    
    cont_df = pd.DataFrame(cont_summary)
    print(cont_df.to_string(index=False))

    print("\n📊 Summary of Categorical Features:\n")
    cat_summary = []
    for col in df_categorical.columns:
        series = df_categorical[col]
        count = series.count()
        pct_missing = 100 * (1 - count / len(series))
        cardinality = series.nunique(dropna=True)
        mode_freqs = series.value_counts(dropna=True)
        mode = mode_freqs.index[0] if not mode_freqs.empty else np.nan
        mode_freq = mode_freqs.iloc[0] if not mode_freqs.empty else 0
        mode_pct = 100 * mode_freq / count if count else 0

        second_mode = mode_freqs.index[1] if len(mode_freqs) > 1 else np.nan
        second_mode_freq = mode_freqs.iloc[1] if len(mode_freqs) > 1 else 0
        second_mode_pct = 100 * second_mode_freq / count if count else 0

        summary = {
            "Feature": col,
            "Count": count,
            "% Miss": round(pct_missing, 2),
            "Card.": cardinality,
            "Mode": mode,
            "Mode Freq": mode_freq,
            "Mode %": round(mode_pct, 2),
            "2nd Mode": second_mode,
            "2nd Mode Freq": second_mode_freq,
            "2nd Mode %": round(second_mode_pct, 2)
        }
        cat_summary.append(summary)

    cat_df = pd.DataFrame(cat_summary)
    print(cat_df.to_string(index=False))

def remove_outliers_iqr(df, columns=None):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"🚨 Removed {len(outliers)} outliers from {col}")
        
        # Remove outliers
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def evaluate_models(X, y, models_dict):
    """Evaluate multiple models using cross-validation"""
    print("\n🏆 Model Evaluation Results:")
    print("="*60)
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models_dict.items():
        print(f"\n📊 Evaluating {name}...")
        
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
    
    return results

def plot_model_comparison(results):
    """Plot model comparison"""
    models = list(results.keys())
    mse_means = [results[model]['MSE_mean'] for model in models]
    mae_means = [results[model]['MAE_mean'] for model in models]
    r2_means = [results[model]['R2_mean'] for model in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE comparison
    axes[0].bar(models, mse_means, color='lightcoral')
    axes[0].set_title('Mean Squared Error (MSE)')
    axes[0].set_ylabel('MSE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    axes[1].bar(models, mae_means, color='lightskyblue')
    axes[1].set_title('Mean Absolute Error (MAE)')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[2].bar(models, r2_means, color='lightgreen')
    axes[2].set_title('R² Score')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# 🔹 Load the data
print("🏠 Loading Houses Dataset...")
df = pd.read_csv("houses.csv")

# 🔹 Initial data exploration
print("\n🔍 Initial Data Exploration:")
explore_dataframe(df)

# 🔹 Separate continuous and categorical features
df_continuous, df_categorical = group_by_feature_type(df)
print("\n📊 Feature Type Summary:")
print(f"Continuous features: {len(df_continuous.columns)}")
print(f"Categorical features: {len(df_categorical.columns)}")

# 🔹 Detailed feature summary
summarize_features_v2(df_continuous, df_categorical)

# 🔹 Data preprocessing
print("\n🧹 Data Preprocessing:")

# Analyze missing values in detail
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Missing values per column:")
missing_info = df.isnull().sum()
missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
for col, count in missing_info.items():
    pct = (count / len(df)) * 100
    print(f"   {col}: {count} ({pct:.1f}%)")

# Check if we can safely drop rows with missing values
rows_with_any_missing = df.isnull().any(axis=1).sum()
print(f"\nRows with any missing values: {rows_with_any_missing} ({(rows_with_any_missing/len(df)*100):.1f}%)")

# More intelligent missing value handling
if rows_with_any_missing == len(df):
    print("⚠️ All rows have missing values. Using column-wise strategy...")
    # Drop columns with too many missing values (>50%)
    cols_to_drop = [col for col, count in missing_info.items() if count > len(df) * 0.5]
    if cols_to_drop:
        print(f"Dropping columns with >50% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Fill remaining missing values
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            # Fill categorical with mode
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')
        else:
            # Fill numerical with median
            df[col] = df[col].fillna(df[col].median())
    
    df_clean = df.copy()
    print(f"After intelligent cleaning: {len(df_clean)} rows, {df_clean.isnull().sum().sum()} missing values")
    
elif rows_with_any_missing < len(df) * 0.8:
    # If less than 80% of rows have missing values, drop them
    df_clean = df.dropna()
    print(f"Dropped rows with missing values: {len(df_clean)} rows remaining")
else:
    # If 80%+ rows have missing values, use imputation
    print("Too many rows with missing values. Using imputation...")
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype in ['object', 'category']:
            # Fill categorical with mode
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
            else:
                df_clean[col] = df_clean[col].fillna('Unknown')
        else:
            # Fill numerical with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    print(f"After imputation: {len(df_clean)} rows, {df_clean.isnull().sum().sum()} missing values")

# Final check
if len(df_clean) == 0:
    print("❌ ERROR: No data remaining after preprocessing!")
    print("This suggests all data was missing or invalid.")
    exit(1)

# Remove outliers (optional - uncomment if needed)
# numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
# df_clean = remove_outliers_iqr(df_clean, numeric_cols)

# 🔹 Encode categorical variables
categorical_columns = df_clean.select_dtypes(include=['object']).columns
label_encoders = {}

print(f"\n🏷️ Encoding categorical variables: {list(categorical_columns)}")
for col in categorical_columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
    print(f"   {col}: {le.classes_[:5]}..." if len(le.classes_) > 5 else f"   {col}: {le.classes_}")

# 🔹 Prepare features and target
# Assuming we want to predict house price (you may need to adjust the target column)
target_candidates = ['price', 'Price', 'value', 'Value', 'cost', 'Cost', 'SalePrice', 'sale_price']
target_col = None

print(f"\n🎯 Looking for target column in: {list(df_clean.columns)}")

for col in target_candidates:
    if col in df_clean.columns:
        target_col = col
        break

if target_col is None:
    # If no obvious target found, use the first numeric column as example
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        target_col = numeric_cols[0]
        print(f"⚠️ No obvious target column found. Using '{target_col}' as target for demonstration.")
    else:
        print("❌ ERROR: No numeric columns found for target variable!")
        print("Available columns:", list(df_clean.columns))
        exit(1)

# Features (exclude target and any ID columns)
exclude_cols = [target_col] + [col for col in df_clean.columns if 'id' in col.lower() or 'ID' in col]

# Check if target column exists and has valid data
if target_col not in df_clean.columns:
    print(f"❌ ERROR: Target column '{target_col}' not found!")
    exit(1)

# Check target variable for issues
y = df_clean[target_col].copy()
print(f"Target column '{target_col}' info:")
print(f"   Data type: {y.dtype}")
print(f"   Non-null count: {y.count()}")
print(f"   Null count: {y.isnull().sum()}")
print(f"   Unique values: {y.nunique()}")

if y.isnull().all():
    print(f"❌ ERROR: Target column '{target_col}' has no valid values!")
    exit(1)

# Remove rows where target is null
if y.isnull().any():
    print(f"Removing {y.isnull().sum()} rows with missing target values")
    valid_indices = y.notnull()
    df_clean = df_clean[valid_indices]
    y = y[valid_indices]

X = df_clean.drop(columns=exclude_cols, errors='ignore')

print(f"\n🎯 Target variable: {target_col}")
print(f"📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")
print(f"📊 Feature columns: {list(X.columns)}")

# 🔹 Feature scaling (important for some algorithms)
print(f"\n🔧 Preparing features for scaling...")
print(f"Features shape before scaling: {X.shape}")
print(f"Features columns: {list(X.columns)}")

# Check if we have any data left
if len(X) == 0:
    print("❌ ERROR: No samples available for scaling!")
    print("Dataset is empty after preprocessing.")
    exit(1)

# Check for non-numeric columns that might cause issues
non_numeric_cols = []
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        non_numeric_cols.append(col)

if non_numeric_cols:
    print(f"⚠️ Warning: Found non-numeric columns after encoding: {non_numeric_cols}")
    # Try to convert to numeric
    for col in non_numeric_cols:
        try:
            X[col] = pd.to_numeric(X[col])
            print(f"   Converted {col} to numeric")
        except:
            print(f"   Failed to convert {col}, dropping it")
            X = X.drop(columns=[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
print(f"✅ Scaling completed. Shape: {X_scaled.shape}")

# 🔹 Model evaluation
print("\n🤖 Machine Learning Model Evaluation:")

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'KNN (k=10)': KNeighborsRegressor(n_neighbors=10)
}

# Evaluate models with original features
print("\n📈 Results with Original Features:")
results_original = evaluate_models(X, y, models)

# Evaluate models with scaled features
print("\n📈 Results with Scaled Features:")
results_scaled = evaluate_models(X_scaled, y, models)

# 🔹 Visualize results
print("\n📊 Creating Comparison Plots...")
plot_model_comparison(results_original)

# 🔹 Find best model
print("\n🏆 Best Model Selection:")
best_model_name = min(results_scaled.keys(), key=lambda x: results_scaled[x]['MSE_mean'])
best_results = results_scaled[best_model_name]

print(f"🥇 Best Model: {best_model_name}")
print(f"   MSE: {best_results['MSE_mean']:.4f}")
print(f"   MAE: {best_results['MAE_mean']:.4f}")
print(f"   R²:  {best_results['R2_mean']:.4f}")
print(f"   RMSE: {best_results['RMSE_mean']:.4f}")

# 🔹 Feature importance (for linear regression)
if 'Linear Regression' in models:
    lr_model = LinearRegression()
    lr_model.fit(X_scaled, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_,
        'Abs_Coefficient': np.abs(lr_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\n📊 Feature Importance (Linear Regression Coefficients):")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    sns.barplot(data=top_features, x='Abs_Coefficient', y='Feature', palette='viridis')
    plt.title('Top 10 Most Important Features (Linear Regression)')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.show()

# 🔹 Final predictions example
print("\n🔮 Making Predictions with Best Model:")
best_model = models[best_model_name]
best_model.fit(X_scaled, y)

# Make predictions on first 5 samples
sample_predictions = best_model.predict(X_scaled.head())
actual_values = y.head()

comparison_df = pd.DataFrame({
    'Actual': actual_values.values,
    'Predicted': sample_predictions,
    'Difference': actual_values.values - sample_predictions
})

print("Sample Predictions vs Actual Values:")
print(comparison_df.to_string(index=False))

print("\n✅ Analysis Complete!")
print(f"📊 Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"🎯 Target: {target_col}")
print(f"🏆 Best Model: {best_model_name}")
print(f"📈 Best R² Score: {best_results['R2_mean']:.4f}")