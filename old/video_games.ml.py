import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def explore_dataframe(df):
    print("🧠 DataFrame Shape:")
    print(df.shape)
    print("\n📌 Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n🕳️ Missing Values per Column:")
    print(df.isnull().sum())
    
    print("\n📊 Summary Statistics (Numerical):")
    print(df.describe())
    
    print("\n🔤 Summary Statistics (Categorical):")
    print(df.describe(include='object'))
    
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

# 🔹 Įkeli duomenis
df = pd.read_csv("june18/vgsales.csv")

# explore_dataframe(df)

# 🔹 Atsikratome išeinančių reikšmių arba nereikalingų duomenų, jei yra
df_clean = merge_duplicate_games(df)
print(df.shape)
print(df_clean.shape)

df_clean = df_clean.dropna()

# explore_dataframe(df_clean)

# 🔹 Užkoduojame kategorinius kintamuosius
label_cols = ['Platform', 'Genre', 'Publisher']
for col in label_cols:
    df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

# 🧹 Pašalink stulpelius, kurių nenaudosi
X = df_clean.drop(columns=['Rank', 'Name'])  # 'Name' neturi reikšmės regresijai
y = df_clean['Rank']

# 🔁 K-Fold Cross-Validation
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

print("Vidutinis MSE:", -scores.mean())

