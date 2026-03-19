import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

# Load dataset
df = sns.load_dataset("iris")

# ----------------------------------------------------
# 1) HANDLING MISSING VALUES
# ----------------------------------------------------

# Iris has no missing values, so we create some for demonstration
df.loc[0, "sepal_length"] = np.nan
df.loc[5, "petal_width"] = np.nan

# Fill numeric missing values with mean
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# ----------------------------------------------------
# 2) ENCODING CATEGORICAL DATA
# ----------------------------------------------------

le = LabelEncoder()
df["species_encoded"] = le.fit_transform(df["species"])

# ----------------------------------------------------
# 3) FEATURE SCALING (StandardScaler)
# ----------------------------------------------------

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

print("Original Data (first 5 rows):")
print(df.head())

print("\nScaled Data (first 5 rows):")
print(df_scaled.head())
