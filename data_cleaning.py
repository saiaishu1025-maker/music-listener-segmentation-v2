import pandas as pd

# Load the dataset
df = pd.read_csv("music_listeners.csv")

# Check the dataset size
print("Dataset Shape:")
print(df.shape)

# Show column names
print("\nColumn Names:")
print(df.columns)

# Show first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())