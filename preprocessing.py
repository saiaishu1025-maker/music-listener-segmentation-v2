import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("music_listeners.csv")

# Select the features used for ML
features = [
    "listening_hours_per_week",
    "songs_per_day",
    "skip_rate",
    "playlist_count"
]

X = df[features]

# Create the scaler
scaler = StandardScaler()

# Scale the data
X_scaled = scaler.fit_transform(X)

# Display the scaled data
print("Original Data:")
print(X.head())

print("\nScaled Data:")
print(X_scaled[:5])