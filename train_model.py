import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv("music_listeners.csv")

# Select the features
features = [
    "listening_hours_per_week",
    "songs_per_day",
    "skip_rate",
    "playlist_count"
]

X = df[features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model
kmeans.fit(X_scaled)

# Add the cluster number to the dataset
df["cluster"] = kmeans.labels_
df.to_csv("final_music_listeners.csv", index=False)

# Display the result
print("Listener Clusters:")
print(df[features + ["cluster"]])

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)
print("\nCluster Interpretation:")
print("\nCluster Counts:")
print(pd.Series(kmeans.labels_).value_counts().sort_index())
print("Cluster 0 - Music Explorers")
print("Cluster 1 - Heavy Listeners")
print("Cluster 2 - Casual Listeners")
silhouette = silhouette_score(X_scaled, kmeans.labels_)

print("\nModel Evaluation:")
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette)

# Save the trained model
joblib.dump(kmeans, "model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")