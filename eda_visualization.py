import pandas as pd
import matplotlib.pyplot as plt

# Load final dataset
df = pd.read_csv("final_music_listeners.csv")

# 1. Listening Hours vs Songs per Day
plt.figure()
plt.scatter(df["listening_hours_per_week"], df["songs_per_day"])
plt.xlabel("Listening Hours per Week")
plt.ylabel("Songs per Day")
plt.title("Listening Hours vs Songs per Day")
plt.show()

# 2. Skip Rate vs Listening Hours
plt.figure()
plt.scatter(df["listening_hours_per_week"], df["skip_rate"])
plt.xlabel("Listening Hours per Week")
plt.ylabel("Skip Rate")
plt.title("Skip Rate vs Listening Hours")
plt.show()

# 3. Number of Listeners in Each Cluster
cluster_counts = df["cluster"].value_counts().sort_index()

plt.figure()
plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
plt.xlabel("Cluster")
plt.ylabel("Number of Listeners")
plt.title("Number of Listeners in Each Cluster")
plt.show()

# Basic Feature Analysis
print("\nFeature Analysis:")
print(df.describe())