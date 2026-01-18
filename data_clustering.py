import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("churn-bigml-20.csv")

print("Dataset loaded successfully")
print(df.head())

X = df[["Total day minutes", "Customer service calls"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

print("\nCluster value counts:")
print(df["Cluster"].value_counts())

plt.figure(figsize=(6, 4))
plt.scatter(
    df["Total day minutes"],
    df["Customer service calls"],
    c=df["Cluster"]
)
plt.xlabel("Total Day Minutes")
plt.ylabel("Customer Service Calls")
plt.title("Customer Clusters using K-Means")
plt.tight_layout()
plt.savefig("kmeans_clusters.png")
plt.show()

print("\n✅ K-Means clustering completed successfully!")
