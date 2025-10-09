from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"DBSCAN found {unique_clusters} clusters (excluding noise).")


