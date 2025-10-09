from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data = load_iris()
X = data.data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")