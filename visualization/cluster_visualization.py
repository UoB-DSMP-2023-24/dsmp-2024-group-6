import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE



# Initialize TruncatedSVD
svd = TruncatedSVD(n_components=50)

# apply TruncatedSVD
X_alpha_svd = svd.fit_transform(distances_alpha)


num_clusters = 54
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(X_alpha_svd)

# apply t-SNE
tsne = TSNE(n_components=2)
X_alpha_tsne = tsne.fit_transform(X_alpha_svd)


cluster_sizes = np.bincount(clusters)

#sort
sorted_clusters = np.argsort(cluster_sizes)

middle_index = len(sorted_clusters) // 2
middle_clusters = sorted_clusters[middle_index-2:middle_index+3]

# visualization
def plot_results(X, labels, title, included_clusters):
    plt.figure(figsize=(8, 6))
    for i in included_clusters:
        subset = X[labels == i]
        plt.scatter(subset[:, 0], subset[:, 1], label=f'Cluster {i}')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

plot_results(X_alpha_tsne, clusters, 'Alpha Chain t-SNE Results', middle_clusters)