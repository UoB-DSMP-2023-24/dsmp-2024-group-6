from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np

# 假设 distances_alpha, distances_beta, distances_combined 是稀疏矩阵
# 这里是如何初始化这些矩阵的示例代码。请用您的实际数据替换这部分。
# distances_alpha = ...
# distances_beta = ...
# distances_combined = ...

# 初始化 TruncatedSVD
svd = TruncatedSVD(n_components=2)

# 应用 TruncatedSVD 降维
X_alpha_svd = svd.fit_transform(distances_alpha)
X_beta_svd = svd.fit_transform(distances_beta)

# 创建示例标签
num_samples_alpha = X_alpha_svd.shape[0]
num_samples_beta = X_beta_svd.shape[0]

spec_labels_alpha = np.array(['Type 1'] * (num_samples_alpha // 2) + ['Type 2'] * (num_samples_alpha - num_samples_alpha // 2))
spec_labels_beta = np.array(['Type 1'] * (num_samples_beta // 2) + ['Type 2'] * (num_samples_beta - num_samples_beta // 2))

# 可视化函数
def plot_svd_results(X, labels, title):
    plt.figure(figsize=(8, 6))
    for i in np.unique(labels):
        subset = X[labels == i]
        plt.scatter(subset[:, 0], subset[:, 1], label=i)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

# 绘制结果
plot_svd_results(X_alpha_svd, spec_labels_alpha, 'Alpha Chain TruncatedSVD')
plot_svd_results(X_beta_svd, spec_labels_beta, 'Beta Chain TruncatedSVD')