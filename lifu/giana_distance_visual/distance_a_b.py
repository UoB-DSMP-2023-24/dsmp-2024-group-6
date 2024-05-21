import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 加载稀疏矩阵和映射
sparse_matrix = pd.read_csv('/Users/lifushen/Desktop/distance_visual/Mat_dist_together_M.txt', sep='\t', index_col=0)
cdr3_to_epitope = pd.read_csv('/Users/lifushen/Desktop/distance_visual/map_a_b_M.txt', sep='\t')

# 将cdr3_to_epitope映射转换为字典
epitope_dict = pd.Series(cdr3_to_epitope.epitope.values, index=cdr3_to_epitope.cdr3_a_b).to_dict()

# 使用PCA降维到2维
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(sparse_matrix)

# 将降维后的数据转换为DataFrame，并添加epitope信息
reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'], index=sparse_matrix.index)
reduced_df['Epitope'] = reduced_df.index.map(epitope_dict)

# 可视化降维后的数据
plt.figure(figsize=(12, 10))  # 可以适当调整以适应你的显示需要

scatter = sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Epitope', palette='viridis', s=10)
plt.title('PCA Reduced Sparse Matrix Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# 获取图例句柄和标签
handles, labels = scatter.get_legend_handles_labels()

# 当颜色标签太多时，可以将图例放置在图表外部
plt.legend(handles=handles, labels=labels, title='Epitope', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()
