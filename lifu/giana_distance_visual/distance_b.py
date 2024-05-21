import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE  # 引入TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载稀疏矩阵和映射
sparse_matrix = pd.read_csv('/Users/lifushen/Desktop/distance_visual/Mat_dist_beta_M.txt', sep='\t', index_col=0)
cdr3_to_epitope = pd.read_csv('/Users/lifushen/Desktop/distance_visual/map_b_M.txt', sep='\t')

# 检查并处理 NaN 值
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
sparse_matrix_imputed = imputer.fit_transform(sparse_matrix)

# 使用t-SNE降维到2维
tsne = TSNE(n_components=2)  # 使用TSNE而不是PCA
reduced_data = tsne.fit_transform(sparse_matrix_imputed)  # 调用fit_transform来执行降维

# 将降维后的数据转换为DataFrame，并添加epitope信息
reduced_df = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'], index=sparse_matrix.index)
epitope_dict = pd.Series(cdr3_to_epitope.epitope.values, index=cdr3_to_epitope.cdr3_b).to_dict()
reduced_df['Epitope'] = reduced_df.index.map(epitope_dict)

# 计算每个epitope出现的频率，并选取出现频率最高的10个
top_epitopes = reduced_df['Epitope'].value_counts().nlargest(10).index

# 创建一个只包含最常见的10个epitope的数据子集
top_epitopes_df = reduced_df[reduced_df['Epitope'].isin(top_epitopes)]

# 为Top 10 epitopes选择一个颜色调色板
palette = sns.color_palette("deep", n_colors=100)  # 修改颜色数量以匹配top_epitopes的数量

# 可视化降维后的数据
plt.figure(figsize=(12, 10))
scatter = sns.scatterplot(
    data=top_epitopes_df,
    x='Dimension 1', y='Dimension 2',
    hue='Epitope',
    palette=palette,
    s=10,
)

plt.title('t-SNE Reduced Sparse Matrix Visualization For Beta - Epitopes')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# 获取图例句柄和标签
handles, labels = scatter.get_legend_handles_labels()

# 将图例放置在图表外部的左上角
plt.legend(handles=handles, labels=labels, title='Top 10 Epitopes', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()
