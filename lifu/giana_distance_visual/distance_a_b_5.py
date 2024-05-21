import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE  # 引入TSNE

# 加载稀疏矩阵和映射
sparse_matrix = pd.read_csv('/Users/lifushen/Desktop/distance_visual/Mat_dist_together_M.txt', sep='\t', index_col=0)
cdr3_to_epitope = pd.read_csv('/Users/lifushen/Desktop/distance_visual/map_a_b_M.txt', sep='\t')

# 将cdr3_to_epitope映射转换为字典
epitope_dict = pd.Series(cdr3_to_epitope.epitope.values, index=cdr3_to_epitope.cdr3_a_b).to_dict()

# 使用t-SNE降维到2维
tsne = TSNE(n_components=2)  # 使用TSNE
reduced_data = tsne.fit_transform(sparse_matrix)  # 调用fit_transform来执行降维

# 将降维后的数据转换为DataFrame，并添加epitope信息
reduced_df = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'], index=sparse_matrix.index)
reduced_df['Epitope'] = reduced_df.index.map(epitope_dict)

plt.figure(figsize=(12, 10))

# 计算每个epitope出现的频率，并选取出现频率最高的5个
epitope_counts = reduced_df['Epitope'].value_counts()
top_epitopes = epitope_counts.nlargest(5).index.tolist()

# 创建一个数据子集，只包含频率最高的5个epitope
top_epitopes_df = reduced_df[reduced_df['Epitope'].isin(top_epitopes)]

# 为Top 5 epitopes生成随机且鲜艳的颜色
top_epitope_palette = sns.color_palette("tab10", n_colors=5)

# 绘制散点图，仅为最常见的5个epitope着色
scatter = sns.scatterplot(
    data=top_epitopes_df, x='Dimension 1', y='Dimension 2', hue='Epitope',
    palette=top_epitope_palette, s=10)

plt.title('t-SNE Reduced Sparse Matrix Visualization - Top 5 Epitopes')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# 获取图例句柄和标签
handles, labels = scatter.get_legend_handles_labels()

# 将图例放置在图表外部的左上角
plt.legend(handles=handles, labels=labels, title='Top 5 Epitopes', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()
