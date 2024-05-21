import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 步骤1: 加载数据
sparse_matrix = pd.read_csv('/Users/lifushen/Desktop/giana_distance_matrix_visual/distance_matrix_b_a.txt', sep='\t', index_col=0)
cdr3_to_epitope = pd.read_csv('/Users/lifushen/Desktop/giana_distance_matrix_visual/map_a_b.csv')

# 步骤2: 提取CDR3序列
cdr3_sequences = sparse_matrix.index

# 步骤3: 映射抗原表位
epitope_map = dict(zip(cdr3_to_epitope['cdr3_a_aa+cdr3_b_aa'], cdr3_to_epitope['antigen.epitope']))
sparse_matrix['epitope'] = [epitope_map.get(seq) for seq in cdr3_sequences]

# 统计每个抗原表位的频率并选择前5个
top_epitopes = sparse_matrix['epitope'].value_counts().index

# 过滤数据只保留前5个抗原表位的记录
filtered_data = sparse_matrix[sparse_matrix['epitope'].isin(top_epitopes)]

# 按照出现次数对抗原表位进行排序
sorted_epitopes = filtered_data['epitope'].value_counts().index

# 将过滤后的稀疏矩阵的数值部分转换为NumPy数组用于t-SNE
data_for_tsne = filtered_data.drop('epitope', axis=1).values

# 步骤4: 应用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data_for_tsne)

# 将t-SNE结果添加回数据框，用于绘图
filtered_data['tsne-2d-one'] = tsne_results[:,0]
filtered_data['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
# 使用5种不同的颜色
colors = ["#3b75af", "#519e3e", "#ef8636", "#8d6ab8", "#FF0000"]
custom_palette = sns.color_palette(colors)

sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="epitope",
    palette=custom_palette,
    data=filtered_data,
    hue_order=sorted_epitopes,
    legend="full",
    s=10
)

plt.title('2-D visualization of CDR3 combination of alpha and beta (top 5 epitopes) using distance matrix')
plt.show()