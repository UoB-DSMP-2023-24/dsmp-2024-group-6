import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# 读取CSV文件
csv_file_path = '/Users/lifushen/Desktop/giana_distance_matrix_visual/output_encoded_a.csv'  # 请替换为您的CSV文件路径
df = pd.read_csv(csv_file_path)

# 去除字符串中的换行符和逗号，并分割字符串以形成一个数值列表
split_data = df['encoded_CDR3a'].str.replace('\n', ' ').str.replace('[', '').str.replace(']', '').str.replace(',', ' ').str.split()

# 将分割后的字符串转换为浮点数
split_data = split_data.apply(lambda x: [float(i) for i in x if i])

# 由于列表的长度可能不同，无法直接转换为DataFrame
# 我们首先将其转换为等长的列表
max_len = max(len(l) for l in split_data)
data = np.array([xi + [np.nan]*(max_len-len(xi)) for xi in split_data])

# 使用 pdist 计算欧氏距离
distance_matrix = pdist(data, 'euclidean')

# 将距离矩阵转换为方阵形式
square_distance_matrix = squareform(distance_matrix)

# 添加行名和列名（'cdr3_a_aa+cdr3_b_aa' 的值）
headers = df['cdr3_a_aa'].tolist()
square_distance_matrix = np.vstack([headers, square_distance_matrix])  # 添加头部
square_distance_matrix = np.column_stack([np.insert(headers, 0, ''), square_distance_matrix])  # 添加侧边

# 保存矩阵到TXT文件
np.savetxt('distance_matrix_a.txt', square_distance_matrix, fmt='%s', delimiter='\t')

