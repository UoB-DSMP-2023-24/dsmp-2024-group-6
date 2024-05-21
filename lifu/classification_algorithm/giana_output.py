import pandas as pd

# 加载数据集
df = pd.read_csv('/Users/lifushen/Desktop/1d_cnn/output_encoded.csv')  # 确保文件路径正确

# 显示原始数据集的前几行
print("原始数据集的前几行:")
print(df.head())

# 替换列
df.iloc[:, 0] = df.iloc[:, 7]  # 将第 8 列的数据复制到第 1 列（列索引从 0 开始）
df.iloc[:, 3] = df.iloc[:, 8]  # 将第 9 列的数据复制到第 4 列

df = df.iloc[:, :-2]

# 显示修改后数据集的前几行
print("\n修改后数据集的前几行:")
print(df.head())

# 可选：保存修改后的数据集到新的 CSV 文件
df.to_csv('/Users/lifushen/Desktop/1d_cnn/giana_data.csv', index=False)
