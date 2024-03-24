import pandas as pd

# 加载和处理CDR3编码
encoded_cdr3_sequences = []
with open('/Users/lifushen/Desktop/1d_cnn/encode.txt', 'r') as file:
    for line in file:
        encoded_sequence = [float(num) for num in line.strip().split()]
        encoded_cdr3_sequences.append(encoded_sequence)

cdr3_encoded_df = pd.DataFrame(encoded_cdr3_sequences)
column_names = [f'feature_{i+1}' for i in range(cdr3_encoded_df.shape[1])]
cdr3_encoded_df.columns = column_names

# 然后，读取原始TCR数据集
df = pd.read_csv("/Users/lifushen/Desktop/1d_cnn/data_clean.csv", sep='\s+', error_bad_lines=False)

# 为V基因和J基因创建One-Hot编码
v_genes_one_hot = pd.get_dummies(df['v.segm'], prefix='V')
j_genes_one_hot = pd.get_dummies(df['j.segm'], prefix='J')

# 因为可能CDR3编码的序列数量与原始数据集中的行数不一致，
# 需要确保这两部分数据在行数上是匹配的。
# 假设cdr3_encoded_df和df具有相同的行数，或者你有一种方法来确保它们对齐。

# 最终的特征数据将是CDR3编码和One-Hot编码的V基因和J基因的拼接
# 注意：这里直接使用cdr3_encoded_df，因为它已经是DataFrame格式
features = pd.concat([cdr3_encoded_df, v_genes_one_hot, j_genes_one_hot], axis=1)

features.dropna(subset=features.columns[:16], how='all', inplace=True)

print(features.head())

features.to_csv('/Users/lifushen/Desktop/1d_cnn/encoded_features.txt',  sep=' ', index=False)
