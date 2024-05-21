import pandas as pd

# 假设df是你的DataFrame
# 首先，读取数据。这里假设数据是从CSV文件读取的
df = pd.read_csv('/Users/lifushen/Desktop/1d_cnn/output_encoded.csv')

# 复制'cdr3_a_aa'列到第一列位置
df.insert(0, 'cdr3_a_aa_new', df['cdr3_a_aa'])

# 删除原来位置上的'cdr3_a_aa'列，注意调整列的位置索引
df.drop('cdr3_a_aa', axis=1, inplace=True)

# 将新插入的列重命名为原列名
df.rename(columns={'cdr3_a_aa_new': 'cdr3_a_aa'}, inplace=True)

# 保存修改后的DataFrame回文件（如果需要的话）
df.to_csv('/Users/lifushen/Desktop/1d_cnn/giana_final.csv', index=False)

# 查看修改后的DataFrame
print(df.head())