import pandas as pd
import numpy as np

# 假设你已经加载了原始数据到DataFrame中
df = pd.read_csv('/Users/lifushen/Desktop/1d_cnn/clean_altogther-DualChainRotationEncodingBL62.csv')

# 这里假设df是你已经有的数据
# 我们会在现有数据上添加一个'1'的binder列
df['binder'] = 1

# 创建一个空的DataFrame用来收集所有伪造的数据
df_fake_total = pd.DataFrame()

# 循环5次来创建伪造的DataFrame，每次都打乱不同的列
np.random.seed(42)  # 保证每次运行代码时打乱方式都相同
for _ in range(5):
    df_fake = df.copy().reset_index(drop=True)
    df_fake['cdr3_a_aa'] = np.random.permutation(df['cdr3_a_aa'])
    df_fake['cdr3_b_aa'] = np.random.permutation(df['cdr3_b_aa'])
    df_fake['cluster.id'] = np.random.permutation(df['cluster.id'])
    df_fake['binder'] = 0
    df_fake_total = pd.concat([df_fake_total, df_fake], ignore_index=True)

# 合并原始的和所有伪造的DataFrame
df_combined = pd.concat([df, df_fake_total]).reset_index(drop=True)

# 如果需要导出为CSV
df_combined.to_csv('data_clean_cluster_neg.csv', index=False)

# 显示前几行以检查
print(df_combined.head())
