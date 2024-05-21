import pandas as pd

# 读取CSV文件
data = pd.read_csv('/Users/lifushen/Desktop/giana_distance_matrix_visual/output_encoded_a_b.csv')  # 替换成你的CSV文件路径

# 合并 'cdr3_a_aa' 和 'cdr3_b_aa' 列并删除方括号
cdr3_combined = (data['cdr3_a_aa'].astype(str) + ',' + data['cdr3_b_aa'].astype(str)).replace('\[|\]', '', regex=True)

# 合并 'encoded_CDR3a' 和 'encoded_CDR3b' 列并删除方括号
encoded_CDR3_combined = (data['encoded_CDR3a'].astype(str) + ',' + data['encoded_CDR3b'].astype(str)).replace('\[|\]', '', regex=True)

# 删除原始的 'cdr3_a_aa', 'cdr3_b_aa', 'encoded_CDR3a', 'encoded_CDR3b' 列
data.drop(['cdr3_a_aa', 'cdr3_b_aa', 'encoded_CDR3a', 'encoded_CDR3b'], axis=1, inplace=True)

# 插入合并后的列到数据框架的最前面
data.insert(0, 'cdr3_a_aa+cdr3_b_aa', cdr3_combined)
data.insert(1, 'encoded_CDR3a+encoded_CDR3b', encoded_CDR3_combined)

# 保存合并后的数据到新的CSV文件
data.to_csv('merged_cleaned_reordered_data.csv', index=False)
