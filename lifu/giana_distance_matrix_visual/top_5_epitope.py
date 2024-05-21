import pandas as pd

# 加载CSV文件到DataFrame
df = pd.read_csv("/Users/lifushen/Desktop/giana_distance_matrix_visual/data_clean_small_MusMusculus.csv", sep='\t')

# 计算antigen.epitope中每个条目的出现频率
value_counts = df['antigen.epitope'].value_counts()

# 选取出现次数最多的前5个条目的索引
top_five_index = value_counts.head(5).index

# 筛选出antigen.epitope列中属于前5个最多的条目的行
filtered_df = df[df['antigen.epitope'].isin(top_five_index)]

# 保存筛选后的结果到新的CSV文件，保留原始CSV文件中的所有内容
filtered_df.to_csv("/Users/lifushen/Desktop/giana_distance_matrix_visual/data_clean_small_MusMusculus.csv", index=False)
