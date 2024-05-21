import pandas as pd

# 加载CSV文件
df = pd.read_csv('/Users/lifushen/Desktop/1d_cnn/data_clean.csv',  sep='\t',error_bad_lines=False)

# 提取列
column_data = df['antigen.epitope']

# 将提取的列保存为TXT文件
column_data.to_csv('antigen.epitope_output.txt', index=False, header=None)