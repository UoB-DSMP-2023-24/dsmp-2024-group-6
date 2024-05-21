import pandas as pd

input_txt_path = '/Users/lifushen/Desktop/giana1/output_data.txt'
output_txt_path = '/Users/lifushen/Desktop/giana1/clean_alpha.txt'  # 输出文件的路径和文件名

# 使用pandas读取TXT文件，这里假定使用制表符（'\t'）作为分隔符
df = pd.read_csv(input_txt_path, sep='\t', usecols=['cdr3', 'v.segm'])

# 仅保留v.segm列以"TRBV"开头的行
df_filtered = df[df['v.segm'].astype(str).str.startswith('TRAV')]

# 保存处理后的DataFrame到新的TXT文件，同样使用制表符作为字段分隔符
df_filtered.to_csv(output_txt_path, index=False, sep='\t', header=False)

print(f"处理后的文件已保存到：{output_txt_path}")
