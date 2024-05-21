import pandas as pd

input_csv_path = '/Users/lifushen/Desktop/giana_for_mouse/data_clean_small_MusMusculus.csv'   # 输入文件的路径和文件名
output_txt_path = '/Users/lifushen/Desktop/giana_for_mouse/clean_altogther.txt'  # 输出文件的路径和文件名，扩展名改为.txt

# 使用pandas读取CSV文件，这里假定第一行是字段名，逗号（','）作为分隔符
df = pd.read_csv(input_csv_path, usecols=['cdr3_a_aa', 'cdr3_b_aa', 'antigen.epitope'], sep='\t')

# 保存处理后的DataFrame到新的TXT文件，使用制表符作为字段分隔符，并包含字段名
df.to_csv(output_txt_path, index=False, sep='\t', header=True)

print(f"处理后的文件已保存到：{output_txt_path}")
