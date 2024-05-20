import pandas as pd


df_alpha_beta = pd.read_csv("./data_clean_large.csv", sep='\t')

# df_alpha_beta = df_alpha_beta.drop_duplicates()

df_alpha_beta = df_alpha_beta.drop_duplicates(subset=['cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene',
       'j_a_gene', 'species', 'antigen.epitope'])

print(df_alpha_beta)

# 按照指定列进行分组
groups = df_alpha_beta.groupby(['cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene',
       'j_a_gene', 'species'])

# 存储不同antigen.epitope的分组
different_epitopes = []

# 遍历每个分组
for group_name, group_df in groups:
    # 检查antigen.epitope是否有不同的值
    epitope_values = group_df['antigen.epitope'].unique()
    if len(epitope_values) > 1:
        different_epitopes.append(group_df)
#
# # 输出存在不同antigen.epitope的分组
# if len(different_epitopes) > 0:
#     print("存在不同antigen.epitope的分组:")
#     for group in different_epitopes:
#         print(group)
# else:
#     print("不存在不同antigen.epitope的分组")


for content in different_epitopes:
    if(len(content)) > 6:
        print(content)