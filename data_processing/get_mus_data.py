import pandas as pd


df = pd.read_csv("./data_clean.csv", sep='\t', index_col=0)

print(df.columns)

df = df.sample(frac=1, random_state=42)

filtered_df = df[
    (df['is_cdr3_alpha_valid'] == 1) &
    # (df['is_mhc_a_valid'] == 1) &
    (df['cdr3'].notna()) &
    (df['v.segm'].notna()) &
    (df['j.segm'].notna()) &
    (df['complex.id'] != 0) &
    (df['species'] == 'MusMusculus') &
    (df['vdjdb.score'] >= 1)
]

def summarize_dataframe(df):
    summary = pd.DataFrame({
        'Variable Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    return summary

print(summarize_dataframe(filtered_df))

def process_data(df):
    trb_data = df[(df['gene'] == 'TRB') & (~df['v.segm'].isnull()) & (~df['j.segm'].isnull()) & (df['complex.id'] != 0)]

    df_alpha_beta = pd.DataFrame(
        columns=['id', 'cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene', 'j_a_gene'] +
                ['species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope', 'antigen.gene',
                 'antigen.species', 'reference.id', 'method', 'meta', 'cdr3fix', 'vdjdb.score',
                 'web.method', 'web.method.seq', 'web.cdr3fix.nc', 'web.cdr3fix.unmp',
                 'is_cdr3_alpha_valid', 'is_mhc_a_valid'])

    for index, row in trb_data.iterrows():
        # 获取当前行的 complex.id
        complex_id = row['complex.id']

        # 在原始 DataFrame 中查找相同 complex.id 的 TRA 数据
        tra_data = df[(df['complex.id'] == complex_id) & (df['gene'] == 'TRA') & (~df['v.segm'].isnull()) &
                      (~df['j.segm'].isnull())]

        # 如果找到了相同 complex.id 的 TRA 数据
        if not tra_data.empty:
            # 提取 cdr3、v.segm 和 j.segm 列的值
            cdr3_a_aa = tra_data.iloc[0]['cdr3']
            v_a_gene = tra_data.iloc[0]['v.segm']
            j_a_gene = tra_data.iloc[0]['j.segm']

            # 将提取的值填充至新的 DataFrame 中
            new_row = {'cdr3_b_aa': row['cdr3'], 'v_b_gene': row['v.segm'], 'j_b_gene': row['j.segm'],
                       'id': complex_id,
                       'cdr3_a_aa': cdr3_a_aa, 'v_a_gene': v_a_gene, 'j_a_gene': j_a_gene}
            # 将原始数据中的其他字段也添加到行数据中
            for field in ['species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope', 'antigen.gene',
                          'antigen.species',
                          'reference.id', 'method', 'meta', 'cdr3fix', 'vdjdb.score', 'web.method',
                          'web.method.seq',
                          'web.cdr3fix.nc', 'web.cdr3fix.unmp', 'is_cdr3_alpha_valid', 'is_mhc_a_valid']:
                new_row[field] = row[field]

            # new_row = pd.Series(new_row)

            df_alpha_beta = pd.concat([df_alpha_beta, pd.DataFrame.from_records([new_row])])

        else:
            print("error")

    return df_alpha_beta

df_alpha_beta = process_data(filtered_df)
# df_alpha_beta_2 = process_data(filtered_df)

df_alpha_beta.to_csv("data_clean_small_MusMusculus.csv", sep='\t', index=False)
# df_alpha_beta_2.to_csv("data_clean_large.csv", sep='\t', index=False)