from scipy.spatial import distance
import pandas as pd

def get_hamming_dist(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
        # return None

    mismatch_columns = 0

    for c in range(len(seq1)):
        if seq1[c] != seq2[c]:
            mismatch_columns += 1

    return mismatch_columns

def get_alpha_distance(df):
    df_alpha = df[df['gene'] == 'TRA']
    hamming_distances_df_alpha = pd.DataFrame(index=df_alpha.index, columns=df_alpha.index)

    for index1, row1 in df_alpha.iterrows():
        for index2, row2 in df_alpha.iterrows():
            if index1 >= index2:
                continue
            if len(row1['cdr3']) != len(row2['cdr3']):
                continue
            hamming_distances_df_alpha.loc[index1, index2] = get_hamming_dist(row1['cdr3'], row2['cdr3'])

    print(hamming_distances_df_alpha)
    return hamming_distances_df_alpha

def get_beta_distance(df):
    df_beta = df[df['gene'] == 'TRB']
    hamming_distances_df_beta = pd.DataFrame(index=df_beta.index, columns=df_beta.index)

    for index1, row1 in df_beta.iterrows():
        for index2, row2 in df_beta.iterrows():
            if index1 >= index2:
                continue
            if len(row1['cdr3']) != len(row2['cdr3']):
                continue
            hamming_distances_df_beta.loc[index1, index2] = get_hamming_dist(row1['cdr3'], row2['cdr3'])

def get_alpha_beta_distance(df = None):
    if df is None:
        df_alpha_beta = pd.read_csv('./data_clean.csv', sep='\t', index_col=0)

    else:
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
                          (~df['j.segm'].isnull()) & (df['species'] == "HomoSapiens")]

            # 如果找到了相同 complex.id 的 TRA 数据
            if not tra_data.empty:
                # 提取 cdr3、v.segm 和 j.segm 列的值
                cdr3_a_aa = tra_data.iloc[0]['cdr3']
                v_a_gene = tra_data.iloc[0]['v.segm']
                j_a_gene = tra_data.iloc[0]['j.segm']

                # 将提取的值填充至新的 DataFrame 中
                new_row = {'cdr3_b_aa': row['cdr3'], 'v_b_gene': row['v.segm'], 'j_b_gene': row['j.segm'], 'id': complex_id,
                           'cdr3_a_aa': cdr3_a_aa, 'v_a_gene': v_a_gene, 'j_a_gene': j_a_gene}
                # 将原始数据中的其他字段也添加到行数据中
                for field in ['species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope', 'antigen.gene',
                              'antigen.species',
                              'reference.id', 'method', 'meta', 'cdr3fix', 'vdjdb.score', 'web.method', 'web.method.seq',
                              'web.cdr3fix.nc', 'web.cdr3fix.unmp', 'is_cdr3_alpha_valid', 'is_mhc_a_valid']:
                    new_row[field] = row[field]

                # new_row = pd.Series(new_row)

                df_alpha_beta = pd.concat([df_alpha_beta, pd.DataFrame.from_records([new_row])])

    hamming_distances_df_alpha_beta = pd.DataFrame(index=df_alpha_beta.index, columns=df_alpha_beta.index)

    for index1, row1 in df_alpha_beta.iterrows():
        for index2, row2 in df_alpha_beta.iterrows():
            if index1 >= index2:
                continue
            if len(row1['cdr3']) != len(row2['cdr3']):
                continue
            hamming_distances_df_alpha_beta.loc[index1, index2] = get_hamming_dist(row1['cdr3'], row2['cdr3'])


df = pd.read_csv('./data_clean.csv', sep='\t', index_col=0)

print(get_beta_distance(df))






