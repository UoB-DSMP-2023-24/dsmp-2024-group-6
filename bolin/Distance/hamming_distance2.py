import numpy as np
from scipy.spatial import distance
import pandas as pd


def get_hamming_dist(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    # 使用向量化操作计算汉明距离
    return np.sum(np.array(list(seq1)) != np.array(list(seq2)))

def get_distance(data):
    n = len(data)
    dist_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(0, i):
            if len(data[i]) != len(data[j]):
                continue
            dist_matrix[i, j] = get_hamming_dist(data[i], data[j])
    return dist_matrix

def get_alpha_beta_distance(data = None):
    if data is None:
        df_alpha_beta = pd.read_csv('./data_tcrdist3.csv', sep='\t', index_col=0)
    else:
        df_alpha_beta = process_data(data)

    cdr3_alpha = df_alpha_beta['cdr3_a_aa'].values
    cdr3_beta = df_alpha_beta['cdr3_b_aa'].values
    dist_matrix_alpha = get_distance(cdr3_alpha)
    dist_matrix_beta = get_distance(cdr3_beta)

    return dist_matrix_alpha, dist_matrix_beta


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
                      (~df['j.segm'].isnull()) & (df['species'] == "HomoSapiens")]

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

    return df_alpha_beta


df = pd.read_csv('./data_clean.csv', sep='\t', index_col=0)
cdr3_alpha = df[df['gene'] == 'TRA']['cdr3'].values
cdr3_beta = df[df['gene'] == 'TRB']['cdr3'].values

# cdr3_beta = cdr3_beta[:100]
# xxx = get_distance(cdr3_beta)
# print(xxx)
# print(get_alpha_distance(cdr3_alpha))
print(get_alpha_beta_distance())
