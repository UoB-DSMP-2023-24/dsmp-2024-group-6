from multiprocessing import freeze_support
import pandas as pd
from tcrdist.repertoire import TCRrep
from tcrdist.breadth import get_safe_chunk

if __name__ == '__main__':
    freeze_support()
    df = pd.read_csv("./data_clean.csv", sep='\t', index_col=0)

    print(df)

    '''data processing'''

    condition = (df['gene'] == "TRB") & (~df['v.segm'].isnull()) & (~df['j.segm'].isnull()) & (
                df['species'] == "HomoSapiens")

    trb_data = df[(df['gene'] == 'TRB') & (~df['v.segm'].isnull()) & (~df['j.segm'].isnull()) & (
                df['species'] == "HomoSapiens")]

    result_df = pd.DataFrame(columns=['id', 'cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene', 'j_a_gene'] +
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

            result_df = pd.concat([result_df, pd.DataFrame.from_records([new_row])])

        # if complex_id > 5:
        #     break

    print(result_df)

    # Define TCRrep
    tr = TCRrep(cell_df=result_df,
                organism='human',
                chains=['alpha', 'beta'],
                compute_distances=False)

    # Set to desired number of CPUs
    tr.cpus = 8

    # Identify a safe chunk size based on input data shape and target number of
    # pairwise distance to be temporarily held in memory per node.
    safe_chunk_size = get_safe_chunk(
        tr.clone_df.shape[0],
        tr.clone_df.shape[0],
        target=10 ** 7)

    tr.compute_sparse_rect_distances(
        df=tr.clone_df,
        radius=50,
        chunk_size=safe_chunk_size)

    print(tr.rw_beta)
