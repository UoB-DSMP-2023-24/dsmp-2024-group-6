from multiprocessing import freeze_support
import pandas as pd
from tcrdist.repertoire import TCRrep
from tcrdist.breadth import get_safe_chunk

if __name__ == '__main__':
    freeze_support()
    df = pd.read_csv("./data_clean.csv", sep='\t', index_col=0)

    print(df)

    '''data processing'''

    # redefine data
    condition = (df['gene'] == "TRB") & (~df['v.segm'].isnull()) & (~df['j.segm'].isnull()) & (
                df['species'] == "HomoSapiens")

    # filter data
    df2 = df[condition].copy()

    df2 = df2[['cdr3', 'v.segm', 'j.segm']]

    df2 = df2.reset_index().drop(columns='index')

    df2 = df2.rename(columns={'cdr3': 'cdr3_b_aa', 'v.segm': 'v_b_gene', 'j.segm': 'j_b_gene'})

    # Define TCRrep
    tr = TCRrep(cell_df=df2,
                organism='human',
                chains=['beta'],
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
