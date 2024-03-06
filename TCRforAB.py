from multiprocessing import freeze_support
import pandas as pd
from tcrdist.repertoire import TCRrep
from tcrdist.breadth import get_safe_chunk


def process_tcr_data(df, chain_type):
    # filter alpha and beta
    if chain_type == 'alpha':
        df_filtered = df[df['gene'] == "TRA"].copy()
        # alpha
        df_filtered.rename(columns={'cdr3': 'cdr3_a_aa', 'v.segm': 'v_a_gene', 'j.segm': 'j_a_gene'}, inplace=True)
    elif chain_type == 'beta':
        df_filtered = df[df['gene'] == "TRB"].copy()
        # beta
        df_filtered.rename(columns={'cdr3': 'cdr3_b_aa', 'v.segm': 'v_b_gene', 'j.segm': 'j_b_gene'}, inplace=True)
    else:
        raise ValueError("chain_type must be 'alpha' or 'beta'.")

    # prepare data
    df_filtered = df_filtered.reset_index(drop=True)

    # define instances
    tr = TCRrep(cell_df=df_filtered, organism='human', chains=[chain_type], compute_distances=False)
    tr.cpus = 8

    # 计算安全的分块大小
    safe_chunk_size = get_safe_chunk(tr.clone_df.shape[0], tr.clone_df.shape[0], target=10 ** 7)

    # 计算稀疏矩阵距离
    tr.compute_sparse_rect_distances(df=tr.clone_df, radius=50, chunk_size=safe_chunk_size)

    print(f"{chain_type.capitalize()}链CDR3距离计算完成。")
    # return tr.rw_beta  # distance matrix
    if chain_type == 'alpha':
        return tr.rw_alpha
    else:
        return tr.rw_beta


if __name__ == '__main__':
    freeze_support()
    df = pd.read_csv("./data_clean.csv", sep='\t', index_col=0)  # 修改为您的文件路径

    print("处理α链数据...")
    distances_alpha = process_tcr_data(df, 'alpha')
    print(distances_alpha)
    print("处理β链数据...")
    distances_beta = process_tcr_data(df, 'beta')
    print(distances_beta)