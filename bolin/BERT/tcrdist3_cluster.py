from tcrdist.repertoire import TCRrep
from tcrdist3.tcrdist.adpt_funcs import get_basic_centroids
import pandas as pd

df = pd.read_csv('./data_tcrdist3.csv', sep='\t').iloc[:1000]

tr = TCRrep(cell_df=df,
            organism='human',
            chains=['alpha', 'beta'],
            compute_distances=True)

tr = get_basic_centroids(tr, max_dist=200)

print(tr.centroids_df)

from tcrdist.rep_diff import neighborhood_diff, hcluster_diff, member_summ
import hierdiff

res, Z= hcluster_diff(tr.clone_df, tr.pw_beta, x_cols = ['antigen.species'], count_col = 'count', test_method='chi2')

res_summary = member_summ(res_df = res, clone_df = tr.clone_df)

res_detailed = pd.concat([res, res_summary], axis = 1)

html = hierdiff.plot_hclust_props(Z,
            title='PA Epitope Example',
            res=res_detailed,
            tooltip_cols=['cdr3_a_aa','v_a_gene', 'j_a_gene'],
            alpha=0.001, colors = ['blue','gray'],
            alpha_col='pvalue')

"""WRITE HTML TO A DISK"""
with open('influenza_alpha_chain_tree.html', 'w') as fh:
    fh.write(html)

print(html)