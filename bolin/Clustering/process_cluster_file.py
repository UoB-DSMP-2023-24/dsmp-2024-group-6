import pandas as pd

df_alpha_beta = pd.read_csv("./data_clean_large_neg.csv", sep='\t')

cluster_dict = {}
with open('./cluster_original_large.txt', 'r') as file:
    for cluster_id, line in enumerate(file):
        ids = line.strip().split(',')
        for id in ids:
            cluster_dict[int(id)] = cluster_id

df_alpha_beta['id'] = df_alpha_beta['id'].astype(int)

# 使用map函数根据cluster_dict将每个id映射到其相应的clusterid
df_alpha_beta['clusterid'] = df_alpha_beta['id'].map(cluster_dict)

print(df_alpha_beta)

df_alpha_beta.to_csv("data_cluter_large_neg.csv", sep='\t', index=False)
