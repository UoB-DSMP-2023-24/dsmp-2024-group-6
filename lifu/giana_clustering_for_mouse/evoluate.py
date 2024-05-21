import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import GIANA_encode

Ndim = 16  ## optimized for isometric embedding
n0 = Ndim * 6
# M0=np.concatenate((np.concatenate((ZERO,M1),axis=1),np.concatenate((M1, ZERO),axis=1)))
ZERO = np.zeros((Ndim, Ndim))
II = np.eye(Ndim)
M0 = np.concatenate((np.concatenate((ZERO, ZERO, II), axis=1), np.concatenate((II, ZERO, ZERO), axis=1),
                     np.concatenate((ZERO, II, ZERO), axis=1)))
## Construct 6-th order cyclic group
ZERO45 = np.zeros((Ndim * 3, Ndim * 3))
M6 = np.concatenate((np.concatenate((ZERO45, M0), axis=1), np.concatenate((M0, ZERO45), axis=1)))


# 读取数据，忽略注释行，并为列指定名称
def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    clean_lines = [line for line in lines if not line.startswith('##')]
    data = pd.DataFrame([line.strip().split('\t') for line in clean_lines],
                        columns=['cdr3_a_aa', 'cluster.id','antigen.epitope'])
    return data


# 根据CDR3长度分组并计算评分
def evaluate_scores(data):
    # 计算每个cdr3_a_aa的长度
    data['CDR3_length'] = data['cdr3_a_aa'].apply(len)

    # 分组并计算每组的聚类效果评分
    scores = []
    for length, group in data.groupby('CDR3_length'):
        # 取出聚类标签
        labels = group['cluster.id']
        encoded_features = np.stack(group['cdr3_a_aa'].apply(GIANA_encode.EncodingCDR3, args=(M6, n0)))

        # 检查是否有足够的聚类来计算分数
        if len(set(labels)) < 2:
            print(f"Skipping length {length} due to insufficient number of clusters.")
            continue

        # 计算评分
        silhouette = silhouette_score(encoded_features, labels)
        calinski_harabasz = calinski_harabasz_score(encoded_features, labels)
        davies_bouldin = davies_bouldin_score(encoded_features, labels)

        scores.append((length, silhouette, calinski_harabasz, davies_bouldin))

    # 返回所有分组的评分
    return pd.DataFrame(scores, columns=['CDR3_length', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'])


# 主函数
def main():
    filename = '/Users/lifushen/Desktop/giana_for_mouse/clean_altogther_clean---DualChainRotationEncodingBL62.txt'
    data = load_data(filename)
    scores_df = evaluate_scores(data)
    print("Scores by CDR3 length:")
    print(scores_df)

    # 计算平均值
    if not scores_df.empty:
        avg_silhouette = scores_df['Silhouette'].mean()
        avg_calinski_harabasz = scores_df['Calinski-Harabasz'].mean()
        avg_davies_bouldin = scores_df['Davies-Bouldin'].mean()

        print("\nAverage Scores:")
        print(f"Average Silhouette Score: {avg_silhouette}")
        print(f"Average Calinski-Harabasz Score: {avg_calinski_harabasz}")
        print(f"Average Davies-Bouldin Score: {avg_davies_bouldin}")
    else:
        print("No valid scores to calculate averages.")


if __name__ == '__main__':
    main()
