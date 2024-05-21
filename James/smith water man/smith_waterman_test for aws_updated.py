import numpy as np
import pandas as pd

# Smith-Waterman algorithm implementation
def smith_waterman(sequence1, sequence2, match_score=3, mismatch_score=-3, gap_penalty=-2):
    m, n = len(sequence1), len(sequence2)
    current_row = np.zeros((n+1), dtype=int)
    previous_row = np.zeros((n+1), dtype=int)
    max_score = 0

    for i in range(1, m + 1):
        current_row, previous_row = previous_row, current_row
        for j in range(1, n + 1):
            if sequence1[i-1] == sequence2[j-1]:
                score = previous_row[j-1] + match_score
            else:
                score = previous_row[j-1] + mismatch_score
            score = max(0, score, previous_row[j] + gap_penalty, current_row[j-1] + gap_penalty)
            current_row[j] = score
            max_score = max(max_score, score)
    return max_score

# Function to compare sequences within clusters and between clusters
def compare_clusters(cluster1, cluster2, cluster_name1, cluster_name2):
    for i in range(len(cluster1)):
        for j in range(i+1, len(cluster1)):
            score = smith_waterman(cluster1[i], cluster1[j])
            yield (cluster_name1, cluster_name1, cluster1[i], cluster1[j], score)
    for i in range(len(cluster2)):
        for j in range(i+1, len(cluster2)):
            score = smith_waterman(cluster2[i], cluster2[j])
            yield (cluster_name2, cluster_name2, cluster2[i], cluster2[j], score)
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            score = smith_waterman(cluster1[i], cluster2[j])
            yield (cluster_name1, cluster_name2, cluster1[i], cluster2[j], score)



def split_data_into_subsets(clusters, n):
    """Splits the list of clusters into n subsets."""
    for i in range(0, len(clusters), n):
        yield clusters[i:i + n]







# Preprocessing steps
#df = pd.read_csv("/home/ubuntu/convergence_groups.txt", sep="\t")
df = pd.read_csv(r"C:\Users\james\OneDrive\Documents\Bristol\Mini Project\gitlocal\dsmp-2024-group-6\convergence_groups.txt", sep="\t")


df['members'] = df['members'].astype(str)
df['members'] = df['members'].str.split('\s')
df_local = df[df['type'] == 'local']
df_global = df[df['type'] == 'global']
df_global = df_global.drop(columns=['type', 'cluster_size', 'unique_cdr3_sample', 'unique_cdr3_ref', 'OvE'])
df_global['index'] = range(0, len(df_global))

clusters = df_global['members'].tolist()  # Ensure clusters is a list of lists

comparison_results = []

# Example subset size - this could be adjusted based on your memory constraints
SUBSET_SIZE = 100  # Adjust based on your system's memory capacity and dataset size

clusters_subsets = list(split_data_into_subsets(clusters, SUBSET_SIZE))

for i, subset1 in enumerate(clusters_subsets):
    for j, subset2 in enumerate(clusters_subsets[i:], start=i):
        for cluster1 in subset1:
            for cluster2 in (subset2 if i != j else subset2[subset1.index(cluster1)+1:]):
                cluster_name1 = f"Cluster {clusters.index(cluster1)+1}"
                cluster_name2 = f"Cluster {clusters.index(cluster2)+1}"
                print(f"Comparing {cluster_name1} with {cluster_name2}:")
                comparison_results.extend(compare_clusters(cluster1, cluster2, cluster_name1, cluster_name2))


# Creating DataFrame from the comparison results
df_comparison = pd.DataFrame(comparison_results, columns=["Cluster 1", "Cluster 2", "Sequence 1", "Sequence 2", "Smith-Waterman Score"])

df_comparison.head()

# Save DataFrame to CSV
#df_comparison.to_csv("/home/ubuntu/global_comparison_results.csv", index=False)
df_comparison.to_csv(r"C:\Users\james\OneDrive\Documents\Bristol\Mini Project\gitlocal\dsmp-2024-group-6\global_comparison_results.csv", index=False)
