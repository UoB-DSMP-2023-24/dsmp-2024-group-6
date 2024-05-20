import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Define the Smith-Waterman alignment algorithm (optimized version assumed)
def optimized_smith_waterman(sequence1, sequence2, match_score=1, mismatch_score=-1, gap_penalty=-1):
    len_seq1, len_seq2 = len(sequence1), len(sequence2)
    matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))
    
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            match = (sequence1[i - 1] == sequence2[j - 1]) * match_score + (sequence1[i - 1] != sequence2[j - 1]) * mismatch_score
            scores = [0,
                      matrix[i-1, j-1] + match,
                      matrix[i-1, j] + gap_penalty,
                      matrix[i, j-1] + gap_penalty]
            matrix[i, j] = max(scores)
            
    return np.max(matrix)

# Parallel comparison function
def compare_sequences(pair):
    seq1, seq2, cluster_name1, cluster_name2 = pair
    score = optimized_smith_waterman(seq1, seq2)
    return cluster_name1, cluster_name2, seq1, seq2, score

def compare_clusters_parallel(cluster_pairs):
    comparisons = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compare_sequences, pair) for pair in cluster_pairs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Comparing Sequences"):
            comparisons.append(future.result())
    return comparisons

# Data loading and preprocessing
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep="\t")
    df['members'] = df['members'].astype(str).str.split('\s+')
    return df

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    filepath = r"C:\Users\james\OneDrive\Documents\Bristol\Mini Project\gitlocal\dsmp-2024-group-6\convergence_groups.txt"  # Update this path to your data file
    df = load_and_preprocess_data(filepath)
    
    # Prepare clusters and names for comparison (simplified example, adjust as needed)
    clusters = df['members'].tolist()
    cluster_pairs = [(seq1, seq2, f"Cluster {i+1}", f"Cluster {j+1}") 
                     for i, seq1 in enumerate(clusters) 
                     for j, seq2 in enumerate(clusters) if i < j]

    # Perform parallel sequence comparisons
    comparison_results = compare_clusters_parallel(cluster_pairs)

    # Creating DataFrame from the comparison results and saving it
    df_comparison = pd.DataFrame(comparison_results, columns=["Cluster 1", "Cluster 2", "Sequence 1", "Sequence 2", "Smith-Waterman Score"])
    df_comparison.to_csv(r"C:\Users\james\OneDrive\Documents\Bristol\Mini Project\Team Code\turbogliph2\output\comparison_results.csv", index=False)
    print("Comparison results saved to comparison_results.csv")
