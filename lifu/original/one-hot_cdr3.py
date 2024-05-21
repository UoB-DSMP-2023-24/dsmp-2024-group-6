import numpy as np
import pandas as pd

data_file_path = '/Users/lifushen/Desktop/1d_cnn/clean_combine.txt'

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

data = pd.read_csv(data_file_path, sep=" ", header=None, names=["cdr3", "v_gene"])

cdr3_sequences = data["cdr3"].values

max_length = max(len(seq) for seq in cdr3_sequences)

# One-Hot编码函数
def cdr3_to_one_hot(seq, aa_to_index, max_length):
    one_hot = np.zeros((max_length, len(aa_to_index)), dtype=int)
    for i, aa in enumerate(seq):
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1
    return one_hot.flatten()

one_hot_encoded_sequences = np.array([cdr3_to_one_hot(seq, aa_to_index, max_length) for seq in cdr3_sequences])

output_file_path = 'encoded_cdr3_sequences.txt'
np.savetxt(output_file_path, one_hot_encoded_sequences, fmt='%d')

print(f"One-Hot encoded sequences have been saved to: {output_file_path}")
