import pandas as pd
import numpy as np


data_file_path = '/Users/lifushen/Desktop/1d_cnn/data_clean.csv'
cdr3_encoded_file_path = '/Users/lifushen/Desktop/1d_cnn/encoded_cdr3_sequences.txt'  # cdr3
output_file_path = '/Users/lifushen/Desktop/1d_cnn/encoded_features.txt'

df = pd.read_csv(data_file_path, sep='\s+', error_bad_lines=False)

v_genes_one_hot = pd.get_dummies(df['v.segm'], prefix='V')
j_genes_one_hot = pd.get_dummies(df['j.segm'], prefix='J')

cdr3_encoded_df = pd.read_csv(cdr3_encoded_file_path, sep=' ', header=None)

feature_columns = ['feature_' + str(i) for i in range(cdr3_encoded_df.shape[1])]
cdr3_encoded_df.columns = feature_columns

encoded_features = pd.concat([cdr3_encoded_df, v_genes_one_hot, j_genes_one_hot], axis=1)

encoded_features_trimmed = encoded_features.iloc[:85114]

encoded_features_trimmed.to_csv(output_file_path, sep=' ', index=False)

print(f'Trimmed encoded features have been saved to: {output_file_path}')
