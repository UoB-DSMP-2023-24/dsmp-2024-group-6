import torch
import pandas as pd

labels_df = pd.read_csv('/Users/lifushen/Desktop/1d_cnn/antigen.epitope_output.txt', header=None, names=['Label'])
labels_one_hot = pd.get_dummies(labels_df['Label'])

labels_one_hot.to_csv('/Users/lifushen/Desktop/1d_cnn/labels_one_hot.csv', index=False)

# 转换为numpy数组以便之后转换为torch Tensor
labels_one_hot_numpy = labels_one_hot.values

labels_file = '/Users/lifushen/Desktop/1d_cnn/labels_one_hot.csv'
labels = pd.read_csv(labels_file).values
labels_tensor = torch.tensor(labels, dtype=torch.float32)
