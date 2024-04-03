import anndata as ad
import scanpy as sc
import collections
import leidenalg
import numpy as np


import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import torch
import pandas as pd
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
import tcr_bert_TwoPartBertClassifier
import tqdm

# 需要改
# device = torch.device("cpu")
# device = torch.device("cuda:1")
device = torch.device("mps")

class TCRDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.cdr3_a_aa = df['cdr3_a_aa'].tolist()
        self.cdr3_b_aa = df['cdr3_b_aa'].tolist()
        # self.epitope = df['encoded_epitopes'].tolist()
        self.labels = df['encoded_epitopes'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cdr3_a_aa_encoding = self.tokenizer(
            self.cdr3_a_aa[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        cdr3_b_aa_encoding = self.tokenizer(
            self.cdr3_b_aa[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids_a': cdr3_a_aa_encoding['input_ids'].flatten(),
            'attention_mask_a': cdr3_a_aa_encoding['attention_mask'].flatten(),
            'input_ids_b': cdr3_b_aa_encoding['input_ids'].flatten(),
            'attention_mask_b': cdr3_b_aa_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def add_space(s):
    return ' '.join(s)

df_alpha_beta = pd.read_csv("./data_clean_small_MusMusculus.csv", sep='\t')
df_alpha_beta = df_alpha_beta.sample(frac=1).reset_index(drop=True)
# df_alpha_beta = df_alpha_beta.iloc[:500]
label_encoder = LabelEncoder()
# df_alpha_beta['antigen_epitope'] = df_alpha_beta['antigen.epitope']
df_alpha_beta['encoded_epitopes'] = label_encoder.fit_transform(df_alpha_beta['antigen.epitope'])
df_alpha_beta['cdr3_a_aa'] = df_alpha_beta['cdr3_a_aa'].apply(add_space)
# df_alpha_beta['TCR'] = df_alpha_beta['cdr3_b_aa']
df_alpha_beta['cdr3_b_aa'] = df_alpha_beta['cdr3_b_aa'].apply(add_space)

def get_encode_data(model, train_loader, device):
    model.to(device)

    output_list_a = []
    output_list_b = []
    label_list = []

    for data in tqdm.tqdm(train_loader, desc="Training Process: "):
        input_ids_a = data['input_ids_a'].to(device)
        attention_mask_a = data['attention_mask_a'].to(device)
        input_ids_b = data['input_ids_b'].to(device)
        attention_mask_b = data['attention_mask_b'].to(device)
        labels = data['labels'].to(device)

        model.eval()
        outputs_a = model(input_ids_a)  # Assuming your model can handle these inputs
        outputs_b = model(input_ids_b)  # Assuming your model can handle these inputs

        embeddings_a = []
        for i, hidden_states in enumerate(outputs_a.last_hidden_state):
            # Assume the sequence does not include special tokens like [CLS], [SEP]
            seq_len = torch.sum(attention_mask_a[i]).item()  # Calculate the length of the actual sequence
            seq_hidden = hidden_states[1:1 + seq_len]  # Exclude [CLS], include only actual tokens

            # Compute the mean across the sequence length dimension
            mean_embedding = seq_hidden.mean(dim=0)

            # Convert tensor to numpy and store
            embeddings_a.append(mean_embedding)

        embeddings_b = []
        for i, hidden_states in enumerate(outputs_b.last_hidden_state):
            # Assume the sequence does not include special tokens like [CLS], [SEP]
            seq_len = torch.sum(attention_mask_b[i]).item()  # Calculate the length of the actual sequence
            seq_hidden = hidden_states[1:1 + seq_len]  # Exclude [CLS], include only actual tokens

            # Compute the mean across the sequence length dimension
            mean_embedding = seq_hidden.mean(dim=0)

            # Convert tensor to numpy and store
            embeddings_b.append(mean_embedding)

        #     # Stack all embeddings into a single numpy array
        # embeddings = np.vstack(embeddings)


        # cls_embedding_b = outputs_b.last_hidden_state[:, 0, :]
        # cls_embedding_a = outputs_a.last_hidden_state[:, 0, :]
        # output_list_a += cls_embedding_a.tolist()
        # output_list_b += cls_embedding_b.tolist()
        for i in range(len(embeddings_a)):
            embeddings_a[i] = embeddings_a[i].tolist()
        for i in range(len(embeddings_b)):
            embeddings_b[i] = embeddings_b[i].tolist()
        output_list_a += embeddings_a
        output_list_b += embeddings_b
        label_list += labels.tolist()

    return output_list_a, output_list_b, label_list


# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")

# 初始化模型、优化器和损失函数
model = BertModel.from_pretrained("wukevin/tcr-bert-mlm-only", add_pooling_layer='cls')

# train_df, test_df = train_test_split(df_alpha_beta, test_size=0.2, random_state=42)

# 实例化数据集和数据加载器
dataset = TCRDataset(df_alpha_beta, tokenizer, 64)

loader = DataLoader(dataset, batch_size=32, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
output_a, output_b, label_list = get_encode_data(model, loader, device)

embeddings = [a + b for a, b in zip(output_a, output_b)]


# Create an anndata object to perform clsutering
embed_adata = ad.AnnData(np.array(embeddings), obs=df_alpha_beta)
# sc.pp.pca(embed_adata, n_comps=50)
# sc.pp.neighbors(embed_adata, use_rep='X')
# # sc.pp.neighbors(embed_adata)
# sc.tl.leiden(embed_adata, resolution=32)
#
# # 计算评价指标
# labels = embed_adata.obs['leiden']
#
# # Silhouette Score
# sil_score = silhouette_score(embed_adata.X, labels)
# print(f'Silhouette Score: {sil_score}')
#
# # Calinski-Harabasz Score
# ch_score = calinski_harabasz_score(embed_adata.X, labels)
# print(f'Calinski-Harabasz Score: {ch_score}')
#
# # Davies-Bouldin Score
# db_score = davies_bouldin_score(embed_adata.X, labels)
# print(f'Davies-Bouldin Score: {db_score}')

# 初始化存储指标的列表
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []
cluster_counts = range(2, 53)

for i in cluster_counts:
    print(f"Clustering for n_clusters = {i}")
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(embed_adata.X)

    # 计算评价指标
    sil_score = silhouette_score(embed_adata.X, labels)
    ch_score = calinski_harabasz_score(embed_adata.X, labels)
    db_score = davies_bouldin_score(embed_adata.X, labels)

    # 存储指标
    silhouette_scores.append(sil_score)
    calinski_harabasz_scores.append(ch_score)
    davies_bouldin_scores.append(db_score)

    # 打印指标
    print(f"Silhouette Score: {sil_score}")
    print(f"Calinski-Harabasz Score: {ch_score}")
    print(f"Davies-Bouldin Score: {db_score}")

# 绘制 Silhouette Score 趋势图
plt.figure(figsize=(8, 6))
plt.plot(cluster_counts, silhouette_scores, label='Silhouette Score', color='blue')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score as a function of k')
plt.grid(True)
plt.show()

# 绘制 Calinski-Harabasz Score 趋势图
plt.figure(figsize=(8, 6))
plt.plot(cluster_counts, calinski_harabasz_scores, label='Calinski-Harabasz Score', color='green')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz Score as a function of k')
plt.grid(True)
plt.show()

# 绘制 Davies-Bouldin Score 趋势图
plt.figure(figsize=(8, 6))
plt.plot(cluster_counts, davies_bouldin_scores, label='Davies-Bouldin Score', color='red')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Score as a function of k')
plt.grid(True)
plt.show()

# outfile = "./cluster_original_small_Mus.txt"
# # Establish groups
# # tcr_groups = embed_adata.obs.groupby("leiden")["TCR"].apply(list)
# clusters_map = collections.defaultdict(list)
# for row in embed_adata.obs.itertuples():
#     clusters_map[row.leiden].append(row.id)
# print(f"Writing {len(clusters_map)} TCR clusters to: {outfile}")
#
# with open(outfile, "w") as sink:
#     for group in clusters_map.values():
#         # sink.write(",".join(group) + "\n")
#         # 使用列表推导式将列表中的每个整数转换为字符串
#         group_str = [str(item) for item in group]
#
#         # 现在可以安全地使用 join() 方法，因为 group_str 只包含字符串
#         sink.write(",".join(group_str) + "\n")