import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class TCRDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.cdr3_a_aa = df['cdr3_a_aa'].tolist()
        self.cdr3_b_aa = df['cdr3_b_aa'].tolist()
        self.labels = df['antigen.epitope'].tolist()
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
            'labels': self.labels[idx]
        }

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def add_space(s):
    return ' '.join(s)

def summarize_dataframe(df):
    summary = pd.DataFrame({
        'Variable Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    return summary


# def get_encode_data(model, train_loader, device):
#     model.to(device)
#
#     output_list_a = []
#     # output_list_b = []
#     label_list = []
#
#     for data in tqdm.tqdm(train_loader, desc="Training Process: "):
#         input_ids_a = data['input_ids_a'].to(device)
#         attention_mask_a = data['attention_mask_a'].to(device)
#         # input_ids_b = data['input_ids_b'].to(device)
#         # attention_mask_b = data['attention_mask_b'].to(device)
#         labels = data['labels']
#
#         model.eval()
#         outputs_a = model(input_ids_a)  # Assuming your model can handle these inputs
#         # outputs_b = model(input_ids_b)  # Assuming your model can handle these inputs
#
#         embeddings_a = []
#         for i, hidden_states in enumerate(outputs_a.last_hidden_state):
#             # Assume the sequence does not include special tokens like [CLS], [SEP]
#             seq_len = torch.sum(attention_mask_a[i]).item()  # Calculate the length of the actual sequence
#             seq_hidden = hidden_states[1:1 + seq_len]  # Exclude [CLS], include only actual tokens
#
#             # Compute the mean across the sequence length dimension
#             mean_embedding = seq_hidden.mean(dim=0)
#
#             # Convert tensor to numpy and store
#             embeddings_a.append(mean_embedding)
#
#         # embeddings_b = []
#         # for i, hidden_states in enumerate(outputs_b.last_hidden_state):
#         #     # Assume the sequence does not include special tokens like [CLS], [SEP]
#         #     seq_len = torch.sum(attention_mask_b[i]).item()  # Calculate the length of the actual sequence
#         #     seq_hidden = hidden_states[1:1 + seq_len]  # Exclude [CLS], include only actual tokens
#         #
#         #     # Compute the mean across the sequence length dimension
#         #     mean_embedding = seq_hidden.mean(dim=0)
#         #
#         #     # Convert tensor to numpy and store
#         #     embeddings_b.append(mean_embedding)
#
#         #     # Stack all embeddings into a single numpy array
#         # embeddings = np.vstack(embeddings)
#
#
#         # cls_embedding_b = outputs_b.last_hidden_state[:, 0, :]
#         # cls_embedding_a = outputs_a.last_hidden_state[:, 0, :]
#         # output_list_a += cls_embedding_a.tolist()
#         # output_list_b += cls_embedding_b.tolist()
#         for i in range(len(embeddings_a)):
#             embeddings_a[i] = embeddings_a[i].tolist()
#         # for i in range(len(embeddings_b)):
#         #     embeddings_b[i] = embeddings_b[i].tolist()
#         output_list_a += embeddings_a
#         # output_list_b += embeddings_b
#         label_list += labels
#
#     return output_list_a, label_list

def get_encode_data(model_a, model_b, train_loader, device, pooling_strategy = "mean"):
    model_a.to(device)
    output_list_a = []

    if model_b != None:
        model_b.to(device)
        output_list_b = []

    label_list = []


    for data in tqdm.tqdm(train_loader, desc="Getting data Process: "):
        input_ids_a = data['input_ids_a'].to(device)
        attention_mask_a = data['attention_mask_a'].to(device)

        if model_b != None:
            input_ids_b = data['input_ids_b'].to(device)
            attention_mask_b = data['attention_mask_b'].to(device)

        labels = data['labels']

        model_a.eval()

        if model_b != None:
            model_b.eval()

        with torch.no_grad():
            outputs_a = model_a(input_ids_a, output_hidden_states=True)

            if model_b != None:
                outputs_b = model_b(input_ids_b, output_hidden_states=True)

        hidden_states_a = outputs_a.hidden_states
        last_hidden_state_a = hidden_states_a[-1]

        if model_b != None:
            hidden_states_b = outputs_b.hidden_states
            # 获取最后一层的输出
            last_hidden_state_b = hidden_states_b[-1]  # shape: (batch_size, sequence_length, hidden_size)

        # Perform seq pooling
        if pooling_strategy == "mean":
            otuput_last_hidden_state_a = last_hidden_state_a.mean(dim=1)
            if model_b != None:
                output_last_hidden_state_b = last_hidden_state_b.mean(dim=1)
        elif pooling_strategy == "max":
            otuput_last_hidden_state_a = last_hidden_state_a.max(dim=1)[0]
            if model_b != None:
                output_last_hidden_state_b = last_hidden_state_b.max(dim=1)[0]
        elif pooling_strategy == "cls":
            otuput_last_hidden_state_a = last_hidden_state_a[:, 0, :]
            if model_b != None:
                output_last_hidden_state_b = last_hidden_state_b[:, 0, :]

        output_list_a += otuput_last_hidden_state_a.tolist()
        if model_b != None:
            output_list_b += output_last_hidden_state_b.tolist()
        label_list += labels

    if model_b != None:
        return output_list_a, output_list_b, label_list
    else:
        return output_list_a, label_list

# 计算距离矩阵
def distance_matrix(data):
    num_points = len(data)
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i, j] = np.linalg.norm(data[i] - data[j])
    return dist_matrix

# 需要改
# device = torch.device("cpu")
# device = torch.device("cuda:1")
device = torch.device("mps")

model_path = "wukevin/tcr-bert"
data_path = "./data_clean_score_3.csv"

df_alpha_beta = pd.read_csv(data_path, sep='\t')
# df_alpha_beta = df_alpha_beta.iloc[:500]
# label_encoder = LabelEncoder()
# df_alpha_beta = df_alpha_beta[
#                                 # (df_alpha_beta['complex.id'] != 0)
#                               # & (df_alpha_beta['gene'] == 'TRB')
#                               # & (df_alpha_beta['antigen.species'] == 'SARS-CoV-2')
#                                (df_alpha_beta['species'] == 'MusMusculus')
#                               # & (df_alpha_beta['vdjdb.score'] >= 1)
#                               ].copy()

# df_alpha_beta['encoded_epitopes'] = label_encoder.fit_transform(df_alpha_beta['antigen.epitope'])
df_alpha_beta['cdr3_a_aa'] = df_alpha_beta['cdr3_a_aa'].apply(add_space)
df_alpha_beta['cdr3_b_aa'] = df_alpha_beta['cdr3_b_aa'].apply(add_space)

epitope_counts = df_alpha_beta['antigen.epitope'].value_counts()

top_epitopes = epitope_counts.head(5).index

df_alpha_beta = df_alpha_beta[df_alpha_beta['antigen.epitope'].isin(top_epitopes)]

df_alpha_beta = df_alpha_beta.reset_index()

# 假设 df 是你的 DataFrame
# 检查含有特定epitope的行
filtered_df = df_alpha_beta[df_alpha_beta['antigen.epitope'] == top_epitopes[0]]

# 获取这些行的索引
index_list = filtered_df.index.tolist()

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained('wukevin/tcr-bert-mlm-only', num_labels=30).to(device)

# 初始化模型、优化器和损失函数
model = BertModel.from_pretrained(model_path, add_pooling_layer='cls')

# train_df, test_df = train_test_split(df_alpha_beta, test_size=0.2, random_state=42)

# 实例化数据集和数据加载器
dataset = TCRDataset(df_alpha_beta, tokenizer, 64)

data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
# # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 训练模型
# output_a, label_list = get_encode_data(model, None, data_loader, device, "cls")
output_a, output_b, label_list = get_encode_data(model, model, data_loader, device, "mean")

merged_list = [a + b for a, b in zip(output_a, output_b)]

# 使用numpy数组方便计算
data_points = np.array(output_a)
#
# # 生成距离矩阵
# dist_matrix = distance_matrix(data_points)


# 计算余弦相似度矩阵
similarity_matrix = cosine_similarity(data_points)

# 转换为余弦距离矩阵
dist_matrix = 1 - similarity_matrix

# 从距离矩阵中提取特定索引的子矩阵
sub_matrix = dist_matrix[np.ix_(index_list, index_list)]

# 计算子矩阵中的所有距离的平均值
# 我们通常不包括对角线元素，因为对角线上的距离是点到自身的距离，通常为0
mean_distance = np.sum(sub_matrix) / (len(index_list) * (len(index_list) - 1))

print(f"平均距离是: {mean_distance}")

# encoded_df = pd.read_csv('CDR3_encoded_dimension.csv')
# cdr3_encoded_strings = encoded_df['cdr3encode']
epitopes = label_list

# cdr3_encoded = np.array([np.fromstring(seq, sep=' ', dtype=int) for seq in cdr3_encoded_strings])
#
tsne = TSNE(n_components=2, random_state=42, init="random")
X_embedded = tsne.fit_transform(np.array(dist_matrix))
# pca = PCA(n_components=2, random_state=42)
# X_embedded = pca.fit_transform(np.array(output_a))

# epitopes_series = pd.Series(epitopes)
# epitope_counts = epitopes_series.value_counts()
# top_epitopes = epitope_counts.nlargest(5).index #####

plt.figure(figsize=(10, 8))
# colors = plt.cm.get_cmap('tab20', len(top_epitopes))

for i, epitope in enumerate(top_epitopes):
    indices = np.where(np.array(epitopes) == epitope)
    # if epitope != 'YLQPRTFLL':
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=epitope)

name = 'Score3 cos Distance Alpha Mean init'
plt.title(name)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Top Epitopes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

file_name = "./visualization/" + name + '.png'
plt.savefig(file_name, format='png', dpi=300)

plt.show()




