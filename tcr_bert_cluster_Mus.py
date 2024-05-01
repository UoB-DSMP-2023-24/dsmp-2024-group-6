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
import anndata as ad
import scanpy as sc
import collections

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

# need to change
# device = torch.device("cpu")
# device = torch.device("cuda:1")
device = torch.device("mps")

model_path = "wukevin/tcr-bert"
data_path = "./data_clean_large_neg.csv"

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


# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained('wukevin/tcr-bert-mlm-only', num_labels=30).to(device)


model = BertModel.from_pretrained(model_path, add_pooling_layer='cls')

# train_df, test_df = train_test_split(df_alpha_beta, test_size=0.2, random_state=42)


dataset = TCRDataset(df_alpha_beta, tokenizer, 64)

data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
# # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# train the model
# output_a, label_list = get_encode_data(model, None, data_loader, device, "cls")
output_a, output_b, label_list = get_encode_data(model, model, data_loader, device, "cls")

merged_list = [a + b for a, b in zip(output_a, output_b)]


# encoded_df = pd.read_csv('CDR3_encoded_dimension.csv')
# cdr3_encoded_strings = encoded_df['cdr3encode']
epitopes = df_alpha_beta['clusterid']

# cdr3_encoded = np.array([np.fromstring(seq, sep=' ', dtype=int) for seq in cdr3_encoded_strings])
#
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(np.array(merged_list))
# pca = PCA(n_components=2, random_state=42)
# X_embedded = pca.fit_transform(np.array(output_a))

epitopes_series = pd.Series(epitopes)
epitope_counts = epitopes_series.value_counts()
top_epitopes = epitope_counts.nlargest(20).index #####

plt.figure(figsize=(10, 8))
# colors = plt.cm.get_cmap('tab20', len(top_epitopes))

for i, epitope in enumerate(top_epitopes):
    indices = np.where(np.array(epitopes) == epitope)
    # if epitope != 'YLQPRTFLL':
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=epitope)

plt.title('MusMusculus Cluster')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Top Epitopes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
