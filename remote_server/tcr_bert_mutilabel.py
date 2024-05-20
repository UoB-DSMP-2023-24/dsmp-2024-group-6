import datetime
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.multiclass import OneVsRestClassifier
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

class TCRDataset_a(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.cdr3_a_aa = df['cdr3_a_aa'].tolist()
        self.cdr3_b_aa = df['cdr3_b_aa'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

    def __len__(self):
        return len(self.cdr3_a_aa)

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

        # 确保张量格式正确（删除.squeeze(0)）
        return {
            'input_ids': cdr3_a_aa_encoding['input_ids'][0],  # [0]去掉多余的批次维度
            'attention_mask': cdr3_a_aa_encoding['attention_mask'][0],
            'labels': cdr3_a_aa_encoding['input_ids'][0].clone()  # 这里标签初设为input_ids的复制
        }
    
class TCRDataset_b(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.cdr3_a_aa = df['cdr3_a_aa'].tolist()
        self.cdr3_b_aa = df['cdr3_b_aa'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

    def __len__(self):
        return len(self.cdr3_a_aa)

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

        # 确保张量格式正确（删除.squeeze(0)）
        return {
            'input_ids': cdr3_b_aa_encoding['input_ids'][0],  # [0]去掉多余的批次维度
            'attention_mask': cdr3_b_aa_encoding['attention_mask'][0],
            'labels': cdr3_b_aa_encoding['input_ids'][0].clone()  # 这里标签初设为input_ids的复制
        }

class TCRDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.cdr3_a_aa = df['cdr3_a_aa'].tolist()
        self.cdr3_b_aa = df['cdr3_b_aa'].tolist()
        # self.epitope = df['encoded_epitopes'].tolist()
        self.labels = df['encoded_epitopes'].tolist()
        # self.labels = df['binder'].tolist()
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
    model_b.to(device)

    output_list_a = []
    output_list_b = []
    label_list = []


    for data in tqdm.tqdm(train_loader, desc="Getting data Process: "):
        input_ids_a = data['input_ids_a'].to(device)
        attention_mask_a = data['attention_mask_a'].to(device)
        input_ids_b = data['input_ids_b'].to(device)
        attention_mask_b = data['attention_mask_b'].to(device)
        labels = data['labels'].to(device)

        model_a.eval()
        model_b.eval()

        with torch.no_grad():
            # outputs_a = model_a(input_ids_a, output_hidden_states=True)
            outputs_a = None

            outputs_b = model_b(input_ids_b, output_hidden_states=True)

        # hidden_states_a = outputs_a.hidden_states

        hidden_states_b = outputs_b.hidden_states

        # Assuming your model can handle these inputs
        # outputs_b = model(input_ids_b)  # Assuming your model can handle these inputs

        # embeddings_a = []
        # for i, hidden_states in enumerate(outputs_a.last_hidden_state):
        #     # Assume the sequence does not include special tokens like [CLS], [SEP]
        #     seq_len = torch.sum(attention_mask_a[i]).item()  # Calculate the length of the actual sequence
        #     seq_hidden = hidden_states[1:1 + seq_len]  # Exclude [CLS], include only actual tokens

        #     # Compute the mean across the sequence length dimension
        #     mean_embedding = seq_hidden.mean(dim=0)

        #     # Convert tensor to numpy and store
        #     embeddings_a.append(mean_embedding)

        # embeddings_b = []
        # for i, hidden_states in enumerate(outputs_b.last_hidden_state):
        #     # Assume the sequence does not include special tokens like [CLS], [SEP]
        #     seq_len = torch.sum(attention_mask_b[i]).item()  # Calculate the length of the actual sequence
        #     seq_hidden = hidden_states[1:1 + seq_len]  # Exclude [CLS], include only actual tokens

        #     # Compute the mean across the sequence length dimension
        #     mean_embedding = seq_hidden.mean(dim=0)

        #     # Convert tensor to numpy and store
        #     embeddings_b.append(mean_embedding)

        # #     # Stack all embeddings into a single numpy array
        # # embeddings = np.vstack(embeddings)

        # # cls_embedding_b = outputs_b.last_hidden_state[:, 0, :]
        # # cls_embedding_a = outputs_a.last_hidden_state[:, 0, :]
        # # output_list_a += cls_embedding_a.tolist()
        # # output_list_b += cls_embedding_b.tolist()
        # for i in range(len(embeddings_a)):
        #     embeddings_a[i] = embeddings_a[i].tolist()
        # for i in range(len(embeddings_b)):
        #     embeddings_b[i] = embeddings_b[i].tolist()

        # 获取最后一层的输出
        # last_hidden_state_a = hidden_states_a[-1]  # shape: (batch_size, sequence_length, hidden_size)
        last_hidden_state_b = hidden_states_b[-1]  # shape: (batch_size, sequence_length, hidden_size)

        # # 计算沿序列长度的平均值，得到每个样本的特征向量
        # otuput_last_hidden_state_a = last_hidden_state_a.mean(dim=1)
        # output_last_hidden_state_b = last_hidden_state_b.mean(dim=1)

        # Perform seq pooling
        if pooling_strategy == "mean":
            # otuput_last_hidden_state_a = last_hidden_state_a.mean(dim=1)
            output_last_hidden_state_b = last_hidden_state_b.mean(dim=1)
        elif pooling_strategy == "max":
            # otuput_last_hidden_state_a = last_hidden_state_a.max(dim=1)[0]
            output_last_hidden_state_b = last_hidden_state_b.max(dim=1)[0]
        elif pooling_strategy == "cls":
            # otuput_last_hidden_state_a = last_hidden_state_a[:, 0, :]
            output_last_hidden_state_b = last_hidden_state_b[:, 0, :]

        # output_list_a += otuput_last_hidden_state_a.tolist()
        output_list_a = None
        output_list_b += output_last_hidden_state_b.tolist()
        label_list += labels.tolist()

    return output_list_a, output_list_b, label_list


def print_evaluation(period=1):
    def callback(env):
        if env.iteration % period == 0:
            print(f"[{env.iteration}] {env.evaluation_result_list}")

    return callback

def evaluation(label_list, y_pred, y_score):

    accuracy = accuracy_score(label_list, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(label_list, y_pred, average='macro')

    label_list = np.array(label_list)
    # 对于多分类问题的AUROC
    try:
        auc_roc_scores = [roc_auc_score(label_list[:, i], y_score[:, i]) for i in range(label_list.shape[1])]
        mean_auc_roc = np.mean(auc_roc_scores)
    except ValueError as e:
        print(f"无法计算AUROC: {e}")
        mean_auc_roc = None


    # 对于多分类问题的AUPRC
    try:
        auprc_scores = [average_precision_score(label_list[:, i], y_score[:, i]) for i in range(label_list.shape[1])]
        mean_auprc = np.mean(auprc_scores)
    except ValueError as e:
        print(f"无法计算AUPRC: {e}")
        mean_auprc = None

    return accuracy, precision, recall, f1, mean_auc_roc, mean_auprc

def print_metric(accuracy, precision, recall, f1, auroc, auprc):
    print(f"模型准确率: {accuracy * 100:.2f}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    if auroc is not None:
        print(f"AUROC: {auroc}")
    if auprc is not None:
        print(f"AUPRC: {auprc}")

class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # 打开文件时启用行缓冲

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 写入后立即刷新到文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def train_model(model, train_loader, val_loader, optimizer, device, epochs=10):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print(str(time_) + ": Epoch: ", epoch + 1, flush=True)
        # time.sleep(1)
        model.train()
        train_loss = 0.0
        for data in tqdm.tqdm(train_loader, desc="Training Process: "):
            # 将数据移动到GPU
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            # print("Input IDs shape:", input_ids.shape)  # 应输出类似 (batch_size, sequence_length)
            # print("Labels shape:", labels.shape)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            # train_loss += loss.item()

        # time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print(str(time_), f"Epoch {epoch + 1}", flush=True)
        # time.sleep(1)


# 需要改
# device = torch.device("cpu")
device = torch.device("cuda:1")
# device = torch.device("mps")

model_path =  "/home/rnd/wmj/berlin/ttc/tcr-bert-mlm-only"
output_path = "/home/rnd/wmj/berlin/ttc/log/training_multi_log_large_5.14.txt"
data_path = "/home/rnd/wmj/berlin/ttc/data_clean_large.csv"
sys.stdout = DualOutput(output_path)


df_alpha_beta = pd.read_csv(data_path, sep='\t')
# df_alpha_beta = df_alpha_beta.iloc[:300]
# df_alpha_beta = df_alpha_beta[df_alpha_beta['antigen.epitope'] == 'DATYQRTRALVR']

# df_alpha_beta = df_alpha_beta[df_alpha_beta['antigen.epitope'] == 'GILGFVFTL']
df_alpha_beta['cdr3_a_aa'] = df_alpha_beta['cdr3_a_aa'].apply(add_space)
df_alpha_beta['cdr3_b_aa'] = df_alpha_beta['cdr3_b_aa'].apply(add_space)

label_encoder = LabelEncoder()
df_alpha_beta['encoded_epitopes'] = label_encoder.fit_transform(df_alpha_beta['antigen.epitope'])
df_alpha_beta['mhc.a'] = label_encoder.fit_transform(df_alpha_beta['mhc.a'])
df_alpha_beta['mhc.b'] = label_encoder.fit_transform(df_alpha_beta['mhc.b'])

# df_alpha_beta = df_alpha_beta.iloc[:200]

# 使用 groupby 聚合相同的 cdr3_b_aa 和 cdr3_a_aa
aggregated_df = df_alpha_beta.groupby(['cdr3_b_aa', 'cdr3_a_aa']).agg({
    # 将所有不同的encoded_epitopes值汇总成一个列表
    'encoded_epitopes': lambda x: list(x.unique())
}).reset_index()

# 多标签格式转换
mlb = MultiLabelBinarizer()
encoded_epitopes = mlb.fit_transform(aggregated_df['encoded_epitopes'])

aggregated_df['encoded_epitopes'] = list(encoded_epitopes)

print(f"Loading model from {model_path}")

model_a = BertForMaskedLM.from_pretrained(model_path)

model_b = BertForMaskedLM.from_pretrained(model_path)

tokenizer = BertTokenizer.from_pretrained(model_path)

tokenizer.add_special_tokens({'mask_token': '[MASK]'})

# 扩展模型的嵌入层以包括新的标记
model_a.resize_token_embeddings(len(tokenizer))
model_b.resize_token_embeddings(len(tokenizer))

optimizer_a = Adam(model_a.parameters(), lr=1e-5)
optimizer_b = Adam(model_b.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()


# train_df, test_df = train_test_split(df_alpha_beta, test_size=0.2, random_state=42)

train_df_muiti, test_df_muti = train_test_split(aggregated_df, test_size=0.2, random_state=42)

# 实例化数据集和数据加载器
train_dataset_multi = TCRDataset(train_df_muiti, tokenizer, 64)
train_dataset_a = TCRDataset_a(df_alpha_beta, tokenizer, 64)
train_dataset_b = TCRDataset_b(df_alpha_beta, tokenizer, 64)
# val_dataset = TCRDataset(validate_df, tokenizer, 64)
# test_dataset = TCRDataset(test_df, tokenizer, 64)
test_dataset_multi = TCRDataset(test_df_muti, tokenizer, 64)


# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
# # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建DataCollator实例
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 创建DataLoader，应用data_collator
train_loader_a = DataLoader(train_dataset_a, batch_size=32, shuffle=True, collate_fn=data_collator)
train_loader_b = DataLoader(train_dataset_b, batch_size=32, shuffle=True, collate_fn=data_collator)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_loader_multi = DataLoader(train_dataset_multi, batch_size=32, shuffle=False)
test_loader_multi = DataLoader(test_dataset_multi, batch_size=32, shuffle=False)

epochs = 10
no_improve_epochs = 0

best_accuracy = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_auroc = 0
best_auprc = 0

for epoch in range(0, epochs):

    time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(str(time_) + ": Epoch: ", epoch + 1, flush=True)
    time.sleep(1)

    # 训练模型
    # train_model(model_a, train_loader_a, test_loader, optimizer_a, device, epochs=1)
    # train_model(model_b, train_loader_b, None, optimizer_b, device, epochs=1)

    # 训练模型
    output_a, output_b, label_list = get_encode_data(model_a, model_b, train_loader_multi, device, pooling_strategy="cls")

    merged_list = [a + b for a, b in zip(output_a, output_b)]

    # 训练XGBoost模型
    clf = OneVsRestClassifier(xgb.XGBClassifier(eval_metric='mlogloss', tree_method = "hist", device = "cuda"))

    # 训练模型
    clf.fit(merged_list, label_list)

    # 获得编码数据
    output_a, output_b, label_list = get_encode_data(model_a, model_b, test_loader_multi, device, pooling_strategy="cls")

    merged_list = [a + b for a, b in zip(output_a, output_b)]

    # 预测测试集
    y_pred = clf.predict(merged_list)

    y_score = clf.predict_proba(merged_list)

    accuracy, precision, recall, f1, auroc, auprc = evaluation(label_list, y_pred, y_score)

    print_metric(accuracy, precision, recall, f1, auroc, auprc)

    if f1 > best_f1:
        best_accuracy = accuracy
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        if not auroc is None:
            best_auroc = auroc
        if not auprc is None:
            best_auprc = auprc
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= 3:
        print("Early stopping triggered.")
        break

print(
    "Best Results - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}".format(
        best_accuracy, best_precision, best_recall, best_f1, best_auroc, best_auprc))









