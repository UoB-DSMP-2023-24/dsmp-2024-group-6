import datetime
import time

import numpy as np
import torch
import pandas as pd
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import tcr_bert_TwoPartBertClassifier_4
import tqdm

# 需要改
# device = torch.device("cpu")
device = torch.device("cuda:0")
# device = torch.device("mps")

class TCRDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.cdr3_a_aa = df['cdr3_a_aa'].tolist()
        self.cdr3_b_aa = df['cdr3_b_aa'].tolist()
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

time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(str(time_) + ": Loading data")


df_alpha_beta = pd.read_csv("/home/rnd/wmj/berlin/ttc/data_clean_large.csv", index_col=0, sep='\t')

# # 统计peptide字段中每个序列的出现次数
# peptide_counts = df_alpha_beta['peptide'].value_counts()

# # 找到出现次数最多的序列和对应的出现次数
# most_common_peptide = peptide_counts.idxmax()
# count_most_common = peptide_counts.max()

# df = df_alpha_beta[df_alpha_beta['peptide'] == 'GILGFVFTL']
# df = df_alpha_beta[df_alpha_beta['antigen.epitope'] == 'GILGFVFTL']
# encode label
label_encoder = LabelEncoder()
df_alpha_beta['encoded_epitopes'] = label_encoder.fit_transform(df_alpha_beta['antigen.epitope'])

aggregated_df = df_alpha_beta.groupby(['cdr3_b_aa', 'cdr3_a_aa']).agg({
    # 将所有不同的encoded_epitopes值汇总成一个列表
    'encoded_epitopes': lambda x: list(x.unique())
}).reset_index()

# 多标签格式转换
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(aggregated_df['encoded_epitopes'])

rows_as_lists = [row.tolist() for row in labels]

aggregated_df['encoded_epitopes'] = rows_as_lists


aggregated_df['cdr3_b_aa'] = aggregated_df['cdr3_b_aa'].apply(add_space)
aggregated_df['cdr3_a_aa'] = aggregated_df['cdr3_a_aa'].apply(add_space)

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, early_stopping_patience=3):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(str(time_) + ": Epoch: ", epoch+1, flush=True)
        time.sleep(1)
        model.train()
        train_loss = 0.0
        for data in tqdm.tqdm(train_loader, desc="Training Process: "):
            input_ids_a = data['input_ids_a'].to(device)
            attention_mask_a = data['attention_mask_a'].to(device)
            input_ids_b = data['input_ids_b'].to(device)
            attention_mask_b = data['attention_mask_b'].to(device)
            labels = data['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids_a, input_ids_b)  # Assuming your model can handle these inputs
            loss = loss_fn(outputs, labels.float())  # Adjust this if your model's output structure differs
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step omitted for brevity; include it following the same pattern as training

        time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(str(time_), f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}", flush=True)
        time.sleep(1)

        # Early stopping logic here
        val_loss = 0.0
        model.eval()
        all_preds = []
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for data in tqdm.tqdm(val_loader, desc="Evaluation: "):
                input_ids_a = data['input_ids_a'].to(device)
                attention_mask_a = data['attention_mask_a'].to(device)
                input_ids_b = data['input_ids_b'].to(device)
                attention_mask_b = data['attention_mask_b'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids_a, input_ids_b)
                loss = loss_fn(outputs, labels.float())
                val_loss += loss.item()

                # preds = torch.argmax(outputs, dim=1)
                preds = (outputs >= 0.4).int()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(np.squeeze(outputs.cpu().numpy()))

        # train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        all_probabilities = np.vstack(all_probabilities)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

        # # 确保所有标签都在预期的类别范围内
        # valid_labels = set(range(2))  # 对于30个类别
        # labels_set = set(all_labels)
        # if not labels_set.issubset(valid_labels):
        #     print("发现无效标签:", labels_set - valid_labels)

        # # 检查概率行和
        # prob_sums = all_probabilities.sum(axis=1)
        # if not np.allclose(prob_sums, 1):
        #     print("某些概率和不为1")


        label_list = np.array(all_labels)
        # 对于多分类问题的AUROC
        try:
            auc_roc_scores = [roc_auc_score(label_list[:, i], all_probabilities[:, i]) for i in range(label_list.shape[1])]
            mean_auc_roc = np.mean(auc_roc_scores)
        except ValueError as e:
            print(f"无法计算AUROC: {e}")
            mean_auc_roc = None


        # 对于多分类问题的AUPRC
        try:
            auprc_scores = [average_precision_score(label_list[:, i], all_probabilities[:, i]) for i in range(label_list.shape[1])]
            mean_auprc = np.mean(auprc_scores)
        except ValueError as e:
            print(f"无法计算AUPRC: {e}")
            mean_auprc = None

        time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(str(time_), f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Validation Loss: {val_loss}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        if mean_auc_roc is not None:
            print(f"AUROC: {mean_auc_roc}")

        if mean_auprc is not None:
            print(f"AUPRC: {mean_auprc}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model_small.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break


# device = torch.device('mps')
# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("/home/rnd/wmj/berlin/ttc/tcr-bert-mlm-only")
# model = BertForSequenceClassification.from_pretrained('wukevin/tcr-bert-mlm-only', num_labels=30).to(device)

# 初始化模型、优化器和损失函数
model = tcr_bert_TwoPartBertClassifier_4.TwoPartBertClassifier(pretrained='/home/rnd/wmj/berlin/ttc/tcr-bert-mlm-only', n_output=30, freeze_encoder=False, separate_encoders=True, seq_pooling="mean")
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCELoss()

# 随机打乱数据
aggregated_df = aggregated_df.sample(frac=1).reset_index(drop=True)

# aggregated_df = aggregated_df.iloc[:1000]
train_df, remaining_data = train_test_split(aggregated_df, test_size=0.2)
# validate_df, test_df = train_test_split(remaining_data, test_size=0.5)

# 实例化数据集和数据加载器
train_dataset = TCRDataset(train_df, tokenizer, 64)
val_dataset = TCRDataset(remaining_data, tokenizer, 64)
# test_dataset = TCRDataset(test_df, tokenizer, 64)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=100, early_stopping_patience=10)
