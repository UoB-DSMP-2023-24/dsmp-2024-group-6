import datetime
import time

import numpy as np
import torch
import pandas as pd
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import tcr_bert_TwoPartBertClassifier
import tqdm

# 需要改
# device = torch.device("cpu")
# device = torch.device("cuda")
device = torch.device("mps")

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

def summarize_dataframe(df):
    summary = pd.DataFrame({
        'Variable Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    return summary

def process_data(df):
    trb_data = df[(df['gene'] == 'TRB') & (~df['v.segm'].isnull()) & (~df['j.segm'].isnull()) & (df['complex.id'] != 0)]

    df_alpha_beta = pd.DataFrame(
        columns=['id', 'cdr3_b_aa', 'v_b_gene', 'j_b_gene', 'cdr3_a_aa', 'v_a_gene', 'j_a_gene'] +
                ['species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope', 'antigen.gene',
                 'antigen.species', 'reference.id', 'method', 'meta', 'cdr3fix', 'vdjdb.score',
                 'web.method', 'web.method.seq', 'web.cdr3fix.nc', 'web.cdr3fix.unmp',
                 'is_cdr3_alpha_valid', 'is_mhc_a_valid'])

    for index, row in trb_data.iterrows():
        # 获取当前行的 complex.id
        complex_id = row['complex.id']

        # 在原始 DataFrame 中查找相同 complex.id 的 TRA 数据
        tra_data = df[(df['complex.id'] == complex_id) & (df['gene'] == 'TRA') & (~df['v.segm'].isnull()) &
                      (~df['j.segm'].isnull()) & (df['species'] == "HomoSapiens")]

        # 如果找到了相同 complex.id 的 TRA 数据
        if not tra_data.empty:
            # 提取 cdr3、v.segm 和 j.segm 列的值
            cdr3_a_aa = tra_data.iloc[0]['cdr3']
            v_a_gene = tra_data.iloc[0]['v.segm']
            j_a_gene = tra_data.iloc[0]['j.segm']

            # 将提取的值填充至新的 DataFrame 中
            new_row = {'cdr3_b_aa': row['cdr3'], 'v_b_gene': row['v.segm'], 'j_b_gene': row['j.segm'],
                       'id': complex_id,
                       'cdr3_a_aa': cdr3_a_aa, 'v_a_gene': v_a_gene, 'j_a_gene': j_a_gene}
            # 将原始数据中的其他字段也添加到行数据中
            for field in ['species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope', 'antigen.gene',
                          'antigen.species',
                          'reference.id', 'method', 'meta', 'cdr3fix', 'vdjdb.score', 'web.method',
                          'web.method.seq',
                          'web.cdr3fix.nc', 'web.cdr3fix.unmp', 'is_cdr3_alpha_valid', 'is_mhc_a_valid']:
                new_row[field] = row[field]

            # new_row = pd.Series(new_row)

            df_alpha_beta = pd.concat([df_alpha_beta, pd.DataFrame.from_records([new_row])])

    return df_alpha_beta



df_alpha_beta = pd.read_csv("./data_clean_large.csv", sep='\t')
# encode label
label_encoder = LabelEncoder()
df_alpha_beta['encoded_epitopes'] = label_encoder.fit_transform(df_alpha_beta['antigen.epitope'])
df_alpha_beta['v_b_gene'] = label_encoder.fit_transform(df_alpha_beta['v_b_gene'])
df_alpha_beta['j_b_gene'] = label_encoder.fit_transform(df_alpha_beta['j_b_gene'])
df_alpha_beta['v_a_gene'] = label_encoder.fit_transform(df_alpha_beta['v_a_gene'])
df_alpha_beta['j_a_gene'] = label_encoder.fit_transform(df_alpha_beta['j_a_gene'])
df_alpha_beta['cdr3_a_aa'] = df_alpha_beta['cdr3_a_aa'].apply(add_space)
df_alpha_beta['cdr3_b_aa'] = df_alpha_beta['cdr3_b_aa'].apply(add_space)


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
            loss = loss_fn(outputs, labels)  # Adjust this if your model's output structure differs
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
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(probabilities.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        all_probabilities = np.vstack(all_probabilities)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

        # 确保所有标签都在预期的类别范围内
        valid_labels = set(range(30))  # 对于30个类别
        labels_set = set(all_labels)
        if not labels_set.issubset(valid_labels):
            print("发现无效标签:", labels_set - valid_labels)

        # 检查概率行和
        prob_sums = all_probabilities.sum(axis=1)
        if not np.allclose(prob_sums, 1):
            print("某些概率和不为1")


        # 对于多分类问题的AUROC
        try:
            auroc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"无法计算AUROC: {e}")
            auroc = None

        time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(str(time_), f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Validation Loss: {val_loss}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        if auroc is not None:
            print(f"AUROC: {auroc}")

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
# model = BertForSequenceClassification.from_pretrained('wukevin/tcr-bert-mlm-only', num_labels=30).to(device)

# 初始化模型、优化器和损失函数
model = BertModel.from_pretrained("wukevin/tcr-bert-mlm-only", add_pooling_layer='cls')
# optimizer = Adam(model.parameters(), lr=1e-5)
# loss_fn = torch.nn.CrossEntropyLoss()

# 随机打乱数据
df_alpha_beta = df_alpha_beta.sample(frac=1).reset_index(drop=True)

# 计算切分点
train_size = int(len(df_alpha_beta) * 0.8)
test_size = int(len(df_alpha_beta) * 0.2)
# 测试集大小可以直接通过剩余的部分来确定

# 切分数据集
train_df = df_alpha_beta.iloc[:train_size]
# validate_df = df_alpha_beta.iloc[train_size:train_size+validate_size]
test_df = df_alpha_beta.iloc[train_size:]

train_df = train_df[:1000]

# 实例化数据集和数据加载器
train_dataset = TCRDataset(train_df, tokenizer, 64)
# val_dataset = TCRDataset(validate_df, tokenizer, 64)
test_dataset = TCRDataset(test_df, tokenizer, 64)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
output_a, output_b, label_list = get_encode_data(model, train_loader, device)

# merged_list = [a + b + c + d + e + f for a, b, c, d, e, f in zip(output_a, output_b,
#                                                                  train_df['v_b_gene'],
#                                                                  train_df['j_b_gene'],
#                                                                  train_df['v_a_gene'],
#                                                                  train_df['j_a_gene'])]

merged_list = []
# 遍历 output_a 和 output_b 中的每一行
for a_row, b_row, v_b_gene, j_b_gene, v_a_gene, j_a_gene in zip(output_a, output_b, df_alpha_beta['v_b_gene'], df_alpha_beta['j_b_gene'], df_alpha_beta['v_a_gene'], df_alpha_beta['j_a_gene']):
    # 合并当前行的数据
    merged_row = a_row + b_row + [v_b_gene] + [j_b_gene] + [v_a_gene] + [j_a_gene]
    # 将合并后的行添加到结果列表中
    merged_list.append(merged_row)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设 X 是你的64维特征矩阵，y 是对应的标签
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 定义一个简单的回调函数来打印评估指标
def print_evaluation(period=1):
    def callback(env):
        if env.iteration % period == 0:
            print(f"[{env.iteration}] {env.evaluation_result_list}")
    return callback

# 训练XGBoost模型
model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='logloss', num_class=30, verbosity=2)
model.fit(merged_list, label_list)

# 训练模型
output_a, output_b, label_list = get_encode_data(model, test_loader, device)

merged_list = [a + b for a, b in zip(output_a, output_b)]

# # 进行预测
y_pred = model.predict(merged_list)
accuracy = accuracy_score(label_list, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")

# output_a = output_a.to_list()
# output_b = output_b.to_list()





