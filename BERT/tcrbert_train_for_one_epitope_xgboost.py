import datetime
import time

import numpy as np

import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder

import tqdm

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tcr_bert_TwoPartBertClassifier_3

from sklearn.decomposition import PCA

# 需要改
# device = torch.device("cpu")
# device = torch.device("cuda:1")
device = torch.device("mps")

class TCRDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.cdr3_a_aa = df['cdr3_a_aa'].tolist()
        self.cdr3_b_aa = df['cdr3_b_aa'].tolist()
        # self.epitope = df['encoded_epitopes'].tolist()
        self.labels = df['binder'].tolist()
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


def get_encode_data(model, train_loader, device):
    model.to(device)

    output_list_a = []
    output_list_b = []
    label_list = []

    for data in tqdm.tqdm(train_loader, desc="Getting data Process: "):
        input_ids_a = data['input_ids_a'].to(device)
        attention_mask_a = data['attention_mask_a'].to(device)
        input_ids_b = data['input_ids_b'].to(device)
        attention_mask_b = data['attention_mask_b'].to(device)
        labels = data['labels'].to(device)

        model.eval()
        outputs_a, outputs_b = model(input_ids_a, input_ids_b,
                                     feature_extraction=True)  # Assuming your model can handle these inputs
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
        output_list_a += outputs_a.tolist()
        output_list_b += outputs_b.tolist()
        label_list += labels.tolist()

    return output_list_a, output_list_b, label_list


def print_evaluation(period=1):
    def callback(env):
        if env.iteration % period == 0:
            print(f"[{env.iteration}] {env.evaluation_result_list}")

    return callback


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, early_stopping_patience=3):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(str(time_) + ": Epoch: ", epoch + 1, flush=True)
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
        print(str(time_), f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}", flush=True)
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

        # train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        all_probabilities = np.vstack(all_probabilities)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

        # 确保所有标签都在预期的类别范围内
        valid_labels = set(range(2))  # 对于30个类别
        labels_set = set(all_labels)
        if not labels_set.issubset(valid_labels):
            print("发现无效标签:", labels_set - valid_labels)

        # 检查概率行和
        prob_sums = all_probabilities.sum(axis=1)
        if not np.allclose(prob_sums, 1):
            print("某些概率和不为1")

        # 对于多分类问题的AUROC
        try:
            auroc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except ValueError as e:
            print(f"无法计算AUROC: {e}")
            auroc = None

        # 对于多分类问题的AUPRC
        try:
            auprc = average_precision_score(all_labels, all_probabilities[:, 1])
        except ValueError as e:
            print(f"无法计算AUPRC: {e}")
            auprc = None

        time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(str(time_), f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Validation Loss: {val_loss}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        if auroc is not None:
            print(f"AUROC: {auroc}")

        if auprc is not None:
            print(f"AUPRC: {auprc}")

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


df_alpha_beta = pd.read_csv("./data_clean_large_neg.csv", sep='\t')
df_alpha_beta = df_alpha_beta[df_alpha_beta['antigen.epitope'] == 'ATDALMTGF']
# df_alpha_beta = df_alpha_beta[df_alpha_beta['antigen.epitope'] == 'GILGFVFTL']
# encode label
# label_encoder = LabelEncoder()
# df_alpha_beta['encoded_epitopes'] = label_encoder.fit_transform(df_alpha_beta['antigen.epitope'])
df_alpha_beta['cdr3_a_aa'] = df_alpha_beta['cdr3_a_aa'].apply(add_space)
df_alpha_beta['cdr3_b_aa'] = df_alpha_beta['cdr3_b_aa'].apply(add_space)

label_encoder = LabelEncoder()
df_alpha_beta['mhc.a'] = label_encoder.fit_transform(df_alpha_beta['mhc.a'])
df_alpha_beta['mhc.b'] = label_encoder.fit_transform(df_alpha_beta['mhc.b'])

tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
# model = BertForSequenceClassification.from_pretrained('wukevin/tcr-bert-mlm-only', num_labels=30).to(device)

# 初始化模型、优化器和损失函数
model = tcr_bert_TwoPartBertClassifier_3.TwoPartBertClassifier(pretrained='wukevin/tcr-bert-mlm-only',
                                                               n_output=2, freeze_encoder=False, separate_encoders=True,
                                                               seq_pooling = "mean")
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 假设 X 是你的64维特征矩阵，y 是对应的标签
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 定义一个简单的回调函数来打印评估指标


# 找到所有独特的表位编码
unique_epitopes = df_alpha_beta['antigen.epitope'].unique()

# 遍历每一个独特的表位编码
for epitope in unique_epitopes:

    # if epitope != "ATDALMTGF":
    #     continue

    epitope_name = "Epitope: " + str(epitope)
    print(f"Results for {epitope_name}")

    # 在处理过的数据集df中过滤出对应epitope_encoded的行
    df_filtered = df_alpha_beta[df_alpha_beta['antigen.epitope'] == epitope]

    train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)

    # 实例化数据集和数据加载器
    train_dataset = TCRDataset(train_df, tokenizer, 64)
    # val_dataset = TCRDataset(validate_df, tokenizer, 64)
    test_dataset = TCRDataset(test_df, tokenizer, 64)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练模型
    # train_model(model, train_loader, test_loader, optimizer, loss_fn, device, epochs=4, early_stopping_patience=10)

    print("get encode")

    # 训练模型
    output_a, output_b, label_list = get_encode_data(model, train_loader, device)

    pca = PCA(n_components=100)

    # reduced_output_a = []
    # for output in output_a:
    #     reduced_output_a.append(pca.fit_transform(output))

    output_a = pca.fit_transform(output_a)
    output_b = pca.fit_transform(output_b)


    merged_list = [a + b for a, b in zip(output_a, output_b)]

    # merged_list = []
    # # 遍历 output_a 和 output_b 中的每一行
    # for a_row, b_row, a, b in zip(output_a, output_b, train_df['mhc.a'], train_df['mhc.b']):
    #     # 合并当前行的数据
    #     merged_row = a_row + b_row + [a] + [b]
    #     # 将合并后的行添加到结果列表中
    #     merged_list.append(merged_row)

    # 训练XGBoost模型
    clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method="hist", device="cuda")

    # # 创建随机森林模型
    # clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 训练模型
    clf.fit(merged_list, label_list)

    # 训练模型
    output_a, output_b, label_list = get_encode_data(model, test_loader, device)

    output_a = pca.fit_transform(output_a)
    output_b = pca.fit_transform(output_b)

    merged_list = [a + b for a, b in zip(output_a, output_b)]

    # merged_list = []
    # # 遍历 output_a 和 output_b 中的每一行
    # for a_row, b_row, a, b in zip(output_a, output_b, test_df['mhc.a'], test_df['mhc.b']):
    #     # 合并当前行的数据
    #     merged_row = a_row + b_row + [a] + [b]
    #     # 将合并后的行添加到结果列表中
    #     merged_list.append(merged_row)

    # 预测测试集
    y_pred = clf.predict(merged_list)

    # print("真实: ", y_pred)
    # print("预测: ", label_list)

    # # 进行预测
    # y_pred = xxgb.predict(merged_list)
    accuracy = accuracy_score(label_list, y_pred)
    print(f"模型准确率: {accuracy * 100:.2f}%")

    # output_a = output_a.to_list()
    # output_b = output_b.to_list()
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, y_pred, average='macro')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # # 确保所有标签都在预期的类别范围内
    # valid_labels = set(range(30))  # 对于30个类别
    # labels_set = set(label_list)
    # if not labels_set.issubset(valid_labels):
    #     print("发现无效标签:", labels_set - valid_labels)

    # # 检查概率行和
    # prob_sums = all_probabilities.sum(axis=1)
    # if not np.allclose(prob_sums, 1):
    #     print("某些概率和不为1")

    y_score = clf.predict_proba(merged_list)

    # 对于多分类问题的AUROC
    try:
        auroc = roc_auc_score(label_list, y_score[:, 1])
    except ValueError as e:
        print(f"无法计算AUROC: {e}")
        auroc = None

    if auroc is not None:
        print(f"AUROC: {auroc}")

    # 对于多分类问题的AUPRC
    try:
        auprc = average_precision_score(label_list, y_score[:, 1])
    except ValueError as e:
        print(f"无法计算AUPRC: {e}")
        auprc = None

    if auprc is not None:
        print(f"AUPRC: {auprc}")




