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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import tcr_bert_TwoPartBertClassifier_2
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


time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(str(time_) + ": Loading data")
df_alpha_beta = pd.read_csv("./data_clean_large.csv", sep='\t')
# encode label
label_encoder = LabelEncoder()
df_alpha_beta['encoded_epitopes'] = label_encoder.fit_transform(df_alpha_beta['antigen.epitope'])
df_alpha_beta['cdr3_a_aa'] = df_alpha_beta['cdr3_a_aa'].apply(add_space)
df_alpha_beta['cdr3_b_aa'] = df_alpha_beta['cdr3_b_aa'].apply(add_space)

def train_model(model, train_loader, device):

    output_list = []
    label_list = []


    for data in tqdm.tqdm(train_loader, desc="Training Process: "):
        input_ids_a = data['input_ids_a'].to(device)
        attention_mask_a = data['attention_mask_a'].to(device)
        input_ids_b = data['input_ids_b'].to(device)
        attention_mask_b = data['attention_mask_b'].to(device)
        labels = data['labels'].to(device)

        model.eval()
        outputs = model(input_ids_a, input_ids_b)  # Assuming your model can handle these inputs
        output_list += outputs.tolist()
        label_list += labels.tolist()

    return output_list, label_list


device = torch.device('mps')

tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
model = tcr_bert_TwoPartBertClassifier_2.TwoPartBertClassifier(pretrained='wukevin/tcr-bert-mlm-only', n_output=30, freeze_encoder=False, separate_encoders=True).to(device)

# 随机打乱数据
df_alpha_beta = df_alpha_beta.sample(frac=1).reset_index(drop=True)

# 计算切分点
train_size = int(len(df_alpha_beta) * 0.8)
validate_size = int(len(df_alpha_beta) * 0.2)
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
output_list, label_list = train_model(model, train_loader, device)

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
model.fit(output_list, label_list)

# 训练模型
output_list, label_list = train_model(model, test_loader, device)

# merged_list = [a + b for a, b in zip(output_a, output_b)]

# # 进行预测
y_pred = model.predict(output_list)
accuracy = accuracy_score(label_list, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")
