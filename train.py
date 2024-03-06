import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, BertModel, BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

from sklearn.preprocessing import LabelEncoder


# 假设你的数据集包含两列：'sequence'（输入序列）和'label'（标签）
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return idx, self.labels[idx], text

device = torch.device('mps')

# 加载数据集
df = pd.read_csv("/Users/berlin/Documents/UoB/DSMP/pycharm_project_DSMP/vdjdb-2023-06-01/vdjdb.txt", sep='	')
texts = df['cdr3'].values  # 根据实际情况修改列名
labels_ = df['antigen.epitope'].values  # 根据实际情况修改列名

# 转换每个序列，加入空格
spaced_texts = [' '.join(list(text)) for text in texts]

# 初始化标签编码器
label_encoder = LabelEncoder()

# 假设df['label']是包含字符串标签的列
# 使用LabelEncoder将它们转换为整数
encoded_labels = label_encoder.fit_transform(labels_)

# 分割数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(spaced_texts, encoded_labels, test_size=0.2)

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
model = BertForSequenceClassification.from_pretrained("wukevin/tcr-bert-mlm-only", num_labels=1169).to(device)

# 创建数据集
train_dataset = CustomDataset(train_texts, train_labels)
val_dataset = CustomDataset(val_texts, val_labels)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮次
    per_device_train_batch_size=16,  # 训练批次大小
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
)


class Trainer(object):

    def __init__(self, args, tokenizer, model, train_dataset, test_dataset):

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=16, shuffle=True,
                                       drop_last=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=16, shuffle=True)

    def training(self):
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

        for epoch_index in range(0, self.args.num_train_epochs):

            print("#--------------------第{}轮训练--------------------#".format(epoch_index + 1))

            pbar_train = tqdm(total=math.floor(len(self.train_loader)))

            total_loss = 0

            for data in self.train_loader:
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                ids, label, content_list = data

                tokens_tensor = self.tokenizer.batch_encode_plus(content_list,
                                                                 add_special_tokens=True,
                                                                 padding='longest',
                                                                 truncation=True,
                                                                 return_attention_mask=True,
                                                                 return_tensors='pt',
                                                                 max_length=512).to(device)

                targets_tensor = torch.LongTensor(label).to(device)


                # 输入模型
                output = self.model(**tokens_tensor, labels=targets_tensor)
                loss = output.loss
                total_loss += loss.item()

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()

                pbar_train.update(1)

            pbar_train.close()

            avg_loss = total_loss / len(self.train_loader)
            print('avg_loss: ' + str(avg_loss))

            lr_scheduler.step()

            print('#--------------------Evaluate Dev--------------------#')
            dev_loss, dev_ave_f1, dev_ave_pre, dev_ave_re, dev_acc, dev_kappa, dev_cla_f1, dev_cla_pre, dev_cla_re \
                = self.evaluate(epoch_index, 'Dev', full_type=True)
            print('#--------------------Evaluate Test--------------------#')
            test_loss, test_ave_f1, test_ave_pre, test_ave_re, test_acc, test_kappa, test_cla_f1, test_cla_pre, \
            test_cla_re = self.evaluate(epoch_index, 'Test', full_type=True)

            self.save_checkpoint(epoch_index, dev_ave_f1, test_ave_f1, dev_acc, test_acc, dev_ave_pre, test_ave_pre,
                                 dev_ave_re, test_ave_re, dev_kappa, test_kappa, dev_cla_f1, test_cla_f1,
                                 dev_cla_pre, test_cla_pre, dev_cla_re, test_cla_re, early_stop_current,
                                 best_dev_ave_F1,
                                 optimizer.state_dict(), lr_scheduler.state_dict())

            # early_stop
            if dev_ave_f1 > best_dev_ave_F1:
                early_stop_current = 0
                best_dev_ave_F1 = dev_ave_f1
                print("#---------------------------------Best Model---------------------------------#")
                print()
                print()

            else:
                early_stop_current += 1
                # 若连续不能提升模型效果，打断训练
                if early_stop_current >= self.early_stop:
                    print("#------Early stopping at epoch {}.--------#".format(epoch_index))
                    return None
        return None

# 初始化Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    test_dataset=val_dataset
)

# 训练模型
trainer.training()
