import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
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

# Load and prepare dataset
df = pd.read_csv("/Users/berlin/Documents/UoB/DSMP/pycharm_project_DSMP/vdjdb-2023-06-01/vdjdb.txt", sep='	')  # Update this path
texts = df['cdr3'].values  # 根据实际情况修改列名
# labels_ = df['antigen.epitope'].values  # 根据实际情况修改列名
labels_ = df['antigen.species'].values  # 根据实际情况修改列名

# add a space
spaced_texts = [' '.join(list(text)) for text in texts]

# encode label
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels_)

# Split dataset
train_texts, temp_texts, train_labels, temp_labels = train_test_split(spaced_texts, encoded_labels, test_size=0.8)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5)

device = torch.device('mps')
# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
model = BertModel.from_pretrained('wukevin/tcr-bert-mlm-only').to(device)

# Create datasets
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=64)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=64)
test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length=64)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     evaluation_strategy="epoch"
# )
#
# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics
# )

# # Train and evaluate
# trainer.train()
# trainer.evaluate()


