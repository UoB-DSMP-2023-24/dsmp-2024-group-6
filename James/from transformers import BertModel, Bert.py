import pandas as pd
import torch
import re
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
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

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'protein_sequence': text,
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
df = pd.read_csv("C:/Users/james/OneDrive/Documents/Bristol/Mini Project/vdjdb-2023-06-01/vdjdb.txt", sep='	')  # Update this path
df['cdr3'] = df['cdr3'].fillna('').apply(lambda x: re.sub(r"[UZOB]", "X", x))
texts = df['cdr3'].values 
# labels_ = df['antigen.epitope'].values  
labels_ = df['antigen.species'].values  

# add a space
spaced_texts = [' '.join(list(text)) for text in texts]

# encode label
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels_)

# Split dataset
train_texts, temp_texts, train_labels, temp_labels = train_test_split(spaced_texts, encoded_labels, test_size=0.8)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5)

device = torch.device('cpu')
# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
model = BertForSequenceClassification.from_pretrained('Rostlab/prot_bert', num_labels=len(set(encoded_labels))).to(device)

# Create datasets
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=64)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=64)
test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length=64)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.0001,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
trainer.evaluate()
