import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
data_df = pd.read_csv('encoded_features.txt', delim_whitespace=True)
labels_df = pd.read_csv('labels_one_hot.csv')

data = data_df.values
labels = labels_df.values

data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(data_tensor, labels_tensor, test_size=0.2, random_state=42)

# 创建数据集
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 从训练数据集中随机选择10%
indices = list(range(len(train_dataset)))
random.shuffle(indices)
subset_indices = indices[:len(indices) // 10]
train_subset = Subset(train_dataset, subset_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

conv_output_size = 18688

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=2)
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(conv_output_size, 1000)  # 使用conv_output_size
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN1D(num_classes=labels.shape[1])

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features = features.unsqueeze(1)  # 增加一个通道维度
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}')

def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.unsqueeze(1)
            outputs = model(features)
            predicted = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total}%')

train_model(25)
evaluate_model()
