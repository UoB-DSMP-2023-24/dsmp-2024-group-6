import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据和标签
data_df = pd.read_csv('encoded_cdr3_seqs_large.txt', delim_whitespace=True)
labels_df = pd.read_csv('labels_one_hot_large.csv')

# 转换为NumPy数组
data_np = data_df.values
labels_np = labels_df.values

# 数据标准化 - 添加一个小的epsilon以避免除以零
epsilon = 1e-10
mean = data_np.mean(axis=0)
std = data_np.std(axis=0) + epsilon  # 防止除以零
data_np = (data_np - mean) / std

# 转换为torch张量
data_tensor = torch.tensor(data_np, dtype=torch.float32)
labels_tensor = torch.tensor(labels_np, dtype=torch.float32)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_tensor, labels_tensor, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义CNN模型
class CNN1D(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * (num_features // 4), 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Conv1d需要额外的维度
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
model = CNN1D(num_features=data_df.shape[1], num_classes=labels_df.shape[1])

# 设置损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 定义训练函数
def train_model(num_epochs):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                print(f"Loss is nan on iteration {i}")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_losses

def evaluate_model_and_generate_labels():
    model.eval()
    all_labels = []
    all_probabilities = []
    all_predictions = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probabilities = np.concatenate(all_probabilities)
    all_predictions = np.concatenate(all_predictions)

    # 计算性能指标
    accuracy = accuracy_score(np.argmax(all_labels, axis=1), all_predictions)
    f1 = f1_score(np.argmax(all_labels, axis=1), all_predictions, average='macro')
    auroc = roc_auc_score(all_labels, all_probabilities, average='macro', multi_class='ovr')
    recall = recall_score(np.argmax(all_labels, axis=1), all_predictions, average='macro')

    # 生成混淆矩阵
    cm = confusion_matrix(np.argmax(all_labels, axis=1), all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Recall: {recall:.4f}")

losses = train_model(10)
evaluate_model_and_generate_labels()

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Train Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()