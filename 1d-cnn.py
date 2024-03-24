import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 数据加载和预处理
# 加载标签
with open('/Users/lifushen/Desktop/1d_cnn/antigen.epitope_output.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
labels.pop()

# 创建从标签到整数的映射
label_to_int = {label: i for i, label in enumerate(sorted(set(labels)))}

# 将标签转换为整数
labels_int = [label_to_int[label] for label in labels]

# 保存转换后的整数标签到文件
with open('labels_int.txt', 'w') as f:
    for label in labels_int:
        f.write(f"{label}\n")

def load_data(encoded_sequences_file, labels_file):
    # 使用Pandas读取数据，假设第一行是列名
    data_df = pd.read_csv(encoded_sequences_file, delim_whitespace=True, skiprows=1)
    # 转换为numpy数组
    data = data_df.values
    # 加载标签
    labels = np.loadtxt(labels_file)
    return data, labels

# 将numpy数组转换为torch张量
def numpy_to_tensor(data, labels):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor

encoded_sequences_file = '/Users/lifushen/Desktop/1d_cnn/encoded_features.txt'
labels_file = '/Users/lifushen/Desktop/1d_cnn/labels_int.txt'
data, labels = load_data(encoded_sequences_file, labels_file)
data_tensor, labels_tensor = numpy_to_tensor(data, labels)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_tensor, labels_tensor, test_size=0.2, random_state=42)

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 构建1D-CNN模型
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * ((data.shape[1] - 2) // 2), 50)  # 根据数据维度调整
        self.fc2 = nn.Linear(50, len(set(labels_int)))  # len(set(labels_int))是类别的总数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = CNN1D()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(train_loader):
    model.train()
    for data, labels in train_loader:
        data = data.unsqueeze(1)  # 增加一个通道维度
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# 评估模型
def evaluate_model(test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.unsqueeze(1)  # 增加一个通道维度
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')


train_model(train_loader)
evaluate_model(test_loader)