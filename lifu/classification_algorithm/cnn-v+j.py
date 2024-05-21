import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * ((X_train.shape[1] // 2) // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * ((X_train.shape[1] // 2) // 2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss, total_acc = 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc.item()

        # Average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)

        # Validation phase
        model.eval()
        with torch.no_grad():
            total_val_loss, total_val_acc = 0, 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_acc = accuracy(outputs, labels)
                total_val_loss += val_loss.item()
                total_val_acc += val_acc.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')


# Load and prepare data as previously described
data = pd.read_csv("/Users/lifushen/Desktop/1d_cnn/CDR3_encoded_complete.csv")
data['cdr3_a_aa_encoded'] = data['cdr3_a_aa_encoded'].apply(lambda x: np.array(list(map(int, x.split()))))
data['cdr3_b_aa_encoded'] = data['cdr3_b_aa_encoded'].apply(lambda x: np.array(list(map(int, x.split()))))
X_a = np.stack(data['cdr3_a_aa_encoded'].values)
X_b = np.stack(data['cdr3_b_aa_encoded'].values)
X_vj = data[['v_b_gene_encoded', 'j_b_gene_encoded', 'v_a_gene_encoded', 'j_a_gene_encoded']].values
X = np.hstack([X_a, X_b, X_vj])
y = data['antigen.epitope_encoded'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
num_classes = len(np.unique(y_train))
model = CNN1D(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1)