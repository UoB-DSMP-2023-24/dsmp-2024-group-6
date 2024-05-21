from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 加载数据
data_df = pd.read_csv('encoded_features.txt', delim_whitespace=True)
labels_df = pd.read_csv('labels_one_hot.csv')

# 将标签转换回类别形式，这对于决策树模型是必需的
labels = np.argmax(labels_df.values, axis=1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data_df, labels, test_size=0.2, random_state=42)

# 创建决策树模型
clf = DecisionTreeClassifier(random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')