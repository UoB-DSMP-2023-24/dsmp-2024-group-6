from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 加载重新编码后的数据
encoded_df = pd.read_csv('/home/rnd/wmj/berlin/ttc/CDR3_encoded2.csv')

# cdr3序列的one-hot编码转换回numpy数组
cdr3_encoded = np.array([np.fromstring(seq, dtype=int, sep=' ') for seq in encoded_df['cdr3encode']])

# epitope标签已经是整数编码，直接使用
epitope_encoded = encoded_df['epitopeencode'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(cdr3_encoded, epitope_encoded, test_size=0.2, random_state=42)

# 创建随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')