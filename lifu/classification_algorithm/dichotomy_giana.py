import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# 加载数据集
data = pd.read_csv('/Users/lifushen/Desktop/1d_cnn/output_encoded_2.csv')

# 将编码列中的字符串转换为实际的数值数组
data['encoded_CDR3a'] = data['encoded_CDR3a'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
data['encoded_CDR3b'] = data['encoded_CDR3b'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# 特征和标签
X = np.column_stack((np.vstack(data['encoded_CDR3a'].values),
                     np.vstack(data['encoded_CDR3b'].values),
                     data['peptide'].values,
                     data['MHC'].values))
y = data['binder'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建XGBoost分类器实例
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率

# 计算各种性能指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)  # 计算AUROC

# 打印性能指标
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision: %.2f%%" % (precision * 100.0))
print("Recall: %.2f%%" % (recall * 100.0))
print("F1 Score: %.2f" % f1)
print("Confusion Matrix:")
print(cm)
print("AUROC: %.2f" % auc_roc)