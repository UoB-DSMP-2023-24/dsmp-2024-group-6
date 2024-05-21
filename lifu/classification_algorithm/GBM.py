import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# 加载数据
encoded_df = pd.read_csv(' $giana_final.csv')

# 将 antigen.epitope 字段的字符串分割为列表
encoded_df['antigen.epitope'] = encoded_df['antigen.epitope'].astype(str).apply(lambda x: x.split(','))

# 聚合相同 cdr3 序列的所有 epitope 以及保持基因编码
aggregated_data = encoded_df.groupby(['cdr3_a_aa', 'cdr3_b_aa']).agg({
    'antigen.epitope': lambda x: list(set([item for sublist in x for item in sublist]))
}).reset_index()

# 更新后的清洗字符串并转换为浮点数数组的函数
def clean_and_convert(seq):
    # 使用正则表达式找出所有可能的浮点数或科学记数法表示的数值
    pattern = r'-?\d+\.\d+(?:[eE][-+]?\d+)?'
    numbers = re.findall(pattern, seq)
    float_nums = [float(num) for num in numbers]
    return np.array(float_nums, dtype=float)

# cdr3_a_aa 和 cdr3_b_aa 的处理转换回 numpy 数组
cdr3_a_encoded = np.array([clean_and_convert(seq) for seq in aggregated_data['cdr3_a_aa']])
cdr3_b_encoded = np.array([clean_and_convert(seq) for seq in aggregated_data['cdr3_b_aa']])

# 多标签格式转换
mlb = MultiLabelBinarizer()
epitope_encoded = mlb.fit_transform(aggregated_data['antigen.epitope'])

# 将所有特征合并为一个特征矩阵
X = np.hstack((cdr3_a_encoded, cdr3_b_encoded))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, epitope_encoded, test_size=0.2, random_state=42)

# 创建一个多标签分类器
xgb = OneVsRestClassifier(XGBClassifier(random_state=42))

# 训练模型
xgb.fit(X_train, y_train)

# 预测测试集
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)

# 计算整体性能指标
accuracy = accuracy_score(y_test, y_pred)
try:
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)

    # 使用try-except来处理仅有一个类别的情况
    try:
        roc_auc = roc_auc_score(y_test, y_proba, average='macro', multi_class='ovo')
    except ValueError as e:
        print("ROC AUC calculation error:", e)
        roc_auc = None

    # 定义zero_division参数
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    # 打印整体结果
    print("---------------------------------------------------------")
    print("Overall Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    if roc_auc:
        print(f"AUROC: {roc_auc:.2f}")
    else:
        print("AUROC: Not calculable")
    print("---------------------------------------------------------")
except Exception as e:
    print("An error occurred during model training or evaluation:", e)