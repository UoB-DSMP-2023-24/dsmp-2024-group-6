import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# 加载数据集
data = pd.read_csv('output_encoded_4.csv')

# 将编码列中的字符串转换为实际的数值数组
data['cdr3_a_aa'] = data['cdr3_a_aa'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
data['cdr3_b_aa'] = data['cdr3_b_aa'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# 获取独特的antigen.epitope整数编码
unique_epitopes = data['antigen.epitope'].unique()

# 对每个独特的antigen.epitope编码训练和测试模型
results = []
for epitope in unique_epitopes:
    # 选取当前epitope的数据
    epitope_data = data[data['antigen.epitope'] == epitope]

    # 特征和标签
    X = np.column_stack((
                         np.vstack(epitope_data['cdr3_a_aa'].values),
                         np.vstack(epitope_data['cdr3_b_aa'].values),
                         #epitope_data['cluster.id'].values
                         ))
    y = epitope_data['binder'].values

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 检查y_test中的类别数量
    if len(np.unique(y_test)) < 2:
        print(f"Skipped ROC AUC calculation for epitope {epitope} due to insufficient classes in y_test.")
        continue  # 跳过当前循环迭代

    # 创建XGBoost分类器实例
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率

    # 计算各种性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else None
    auprc = average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else None

    # 存储结果
    results.append({
        'antigen_epitope': epitope,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auprc': auprc
    })
# Mapping from integers to epitope strings
epitope_mapping = {
    2: "IVTDFSVIK", 3: "KLGGALQAK", 4: "GILGFVFTL", 5: "ELAGIGILTV", 6: "HPVTKYIM",
    7: "RAQAPPPSW", 8: "RLRAEAQVK", 9: "AVFDRKSDAK", 10: "RAKFKQLL", 11: "TPRVTGGGAM",
    12: "YLQPRTFLL", 13: "PKYVKQNTLKLAT", 14: "TFEYVSQPFLMDLE", 15: "NLVPMVATV", 16: "GLCTLVAML",
    17: "SPRWYFYYL", 18: "DATYQRTRALVR", 19: "ATDALMTGF", 20: "TTDPSFLGRY", 21: "CINGVCWTV",
    22: "MEVTPSGTWL", 23: "KSKRTPMGF", 24: "QYIKWPWYI", 25: "LLWNGPMAV", 26: "LTDEMIAQY",
    27: "NQKLIANQF", 28: "RPRGEVRFL", 29: "NYNYLYRLF", 30: "KTFPPTEPK", 31: "GPRLGVRAT"
}

# Modify the loop that prints performance metrics
for result in results:
    epitope_name = epitope_mapping[int(result['antigen_epitope'])]
    print(f"Antigen Epitope ID: {result['antigen_epitope']}")
    print(f"Antigen Epitope: {epitope_name}")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print(f"Precision: {result['precision']:.2f}")
    print(f"Recall: {result['recall']:.2f}")
    print(f"F1 Score: {result['f1_score']:.2f}")
    print(f"AUROC: {result['auc_roc']:.2f}")
    print(f"AUPRC: {result['auprc']:.2f}")
    print("-" * 40)

