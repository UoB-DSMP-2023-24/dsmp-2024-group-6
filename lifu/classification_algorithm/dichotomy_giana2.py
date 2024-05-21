import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, precision_recall_fscore_support

# Load the dataset
data = pd.read_csv('output_encoded_3.csv')

# Convert string-encoded arrays into actual numerical arrays
data['cdr3_a_aa'] = data['cdr3_a_aa'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
data['cdr3_b_aa'] = data['cdr3_b_aa'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Obtain unique integer codes for antigen.epitope
unique_epitopes = data['antigen.epitope'].unique()


def evaluation(label_list, y_pred, y_score):
    accuracy = accuracy_score(label_list, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, y_pred, average='macro')

    try:
        auroc = roc_auc_score(label_list, y_score[:, 1])
    except ValueError as e:
        print(f"Cannot calculate AUROC: {e}")
        auroc = None

    try:
        auprc = average_precision_score(label_list, y_score[:, 1])
    except ValueError as e:
        print(f"Cannot calculate AUPRC: {e}")
        auprc = None

    return accuracy, precision, recall, f1, auroc, auprc


# Train and test model for each unique antigen.epitope code
results = []
for epitope in unique_epitopes:
    epitope_data = data[data['antigen.epitope'] == epitope]
    X = np.column_stack((np.vstack(epitope_data['cdr3_a_aa'].values), np.vstack(epitope_data['cdr3_b_aa'].values)))
    y = epitope_data['binder'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)

    results.append({
        'antigen_epitope': epitope,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auprc': auprc
    })

# Map from integers to epitope strings
epitope_mapping = {i + 2: s for i, s in enumerate([
    "IVTDFSVIK", "KLGGALQAK", "GILGFVFTL", "ELAGIGILTV", "HPVTKYIM",
    "RAQAPPPSW", "RLRAEAQVK", "AVFDRKSDAK", "RAKFKQLL", "TPRVTGGGAM",
    "YLQPRTFLL", "PKYVKQNTLKLAT", "TFEYVSQPFLMDLE", "NLVPMVATV", "GLCTLVAML",
    "SPRWYFYYL", "DATYQRTRALVR", "ATDALMTGF", "TTDPSFLGRY", "CINGVCWTV",
    "MEVTPSGTWL", "KSKRTPMGF", "QYIKWPWYI", "LLWNGPMAV", "LTDEMIAQY",
    "NQKLIANQF", "RPRGEVRFL", "NYNYLYRLF", "KTFPPTEPK", "GPRLGVRAT"
])}

# Print performance metrics
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
