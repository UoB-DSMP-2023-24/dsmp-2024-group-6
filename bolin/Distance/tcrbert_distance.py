import datetime
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import torch
import pandas as pd

# 需要改
# device = torch.device("cpu")
# device = torch.device("cuda")
device = torch.device("mps")

def add_space(s):
    return ' '.join(s)

tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
model = AutoModelForMaskedLM.from_pretrained("wukevin/tcr-bert-mlm-only").to(device)
data_size = 2000

print("Data size: ", data_size)
time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(str(time_) + ": Loading data")

df = pd.read_csv("./data_clean.csv", sep='\t', index_col=0)

df = df.sample(frac=1, random_state=42)

# 按照 score 列进行降序排序
sorted_df = df.sort_values(by='vdjdb.score', ascending=False)

top_df = sorted_df.head(data_size)

data_input = top_df['cdr3'].apply(add_space).tolist()

time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(str(time_) + ": Tokenizing")
encoded_dict = tokenizer.batch_encode_plus(
    data_input,
    add_special_tokens=True,  # 添加special tokens，比如[BOS]和[EOS]
    max_length=64,  # 设定最大序列长度
    padding='max_length',  # 进行填充到最大长度
    return_attention_mask=True,  # 返回attention mask
    return_tensors='pt',  # 返回PyTorch tensors
).to(device)

time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(str(time_) + ": Encoding data")
# 使用BERT模型编码文本
with torch.no_grad():
    outputs = model(**encoded_dict, output_hidden_states=True)

time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(str(time_) + ": Calculating distance")

dist_matrix = torch.cdist(outputs.hidden_states[12][0:data_size, 0], outputs.hidden_states[12][0:data_size, 0])

time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(str(time_) + ": Finished")
print("Distance Matrix:")
print(dist_matrix)





