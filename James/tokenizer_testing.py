from transformers import BertModel, BertTokenizer
import pandas as pd
import re
import torch

# File path of the data text file
file_path = r'C:\Users\james\OneDrive\Documents\Bristol\Other\practice data\small_string_data_for_token_test.csv'

# Read the text file into a DataFrame
df = pd.read_csv(file_path, sep=',')

# Store features as a list
features_list = df.columns.tolist()
print(features_list)

# View proportion of missing data
print(f'\nMissing values distribution:\n{df.isnull().mean()*100}')

# Count the occurrence of each score value in the dataframe
score_key = [0, 1, 2, 3]
score_occurrence_list = []

for score in score_key:
    occurrences = (df['vdjdb.score'] == score).sum()
    score_occurrence_list.append(occurrences)
    print(f"The number of occurrences of '{score}' in vdjdb.score is: {occurrences}")

# Zip score key and occurrence value as tuple
zipped_score_occurrence = zip(score_key, score_occurrence_list)
print(list(zipped_score_occurrence))

# Check data type of each column
print(f'\nColumn datatypes:\n{df.dtypes}')

# Get all features with string or mixed type values
str_cols = [feature for feature in df.columns if df[feature].dtype != 'int64']
print(str_cols)

# Remove all leading and trailing characters from columns with string type
for i in str_cols:
    df[i] = df[i].str.strip()

# Insert space between every character in the 'cdr3' column
df['cdr3'] = df['cdr3'].apply(lambda x: ' '.join(x))

# Convert NaN to a blank string type
df = df.fillna('')


# Initialise the tokenizer using the Rostlab/prot_bert model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

model = BertModel.from_pretrained("Rostlab/prot_bert")

sequence_to_tokenize = df['cdr3'].tolist()

sequence_to_tokenize = [re.sub(r"[UZOB]", "X", seq) for seq in sequence_to_tokenize]
tokenized_sequences = [tokenizer(seq, padding=True, truncation=False, return_tensors="pt") for seq in sequence_to_tokenize]

# Outputs will store the results of passing tokenized sequences to the model
outputs = [model(**seq) for seq in tokenized_sequences]

# Initialize an empty list to store the tensors
tensor_list = []

# Convert each BatchEncoding object into a tensor and append to the list
for seq in tokenized_sequences:
    input_ids = seq['input_ids']
    attention_mask = seq['attention_mask']
    token_type_ids = seq['token_type_ids'] if 'token_type_ids' in seq else None
    tensor_list.append((input_ids, attention_mask, token_type_ids))

# Store the list of tensors in a new column in the DataFrame
df['cdr3_tokenized'] = tensor_list
print(df.head())

#df.to_csv(r'C:\Users\james\OneDrive\Documents\Bristol\Mini Project\vdjdb-2023-06-01\cdr3_tokenized.csv', index= False)