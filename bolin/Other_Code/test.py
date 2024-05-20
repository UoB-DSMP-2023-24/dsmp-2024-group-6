import pandas as pd

dtype_dict = {
    'labels': int,
    'probs': float,
    'description_translation_google': str
}

df = pd.read_csv("/Users/berlin/Desktop/postings_linkedin_individual_0290_part_15.csv", sep='\t', dtype=dtype_dict)

for index, row in df.iterrows():
    if row['description_translation_google'] == '':
        print('xxx')