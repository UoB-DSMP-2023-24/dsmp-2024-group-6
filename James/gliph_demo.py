import pandas as pd

# Load the data
df = pd.read_csv('data_clean.csv', sep='\t')

# Delete rows of complex.id == 0
df = df[df['complex.id'] != 0]

# Pivot the dataframe and set 'complex.id' as index
df_pivot = df.pivot(index='complex.id', columns='gene')

# Flatten the MultiIndex columns
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

# Reset index to bring 'complex.id' back as a column
df_pivot.reset_index(inplace=True)

# Rename columns as needed
df_pivot = df_pivot.rename(columns={
    'cdr3_TRA': 'cdr3a',
    'v.segm_TRA': 'TRBVTRA',
    'j.segm_TRA': 'TRBJTRA',
    'cdr3_TRB': 'cdr3b',
    'v.segm_TRB': 'TRBVTRB',
    'j.segm_TRB': 'TRBJTRB'
})

# Select only the desired columns and drop the rest
df_pivot = df_pivot[['cdr3b', 'TRBVTRB', 'TRBJTRB', 'cdr3a', 'TRBVTRA', 'TRBJTRA']]

# Remove rows with NaN values
df_pivot = df_pivot.dropna()

print(df_pivot)

export_txt = df_pivot.to_csv('gliph_input.txt', sep='\t', index=False)
