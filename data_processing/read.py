import pandas as pd

df = pd.read_csv("/Users/berlin/Documents/UoB/DSMP/pycharm_project_DSMP/vdjdb-2023-06-01/vdjdb.txt", sep='	')

print(df)

pd.set_option('display.max_columns', 34)
print(df.head(5))

print(df.columns)

print(df.info())

print(df.describe().T)

# Function to summarize the DataFrame
def summarize_dataframe(df):
    summary = pd.DataFrame({
        'Variable Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    return summary

# Display the summary
print(summarize_dataframe(df))

unique_values_1 = df['antigen.species'].unique()
# unique_values_2 = df['meta.subject.cohort'].unique()

# 打印所有唯一值
print(unique_values_1)
print()
print()
# print(unique_values_2)


