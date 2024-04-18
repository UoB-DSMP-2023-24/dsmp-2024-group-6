import pandas as pd
def generate_negative(df, neg_per_pos=5):
    """
    Generate negative data by randomly sampling from the dataset.
    For every positive datapoint, we resample the TCRs among tcrs not binding the same epitope until we have neg_per_pos negative datapoints.

    """
    epit = df["antigen.epitope"].unique()
    epitope_A = {epitope: df[df["antigen.epitope"] == epitope]["cdr3_a_aa"].unique() for epitope in epit}
    epitope_B = {epitope: df[df["antigen.epitope"] == epitope]["cdr3_b_aa"].unique() for epitope in epit}
    df_columns = df.columns.tolist()
    df_columns.append("binder")
    df2 = pd.DataFrame(columns=df_columns)
    for i in range(len(df)):
        neg = 0
        row = df.loc[i]
        while neg < neg_per_pos:
            s = df.sample(n=1).iloc[0]

            if s["antigen.epitope"] != row["antigen.epitope"]:
                if (s["cdr3_a_aa"] == None) & (s["cdr3_b_aa"] == None):
                    continue
                elif s["cdr3_a_aa"] == None:
                    if s["cdr3_b_aa"] in epitope_B[row["antigen.epitope"]]:
                        continue

                elif s["cdr3_b_aa"] == None:
                    if s["cdr3_a_aa"] in epitope_A[row["antigen.epitope"]]:
                        continue

                else:
                    if s["cdr3_a_aa"] in epitope_A[row["antigen.epitope"]]:
                        continue
                    elif s["cdr3_b_aa"] in epitope_B[row["antigen.epitope"]]:
                        continue
                    else:
                        # new_row = {"antigen.epitope":row["antigen.epitope"],"cdr3_a_aa":s["cdr3_a_aa"],"cdr3_b_aa":s["cdr3_b_aa"], "binder":0, "MHC":row["MHC"]},
                        # Create a new DataFrame with the data you want to append
                        new_data = s.copy()  # 复制s到new_data
                        new_data['antigen.epitope'] = row['antigen.epitope']
                        new_data['binder'] = 0  # 添加一个新的列'binder'，并设置其值为0

                        # 将new_data转换为DataFrame
                        new_data = pd.DataFrame([new_data])

                        # Use pandas.concat to concatenate the new_data DataFrame to df2
                        df2 = pd.concat([df2, new_data], ignore_index=True)

                        # df2 = pd.concat({"antigen.epitope":row["antigen.epitope"],"cdr3_a_aa":s["cdr3_a_aa"],"cdr3_b_aa":s["cdr3_b_aa"], "binder":0, "MHC":row["MHC"]}, ignore_index=True)
                        neg = neg + 1
    for index, row in df.iterrows():
        new_data = row.copy()
        new_data['binder'] = 1
        new_data = pd.DataFrame([new_data])
        df2 = pd.concat([df2, new_data], ignore_index=True)
    return df2

df = pd.read_csv("/Users/berlin/Documents/UoB/DSMP/pycharm_project_DSMP/dsmp-2024-group-6/data_clean_small.csv", sep='\t')
print(df.columns)

# df = df.iloc[:100]

df_2 = generate_negative((df))

print(df_2)

df_2.to_csv("data_clean_small_neg.csv", sep='\t', index=False)

# df_3 = pd.concat([df, df_2], ignore_index=True)

# print(df_3)