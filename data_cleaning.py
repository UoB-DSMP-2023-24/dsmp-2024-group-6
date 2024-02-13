import pandas as pd

df = pd.read_csv("../vdjdb-2023-06-01/vdjdb.txt", sep='	')

def check_alpha_cdr3(item):
    # Check if it starts with C and ends with F or W
    return item.startswith('C') and (item.endswith('F') or item.endswith('W'))

def check_v_alpha(item):
    # Check if string starts with TRAV
    return item.startswith('TRAV')

def check_mhc_a(item):
    # Check if string starts with HLA-
    return item.startswith('HLA-')

def check_mhc_class(item):
    if item == 'MHCI' or item == 'MHCII':
        return True

    return False

count_cdr3_invalid = 0
count_v_alpha_invalid = 0
count_mhc_a_invalid = 0
count_mhc_class_invalid = 0

df["is_cdr3_alpha_valid"] = 1
# df["is_v_alpha_valid"] = 1
df["is_mhc_a_valid"] = 1
# df["is_mhc_class_valid"] = 1


for index, row in df.iterrows():
    if str(row['gene']) == "TRA":
        is_cdr3_valid = check_alpha_cdr3(row['cdr3'])

        if not str(row['v.segm']) == "nan":
            is_v_alpha_valid = check_v_alpha(str(row['v.segm']))

        if not is_cdr3_valid:
            df.at[index, 'is_cdr3_alpha_valid'] = 0
            count_cdr3_invalid += 1

        if not is_v_alpha_valid:
            # df.at[index, "is_v_alpha_valid"] = 0
            count_v_alpha_invalid += 1


    is_mhc_a_valid = check_mhc_a(row['mhc.a'])

    if not is_mhc_a_valid:
        count_mhc_a_invalid += 1
        df.at[index, "is_mhc_a_valid"] = 0

    is_mhc_class_valid = check_mhc_class(row['mhc.class'])

    if not is_mhc_class_valid:
        count_mhc_class_invalid += 1
        # df.at[index, "is_mhc_class_valid"] = 0

print("-" * 50)
print("count_cdr3_valid: ", count_cdr3_invalid)
print("-" * 50)
print("count_v_alpha_valid: ", count_v_alpha_invalid)
print("-" * 50)
print("count_mhc_a_valid: ", count_mhc_a_invalid)
print("-" * 50)
print("count_mhc_class_valid: ", count_mhc_class_invalid)
print("-" * 50)

print(df)

df.to_csv("./df_cleaned.csv", sep='\t')

print()


