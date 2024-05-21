import pandas as pd

csv_file_path = '/Users/lifushen/Desktop/giana_distance_matrix_visual/merged_cleaned_reordered_data.csv'
df = pd.read_csv(csv_file_path, sep=',')

selected_columns = df.iloc[:, [0, 2]]

new_csv_file_path = 'map_a_b.csv'
selected_columns.to_csv(new_csv_file_path, index=False)

