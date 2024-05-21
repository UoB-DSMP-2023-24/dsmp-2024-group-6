import pandas as pd

csv_file_path = '/Users/lifushen/Desktop/giana_distance_matrix_visual/data_clean_small_MusMusculus.csv'
df = pd.read_csv(csv_file_path, sep=',')

selected_columns = df.iloc[:, [4, 11]]

new_csv_file_path = 'map_a.csv'
selected_columns.to_csv(new_csv_file_path, index=False)

