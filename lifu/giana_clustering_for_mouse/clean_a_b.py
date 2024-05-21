import pandas as pd


def preprocess_file(input_filename, output_filename):
    # Load the data, skipping the first two lines
    data = pd.read_csv(input_filename, header=None, skiprows=2, sep='\t')

    # Concatenate the first and second columns and form a new column
    data['cdr3_a_aa+cdr3_b_aa'] = data.iloc[:, 0].astype(str) + data.iloc[:, 1].astype(str)

    # Reorder the columns to have the concatenated one first, then drop the old columns
    data = data[['cdr3_a_aa+cdr3_b_aa'] + [col for col in data.columns if col not in [0, 1, 'cdr3_a_aa+cdr3_b_aa']]]

    # Assign new column names
    data.columns = ['cdr3_a_aa+cdr3_b_aa'] + ['cluster.id', 'antigen.epitope'] + list(data.columns[3:])

    # Save the DataFrame to a new text file
    data.to_csv(output_filename, index=False, sep='\t', header=True)


# Define the input and output file names
input_filename = '/Users/lifushen/Desktop/giana_for_mouse/clean_altogther---DualChainRotationEncodingBL62.txt'  # Replace with the path to your input file
output_filename = '/Users/lifushen/Desktop/giana_for_mouse/clean_altogther_clean---DualChainRotationEncodingBL62.txt'  # Replace with the desired output file path
# Call the function to preprocess the file and save it
preprocess_file(input_filename, output_filename)