import GIANA_encode
import numpy as np
import pandas as pd

Ndim = 16  ## optimized for isometric embedding
n0 = Ndim * 6
# M0=np.concatenate((np.concatenate((ZERO,M1),axis=1),np.concatenate((M1, ZERO),axis=1)))
ZERO = np.zeros((Ndim, Ndim))
II = np.eye(Ndim)
M0 = np.concatenate((np.concatenate((ZERO, ZERO, II), axis=1), np.concatenate((II, ZERO, ZERO), axis=1),
                     np.concatenate((ZERO, II, ZERO), axis=1)))
## Construct 6-th order cyclic group
ZERO45 = np.zeros((Ndim * 3, Ndim * 3))
M6 = np.concatenate((np.concatenate((ZERO45, M0), axis=1), np.concatenate((M0, ZERO45), axis=1)))

input_file = "/Users/lifushen/Desktop/giana_distance_matrix_visual/data_clean_small_MusMusculus.csv"
data = pd.read_csv(input_file, sep=',', header=0)

expected_columns = ['cdr3_a_aa', 'cdr3_b_aa', 'antigen.epitope']

data['encoded_CDR3a'] = data['cdr3_a_aa'].apply(lambda s: (GIANA_encode.EncodingCDR3(s, M6, n0)))
data['encoded_CDR3b'] = data['cdr3_b_aa'].apply(lambda s: (GIANA_encode.EncodingCDR3(s, M6, n0)))

new_column_order = ['encoded_CDR3a', 'encoded_CDR3b', 'antigen.epitope']

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data[new_column_order]

encoded_CDR3_combined = (data['encoded_CDR3a'].astype(str) + ',' + data['encoded_CDR3b'].astype(str)).replace('\[|\]', '', regex=True)
data.drop(['encoded_CDR3a', 'encoded_CDR3b'], axis=1, inplace=True)
data.insert(0, 'encoded_CDR3a+encoded_CDR3b', encoded_CDR3_combined)

output_file = "output_encoded_ab.csv"
data.to_csv(output_file, index=False)


