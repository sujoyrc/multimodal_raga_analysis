'''
Script is used to add the mid-point frame number of each subsequence to the summary.csv file
'''

import pandas as pd
import numpy as np

summary_path = '../Seqs/summary.csv'
dest_path = '../Seqs/mod-summary.csv'

sum_df = pd.read_csv(summary_path)
t_mid_array = []
for ind, row in sum_df.iterrows():

    # 1. calculate the midpoint
    mid_point = np.around(row['start_times'] + 6, 2)

    # 2. round the mid point to 0.04s precision
    mod_val = np.around(mid_point*100, 0)%4
    if mod_val == 3:
        mid_point += 0.01
    else:
        mid_point -= (mod_val * 0.01)
    mid_point = np.around(mid_point, 2)

    # 3. calculate the time instant
    t_mid = np.around(mid_point * 25, 0)
    t_mid_array.append(t_mid)

sum_df['Mid Frame Number'] = t_mid_array
sum_df.to_csv(dest_path, index=False)