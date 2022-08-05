'''
Script is used to add the mid-point frame number of each subsequence to the summary.csv file
'''

import pandas as pd
import numpy as np

summary_path = '../Seqs/summary.csv'
dest_path = '../Final Both Data/summary_precise.csv'

start_times = []
end_times = []
sum_df = pd.read_csv(summary_path)
for ind, row in sum_df.iterrows():
    if np.around(row['start_times']*100, 0)%2 == 1:
        start_times.append(row['start_times'] - 0.01)
        end_times.append(row['end_times'] - 0.01)
    else:
        start_times.append(row['start_times'])
        end_times.append(row['end_times'])
    

sum_df.loc[:, 'start_times'] = start_times
sum_df.loc[:, 'end_times'] = end_times
sum_df.to_csv(dest_path, index=False)