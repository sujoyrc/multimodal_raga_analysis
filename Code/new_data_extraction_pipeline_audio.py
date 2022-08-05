import os
import pandas as pd
import numpy as np
import random
from data_extraction_utils import get_subseq_times, label_index_map, replace_unvoiced
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
sys.path.append('../../CommonScripts/')
from common_utils import addBack, checkPath
import pdb

ogDataFolder = '../Data/'
dataFolder = '../Final Both Data/Audio/'

csvFolder = '../Final Both Data/csvs/'

keywords = ['pitch', 'mask']
summaryFile = '../Final Both Data/summary_precise.csv'

random.seed(42)

def create_csv(keywords, summaryFile, dataFolder, destFolder, subseq_len=1200, class_index_file=None):
    '''
    Creates csv files for each feature based on summary csv created from create_summary()

    Parameters
        keywords (list): column name to extract values from; each value corresponds to a value in the file_extensions list
        summaryFile (str): path to summary file
        dataFolder (str): path to folder with data csvs
        destFolder (str): folder to store feature csvs at
        subseq_len (int): length of each subsequence
        class_index_file (str): path to csv file to map raga labels to integer values. If None, new map is created; if file path doesn't exist, new map is created and stored at the path
    '''

    # pdb.set_trace()
    # convert raga class to integers
    mapping = label_index_map(summaryFile=summaryFile, class_index_file=class_index_file)
    summary = pd.read_csv(summaryFile)
    summary['start_times'] = summary['start_times'].round(decimals=2)   # round off start_times to avoid errors
    summary['end_times'] = summary['end_times'].round(decimals=2)   # round off end_times to avoid errors
    cols = [f'x_{i}' for i in range(subseq_len)]
    cols.append('y')
    cols.append('unique_id')

    for ind, keyword in enumerate(keywords):
        data_vals = []
        for root, _, fileNames in os.walk(dataFolder):
            for fileName in fileNames:
                fileVals = pd.read_csv(os.path.join(root, fileName))
                fileVals['time'] = fileVals['time'].round(decimals=2)   # round time to 2 decimal places to avoid errors while searching for subsequence values based on time
                for _, row in summary.loc[summary['filename'] == os.path.join(root.replace(dataFolder, ogDataFolder), fileName.rsplit('.', 1)[0], fileName.replace('.csv', '-pitch.csv'))].iterrows():
                    temp_vals = fileVals.loc[(fileVals['time'] >= row['start_times']) & (fileVals['time'] <= row['end_times']), keyword].values
                    if keyword == 'pitch':
                        # for pitch vals, replace unvoiced frame value from -3000 to -550
                        # pdb.set_trace()
                        temp_vals = replace_unvoiced(temp_vals, -3000, -550)
                        temp_vals = (temp_vals + 550)/(1900+550)
                        
                    temp_vals = np.append(temp_vals, [mapping[row['raga']], row['unique_id']])
                    data_vals.append(temp_vals)
        data_df = pd.DataFrame(data_vals, columns=cols)     # dataframe for values
        # data_df = data_df.set_index('unique_id')
        data_df.to_csv(checkPath(os.path.join(destFolder, keyword + '.csv')), index=False)



# def main():
#     create_csv(keywords, summaryFile, dataFolder, csvFolder, subseq_len=600, class_index_file=None, replace=True, mask=False)

# if __name__=="__main__":
#     main()