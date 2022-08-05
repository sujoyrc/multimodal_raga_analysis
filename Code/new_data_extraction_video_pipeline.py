import csv
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import scipy.signal as signal
from data_extraction_utils import label_index_map
sys.path.append('../../CommonScripts/')
from common_utils import checkPath, addBack
import pdb

# orig_folder = '../Final Video Data/CSV_solo_unfiltered/'
# replaced_nan_folder = '../Final Video Data/Replaced_nan/'
# filter_folder = '../Final Video Data/Filter_values/'
normalized_folder = '../Final Both Data/Video/'
vid_seqs_folder = '../Final Both Data/csvs/'


seqsFolder = '../Final Both Data/'


keywords = [
    ('RWrist', 'x'),
    ('RWrist', 'y'),
    ('LWrist', 'x'),
    ('LWrist', 'y')
]

# fps = 50

def replace_nan(old_path=None, new_path=None, store_conf=False):
    '''
    Function to replace nans if present in a csv
    '''
    hist_vals = defaultdict(lambda: np.zeros(shape=(11,)))
    for root, _, fileNames in os.walk(old_path):
        for fileName in fileNames:
            # pdb.set_trace()
            csvFile = pd.read_csv(os.path.join(root, fileName), header=[0, 1])
            new_df = {}
            for cols in csvFile.columns:
                if 'Ear' in cols[0]:
                    # skip Ear column
                    continue
                if cols[1] == 'c' and store_conf:
                    hist_vals[cols[0]] += np.histogram(csvFile[cols].values, bins=np.arange(0, 1.2, 0.1), density=False)[0]
                else:
                    data_vals = csvFile[cols].values

                    # find nan indices
                    nan_idx = np.where(np.isnan(data_vals))[0]

                    # replace nan values
                    for idx in nan_idx:
                        if idx == 0:
                            # if first value is nan replace it with the next non-nan number
                            data_vals[idx] = np.where(~np.isnan(data_vals))[0][0]
                        else:
                            data_vals[idx] = data_vals[idx - 1]

                    # store the changed data as a new column
                    new_df[cols] = data_vals
            # save the csv file in the new location
            pd.DataFrame(new_df).to_csv(checkPath(os.path.join(root, fileName).replace(old_path, new_path)), index=False)
    if store_conf:
        fig, axs = plt.subplots(3, 4)
        for plt_idx, key in enumerate(list(hist_vals.keys())):
            axs[plt_idx//4, plt_idx%4].bar(np.arange(0, 1.1, 0.1), hist_vals[key], width=0.1, align='edge')
            axs[plt_idx//4, plt_idx%4].set_title(key)
            #axs[plt_idx//4, plt_idx%4].set_yticks([0, 10000, 20000])
            #axs[plt_idx//4, plt_idx%4].set_yticklabels(['0', '10k', '20k'])
            fig.suptitle('Histogram of confidence for each keypoint')
        fig.tight_layout()
        fig.savefig(os.path.join(new_path, 'confidence.png'))
        
def filter_data(old_path, new_path):
    for root, _, fileNames in os.walk(old_path):
        for fileName in fileNames:
            csvFile = pd.read_csv(os.path.join(root, fileName), header=[0, 1])
            new_df = {}
            for cols in csvFile.columns:
                if not (cols[1] == 'x' or cols[1] == 'y'):
                    new_df[cols] = csvFile[cols].values.astype(int)
                else:
                    data_vals = csvFile[cols].values
                    yhat = signal.savgol_filter(data_vals, 13, 4)
                    new_df[cols] = yhat
             # save the csv file in the new location
            pd.DataFrame(new_df).to_csv(checkPath(os.path.join(root, fileName).replace(old_path, new_path)), index=False)

def normalize_data(old_path, new_path):
    for root, _, fileNames in os.walk(old_path):
        for fileName in fileNames:
            xmax = 0
            ymax = 0
            xmin = 10000
            ymin = 10000
            csvFile = pd.read_csv(os.path.join(root, fileName), header=[0, 1])
            new_df = {}
            for cols in csvFile.columns[1:]:
                data_num = csvFile[cols].values
                if cols[1] == 'x':
                    xmax = max(xmax, np.max(data_num))
                    xmin = min(xmin, np.min(data_num))
                elif cols[1] == 'y':
                    ymax = max(ymax, np.max(data_num))
                    ymin = min(ymin, np.min(data_num))

                if cols[0] == "MidHip" and cols[1] == "x":
                    data_midhip_x = data_num
                if cols[0] == "Neck" and cols[1] == "x":
                    data_neck_x = data_num

            xc = (np.mean(data_neck_x) + np.mean(data_midhip_x)) / 2    # 0 of the x axis
            width = 2 * max(xc - xmin, xmax - xc)
            height = ymax - ymin

            for cols in csvFile.columns:
                data_vals = csvFile[cols].values
                if cols[1] == 'x':
                    data_vals = np.round(list((np.array(data_vals) - xmin) / (width)), 5)
                elif cols[1] == 'y':
                    data_vals = np.round(list((np.array(data_vals) - ymin) / height), 5)
                new_df[cols] = data_vals
             # save the csv file in the new location
            pd.DataFrame(new_df).to_csv(checkPath(os.path.join(root, fileName).replace(old_path, new_path)), index=False)

def create_csv(keywords, summaryFile, dataFolder, destFolder, subseq_len=300, class_index_file=None, replace=True, fps=25):
    # convert raga class to integers
    mapping = label_index_map(summaryFile=summaryFile, class_index_file=class_index_file)
    summary = pd.read_csv(summaryFile)
    summary['start_times'] = summary['start_times'].round(decimals=2)   # round off start_times to avoid errors
    summary['end_times'] = summary['end_times'].round(decimals=2)   # round off end_times to avoid errors
    cols = [f'x_{i}' for i in range(subseq_len)]
    cols.append('y')
    cols.append('unique_id')
    for ind, keyword in enumerate(keywords):
        print('Creating file for ' + keyword[0] + ' ' + keyword[1])
        if os.path.exists(os.path.join(destFolder, keyword[0] + '-' + keyword[1] + '.csv')) and not replace:
            print(f'{keyword[0]} {keyword[1]} already exists')
            # continue
        data_vals = []  # values to populate dataframe
        for root, _, fileNames in os.walk(dataFolder):
            for fileName in fileNames:
                fileVals = pd.read_csv(os.path.join(root, fileName), header=[0, 1])    # normalized video data
                time_vals = (fileVals['Body Part', 'Variable']/fps).round(decimals=2)   # stores the time values corresponding to each row in the dataframe
                # pdb.set_trace()
                for _, row in summary.loc[summary['filename'] == os.path.join(root, fileName.rsplit('.', 1)[0], fileName).replace(addBack(dataFolder), '../Data/').replace('.csv', '-pitch.csv')].iterrows():
                    temp_vals = fileVals.loc[(time_vals >= row['start_times']) & (time_vals < np.round(row['start_times']+12, decimals=2)), keyword].values
                    if len(temp_vals) == subseq_len - 1:
                        # if the subsequence is short due to sampling issue, append the last sequence to the end
                        temp_vals = np.append(temp_vals, temp_vals[-1])
                    temp_vals = np.append(temp_vals, [mapping[row['raga']], row['unique_id']])
                    data_vals.append(temp_vals)
        data_df = pd.DataFrame(data_vals, columns=cols)     # dataframe for values
        # data_df = data_df.set_index('unique_id')
        data_df.to_csv(checkPath(os.path.join(destFolder, keyword[0] + '-' + keyword[1] + '.csv')), index=False)

# replace_nan(orig_folder, replaced_nan_folder, store_conf=False)
# filter_data(old_path=replaced_nan_folder, new_path=filter_folder)
# normalize_data(filter_folder, normalized_folder)
# create_csv(keywords=keywords, summaryFile=os.path.join(seqsFolder, 'summary_precise.csv'), dataFolder=normalized_folder, destFolder=vid_seqs_folder, subseq_len=600, class_index_file=os.path.join(seqsFolder, 'raga_mapping.csv'), replace=True)