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
from common_utils import checkPath
import pdb

splitFolder = '/home/nithya/Projects/Gesture Analysis/Seqs/splits/easy_1/'
csvFolder = '/home/nithya/Projects/Gesture Analysis/Seqs/csvs/'
finalFolder = '/home/nithya/Projects/Gesture Analysis/Final Both Data/npzs/audio-1200/easy_1/'
subseq_len = 1200

AUDIO = ['pitch-pitch.csv', 'mask.csv']
VIDEO = ['LWrist-x.csv', 'LWrist-y.csv', 'RWrist-x.csv', 'RWrist-y.csv']

def train_test_split(splitFolder, csvFolder, destFolder, subseq_len=1200, option=None):
    '''
    splits data into train and test npz files.

    Parameters
        splitFolder (str): folder path with splits (list of files to include in test set)
        csvFolder (str): folder path to csvs with feature values
        destFolder (str): folder to store train and test npz files at
        subseq_len (int): subsequence length
        normalize (bool): if True, will save a normalized version of the data
        option (str): if None, will add all 6 channels. Otherwise has to be either 'AUDIO' or 'VIDEO'
    '''
    if option == 'BOTH':
        print('Using audio and video features')
        csvFiles = [pd.read_csv(os.path.join(csvFolder, fileName)) for fileName in os.listdir(csvFolder)]  
    elif option == 'AUDIO':
        print('Using only audio features')
        csvFiles = [pd.read_csv(os.path.join(csvFolder, fileName)) for fileName in AUDIO]
    elif option == 'VIDEO':
        print('Using only video features')
        csvFiles = [pd.read_csv(os.path.join(csvFolder, fileName)) for fileName in VIDEO]
    else:
        raise Exception('option is incorrect')
    for fileName in os.listdir(splitFolder):
        print('Creating split - ' + fileName)
        with open(os.path.join(splitFolder, fileName), 'r') as f:
            test_files = f.readlines()
            test_files = [test_file.rstrip('\n') for test_file in test_files]
        # data arrays
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        train_ids = []
        test_ids = []
        xCols = [f'x_{i}' for i in range(subseq_len)]
        # pdb.set_trace()
        for unique_id in csvFiles[0]['unique_id'].values:
            X = []
            for csv in csvFiles:
                try:
                    X.append(csv.loc[csv['unique_id'] == unique_id, xCols].values[0])
                except:
                    print(csv.loc[csv['unique_id'] == unique_id, xCols].values)
                    pdb.set_trace()
            y = csv.loc[csv['unique_id'] == unique_id, 'y']

            if unique_id.rsplit('_', 1)[0] in test_files:
                # test data
                X_test.append(X)
                y_test.append(y)
                test_ids.append(unique_id)
            else:
                # train data
                X_train.append(X)
                y_train.append(y)
                train_ids.append(unique_id)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        train_ids = np.array(train_ids)
        test_ids = np.array(test_ids)

        # swap axes to shift the channels from the penultimate to the last dimension i.e. (#training samples, #channels, 1200) -> (#training samples, 1200, #channels)
        X_train = np.swapaxes(X_train, 1, 2)
        X_test = np.swapaxes(X_test, 1, 2)

        # pdb.set_trace()
        splitFileNames = fileName.rsplit("-", 1)
        if len(splitFileNames) > 1:
            destFilename = os.path.join(destFolder, f'{fileName.rsplit("-", 1)[1].rsplit(".", 1)[0]}.npz')
        else:
            destFilename = os.path.join(destFolder, fileName.rsplit('.', 1)[0] + '.npz')
        np.savez(checkPath(destFilename), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, train_ids=train_ids, test_ids=test_ids, channels=np.array([f.rsplit('.', 1)[0] for f in os.listdir(csvFolder)]))

# train_test_split(splitFolder, csvFolder, finalFolder, subseq_len=subseq_len, option='AUDIO')