import numpy as np
from numpy.core.defchararray import split
import pandas as pd
import random
from copy import copy
import os
import sys
import pdb
sys.path.append('../../CommonScripts/')
from utils import checkPath

# replace_unvoiced_val = -550
seqsFolder = '../Seqs_Jin-Normal/easy-1/'
def cv_split(filename_df, cv_file=os.path.join(seqsFolder, 'cv.csv')):
    '''Splits the data into `n_splits` cv groups

    Parameters
        filename_df (pd.DataFrame): dataframe containing filename for each raga
        n_splits (int): number of cv splits
        cv_file (str): file path to store cv splits in
    
    Returns
        cv_df (pd.DataFrame): shuffled dataframe contains filename, raag,  cv_split
    '''
    random.seed(42)
    cv_dict = {'Filename': [], 'Raag': [], 'CV': []}
    group_by_artist = filename_df.groupby(by='Singer')
    with open(os.path.join(seqsFolder, 'cv_id.txt'), 'w') as f:
        f.write('CV Group\tSinger\n')
        for artist_id, (artist, df_artist) in enumerate(group_by_artist):
            for filename, df_filename in df_artist.groupby(by='Filename'):
                cv_dict['Filename'].append(filename)
                cv_dict['Raag'].append(df_filename['Raag'].values[0])  # the same filename
                cv_dict['CV'].append(artist_id)
            f.write(f'{artist_id}\t{artist}\n')
    cv_df = pd.DataFrame(cv_dict)
    cv_df.to_csv(cv_file, index=False)
    return cv_df

def _write_data(train_summary_file=os.path.join(seqsFolder, 'trainFileList.csv'), trainDest=os.path.join(seqsFolder,'train.csv'), test_summary_file=os.path.join(seqsFolder,'testFileList.csv'), testDest=os.path.join(seqsFolder,'test.csv'), raag_to_idx_file=os.path.join(seqsFolder,'raag_to_id.txt'), seq_len=3000):
    '''
    Deprecated
    Loads all arrays, with labels, adds cv group labels if required, shuffles the arrays, adds them to a csv

    Parameters
        train_summary_file (str): path to metadata csv for all train arrays stored
        trainDest (str): path to store train csv at
        test_summary_file (str): path to metadata csv for all test arrays stored
        seq_len (int): length of subsequence
        raag_to_idx_file (str): file path to store raag to index mapping; if none, not stored
        
    Returns
        None
    '''
    train_data = {f'x_{i}': [] for i in range(seq_len)}
    train_data['y'] = []
    train_data['CV'] = []
    test_data = {f'x_{i}': [] for i in range(seq_len)}
    test_data['y'] = []

    # train set
    train_summary = pd.read_csv(train_summary_file)
    cv_train_list = cv_split(train_summary, cv_file=os.path.join(seqsFolder, 'cv_train.csv'))
    cv_train_list['Raag'] = cv_train_list['Raag'].astype('category')
    raag_to_idx = dict(enumerate(cv_train_list['Raag'].cat.categories))
    cv_train_list['Raag'] = cv_train_list['Raag'].cat.codes
    if raag_to_idx_file is not None:
        with open(raag_to_idx_file, 'w') as f:
            f.write('Raag\tIndex\n')
            for key, value in raag_to_idx.items():
                f.write(f'{key}\t{value}\n')
    # load data
    for dfi, row in cv_train_list.iterrows():
        vals = np.load(row['Filename'])
        for x in vals['arr_0']:
            for arri, xval in enumerate(x):
                train_data[f'x_{arri}'].append(xval)
            train_data['y'].append(row['Raag'])
            train_data['CV'].append(row['CV'])
    # create dataframe and store it
    train_data = pd.DataFrame(train_data)
    train_data = train_data.sample(frac=1, random_state=42) # shuffle data
    # train_data.iloc[:, :-2] = train_data.iloc[:, :-2].replace(-3000, replace_unvoiced_val) # replace unvoiced frame value
    train_data.to_csv(checkPath(trainDest), index=False)    # save file
    
    # test set
    test_summary = pd.read_csv(test_summary_file)
    # pdb.set_trace()
    test_summary['Raag'].replace(list(raag_to_idx.values()), list(raag_to_idx.keys()), inplace=True)
    # load data
    for dfi, row in test_summary.iterrows():
        vals = np.load(row['Filename'])
        for x in vals['arr_0']:
            for arri, xval in enumerate(x):
                test_data[f'x_{arri}'].append(xval)
            test_data['y'].append(row['Raag'])
    # create dataframe and store it
    test_data = pd.DataFrame(test_data)
    test_data = test_data.sample(frac=1, random_state=42) # shuffle data
    # test_data.iloc[:, :-1] = test_data.iloc[:, :-2].replace(-3000, replace_unvoiced_val) # replace unvoiced frame value
    test_data.to_csv(checkPath(testDest), index=False)    # save file

def write_data_easy_split_1(trainDest=os.path.join(seqsFolder,'train_easy.csv'), testDest=os.path.join(seqsFolder,'test_easy.csv'), summary_file=os.path.join(checkPath(seqsFolder).rsplit('/', 2)[0], 'summary.csv'), test_file_list=os.path.join(seqsFolder, 'easy_split_1_test.txt'), raag_to_idx_file=os.path.join(checkPath(seqsFolder).rsplit('/', 2)[0],'raag_to_id.txt'), seq_len=3000):
    '''
    Loads all npz arrays and splits them into train and test csv files based on Jin's easy split

    Parameters
        trainDest (str): file path to store train file at
        testDest (str): file path to store test file at
        summary_file (str): file path with metadata on the numpy arrays stored
        test_file_list (str): file path with a list of folders (these are the same as the filename, just without the file extension) to add to the test dataset; all other files will be added to the train set
        raag_to_idx_file (str): stores a mapping to raga name to the number used in the data to represent it; in None, no mapping will be stored
        seq_len (int): Length of each sequence
    
    Returns
        None
    '''

    # load summary file and replace raga labels with numbers
    summary = pd.read_csv(summary_file)
    if raag_to_idx_file is not None and not os.path.isfile(raag_to_idx_file):
        summary['Raag'] = summary['Raag'].astype('category')
        raag_to_idx = dict(enumerate(summary['Raag'].cat.categories))
        with open(raag_to_idx_file, 'w') as f:
            f.write('Raag\tIndex\n')
            for key, value in raag_to_idx.items():
                f.write(f'{key}\t{value}\n')
    if os.path.isfile(raag_to_idx_file):
        raag_to_idx_df = pd.read_csv(raag_to_idx_file, sep='\t', header=None)
        raag_to_idx = dict(zip(raag_to_idx_df.iloc[:, 0], raag_to_idx_df.iloc[:, 1]))
        summary['Raag'].replace(list(raag_to_idx.values()), list(raag_to_idx.keys()), inplace=True)

    for singer, _ in summary.groupby('Singer'):
        # create a list of files to put in test 
        with open(test_file_list.rsplit('.', 1)[0] + f'-{singer}.txt', 'r') as f:
            test_files = f.readlines()
            test_files = [test_file.rstrip('\n') for test_file in test_files]
        
        # create dictionaries to store train and test data in
        train_data = {f'x_{i}': [] for i in range(seq_len)}
        train_data['y'] = []
        test_data = {f'x_{i}': [] for i in range(seq_len)}
        test_data['y'] = []

        # load arrays
        for filename, df in summary.groupby('Filename'):
            vals = np.load(filename)
            raga = df['Raag'].values[0]     # all raga values should be the same so just pick the first one
            if np.any([x in filename for x in test_files]):
                # filename belongs to the test set
                for x in vals['arr_0']:
                    for arri, xval in enumerate(x):
                        test_data[f'x_{arri}'].append(xval)
                    test_data['y'].append(raga)
            else:
                # filename belongs to the train set
                for ind, x in enumerate(vals['arr_0']):
                    for arri, xval in enumerate(x):
                        if np.any(xval > 1950):
                            pdb.set_trace()
                        train_data[f'x_{arri}'].append(xval)
                    train_data['y'].append(raga)
            
        # save dataframes
        # train
        train_data = pd.DataFrame(train_data)
        train_data = train_data.sample(frac=1, random_state=42) # shuffle data
        # train_data.iloc[:, :-1] = train_data.iloc[:, :-1].replace(-3000, replace_unvoiced_val) # replace unvoiced frame value
        train_data.to_csv(checkPath(trainDest.rsplit('.', 1)[0] + f'-{singer}.csv'), index=False)    # save file
        # test
        test_data = pd.DataFrame(test_data)
        test_data = test_data.sample(frac=1, random_state=42) # shuffle data
        # test_data.iloc[:, :-1] = test_data.iloc[:, :-1].replace(-3000, replace_unvoiced_val) # replace unvoiced frame value
        test_data.to_csv(checkPath(testDest.rsplit('.', 1)[0] + f'-{singer}.csv'), index=False)    # save file

def write_data_easy_split_5(trainDest=os.path.join(seqsFolder,'train_easy.csv'), testDest=os.path.join(seqsFolder,'test_easy.csv'), summary_file=os.path.join(seqsFolder, 'summary.csv'), test_file_list=os.path.join(seqsFolder, 'easy_split_test.txt'), raag_to_idx_file=os.path.join(seqsFolder,'raag_to_id.txt'), seq_len=3000):
    '''
    Loads all npz arrays and splits them into train and test csv files based on Jin's easy split

    Parameters
        trainDest (str): file path to store train file at
        testDest (str): file path to store test file at
        summary_file (str): file path with metadata on the numpy arrays stored
        test_file_list (str): file path with a list of folders (these are the same as the filename, just without the file extension) to add to the test dataset; all other files will be added to the train set
        raag_to_idx_file (str): stores a mapping to raga name to the number used in the data to represent it; in None, no mapping will be stored
        seq_len (int): Length of each sequence
    
    Returns
        None
    '''

    # create a list of files to put in test 
    with open(test_file_list, 'r') as f:
        test_files = f.readlines()
        test_files = [test_file.rstrip('\n') for test_file in test_files]
    
    # create dictionaries to store train and test data in
    train_data = {f'x_{i}': [] for i in range(seq_len)}
    train_data['y'] = []
    test_data = {f'x_{i}': [] for i in range(seq_len)}
    test_data['y'] = []

    # load summary file and replace raga labels with numbers
    summary = pd.read_csv(summary_file)
    if raag_to_idx_file is not None and not os.path.isfile(raag_to_idx_file):
        summary['Raag'] = summary['Raag'].astype('category')
        raag_to_idx = dict(enumerate(summary['Raag'].cat.categories))
        with open(raag_to_idx_file, 'w') as f:
            f.write('Raag\tIndex\n')
            for key, value in raag_to_idx.items():
                f.write(f'{key}\t{value}\n')
    if os.path.isfile(raag_to_idx_file):
        raag_to_idx_df = pd.DataFrame(raag_to_idx_file, sep='\t', header=None)
        raag_to_idx = dict(zip(raag_to_idx_df.iloc[:, 0], raag_to_idx_df.iloc[:, 1]))
        summary['Raag'].replace(list(raag_to_idx.values()), list(raag_to_idx.keys()), inplace=True)

    # load arrays
    for filename, df in summary.groupby('Filename'):
        vals = np.load(filename)
        raga = df['Raag'].values[0]     # all raga values should be the same so just pick the first one
        if np.any([x in filename for x in test_files]):
            # filename belongs to the test set
            for x in vals['arr_0']:
                for arri, xval in enumerate(x):
                    test_data[f'x_{arri}'].append(xval)
                test_data['y'].append(raga)
        else:
            # filename belongs to the train set
            for x in vals['arr_0']:
                for arri, xval in enumerate(x):
                    train_data[f'x_{arri}'].append(xval)
                train_data['y'].append(raga)
        
    # save dataframes
    # train
    train_data = pd.DataFrame(train_data)
    train_data = train_data.sample(frac=1, random_state=42) # shuffle data
    train_data.iloc[:, :-1] = train_data.iloc[:, :-1].replace(-3000, replace_unvoiced_val) # replace unvoiced frame value
    train_data.to_csv(checkPath(trainDest), index=False)    # save file
    # test
    test_data = pd.DataFrame(test_data)
    test_data = test_data.sample(frac=1, random_state=42) # shuffle data
    test_data.iloc[:, :-1] = test_data.iloc[:, :-1].replace(-3000, replace_unvoiced_val) # replace unvoiced frame value
    test_data.to_csv(checkPath(testDest), index=False)    # save file

def write_data_hard_split(trainDest=os.path.join(seqsFolder,'train_hard.csv'), testDest=os.path.join(seqsFolder,'test_hard.csv'), summary_file=os.path.join(seqsFolder, 'summary.csv'), raag_to_idx_file=os.path.join(seqsFolder,'raag_to_id.txt'), seq_len=3000):
    '''
    Loads all npz arrays and splits them into 3 train and test csv files based on Jin's hard split

    Parameters
        trainDest (str): file path to store train file at (index is added at the end of the file name)
        testDest (str): file path to store test file at (index is added at the end of the file name)
        summary_file (str): file path with metadata on the numpy arrays stored
        raag_to_idx_file (str): stores a mapping to raga name to the number used in the data to represent it; in None, no mapping will be stored
        seq_len (int): Length of each sequence
    
    Returns
        None
    '''
     # create dictionaries to store train and test data in
    data = {f'x_{i}': [] for i in range(seq_len)}
    data['y'] = []
    data['Singer'] = []

    # load summary file and replace raga labels with numbers
    summary = pd.read_csv(summary_file)
    if not os.path.isfile(raag_to_idx_file):
        # raag_idex_file already exists then use the mapping from that file
        summary['Raag'] = summary['Raag'].astype('category')
        raag_to_idx = dict(enumerate(summary['Raag'].cat.categories))
        if raag_to_idx_file is not None:
            with open(raag_to_idx_file, 'w') as f:
                f.write('Raag\tIndex\n')
                for key, value in raag_to_idx.items():
                    f.write(f'{key}\t{value}\n')
    if os.path.isfile(raag_to_idx_file):
        raag_to_idx_df = pd.read_csv(raag_to_idx_file, sep='\t', header=None)
        raag_to_idx = dict(zip(raag_to_idx_df.iloc[:, 0], raag_to_idx_df.iloc[:, 1]))
        summary['Raag'].replace(list(raag_to_idx.values()), list(raag_to_idx.keys()), inplace=True)

    # load arrays
    for filename, df in summary.groupby('Filename'):
        vals = np.load(filename)
        raga = df['Raag'].values[0]     # all raga values should be the same so just pick the first one
        singer = df['Singer'].values[0]     # all singers should be the same so just pick the first one
        # filename belongs to the test set
        for x in vals['arr_0']:
            for arri, xval in enumerate(x):
                data[f'x_{arri}'].append(xval)
            data['y'].append(raga)
            data['Singer'].append(singer)
    # shuffle data and replace unvoiced frames values
    data = pd.DataFrame(data)
    data = data.sample(frac=1, random_state=42) # shuffle data
    data.iloc[:, :-1] = data.iloc[:, :-1].replace(-3000, replace_unvoiced_val) 
    # save files based on singers
    for singer, _ in data.groupby('Singer'):
        data.loc[data['Singer'] != singer, data.columns != 'Singer'].to_csv(trainDest.rsplit('.', 1)[0] + f'-{singer}.csv', index=False)
        data.loc[data['Singer'] == singer, data.columns != 'Singer'].to_csv(testDest.rsplit('.', 1)[0] + f'-{singer}.csv', index=False)
write_data_easy_split_1(seq_len=1200)