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

dataFolder = '../Data/'
seqsFolder = '../Seqs/'
csvFolder = '../Seqs/csvs/'
# splitFolder = '../Seqs/splits/easy_1/'
# finalData = '../Seqs/finalData/band1vaishaliInten/easy_1/'

# csvFolder = '../Final Video Data/Sequences/'
# splitFolder = '../Seqs/splits/hard_2/'
# finalData = '../Final Video Data/finalData/4_channels/hard_2'
summaryFile = '../Seqs/summary.csv'

# all
# file_extensions = ['-pitch', '_voicing', '_voicing', '_voicing', '_voicing', '_voicing', '_voicing', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others']    # file extension used to create csvs
# keywords = ['pitch', 'zeroCross', 'autoCorr', 'LPC1', 	'error', 'bandRatio', 'HNR', 'energy', 'intensity',	'sonorantEnergy', 'sonorantIntensity', 'band2to20Energy', 'band2to20Intensity', 'band1Intensity', 'band2Intensity', 'band3Intensity', 'band4Intensity', 'band1vaishaliInten', 'band2vaishaliInten', 'band3vaishaliInten', 'band4vaishaliInten', 'spectralTilt'
# ]   # keywords used to create csvs

# # others
# file_extensions = ['-pitch', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others', '_others']    # file extension used to create csvs
# keywords = ['pitch', 'energy', 'intensity',	'sonorantEnergy', 'sonorantIntensity', 'band2to20Energy', 'band2to20Intensity', 'band1Intensity', 'band2Intensity', 'band3Intensity', 'band4Intensity', 'band1vaishaliInten', 'band2vaishaliInten', 'band3vaishaliInten', 'band4vaishaliInten', 'spectralTilt'
# ]   # keywords used to create csvs

# # voicing
# file_extensions = ['-pitch', '_voicing', '_voicing', '_voicing', '_voicing', '_voicing', '_voicing']    # file extension used to create csvs
# keywords = ['pitch', 'zeroCross', 'autoCorr', 'LPC1', 	'error', 'bandRatio', 'HNR']   # keywords used to create csvs

# pitch
file_extensions = ['-pitch']
keywords = ['pitch']

# # video data
# file_extensions = ['-RWrist', '-RWrist', '-LWrist', '-LWrist']
# keywords = ['x', 'y', 'x', 'y']
random.seed(42)

def create_summary(folder, dest):
    '''
    Creates a summary file with filename, raag, unique subsequence id, start time, end time

    Parameters
        folder (str): input folder with pitch files
        dest (str): filename of destination summary
    '''

    res = {
        'filename': [],
        'singer': [],
        'raga': [],
        'group': [],    # Alap or Pakad
        'unique_id': [],
        'start_times': [],
        'end_times': []
    }
    for root, _, fileNames in os.walk(folder):
        for fileName in fileNames:
            if fileName.endswith('-pitch.csv'):
                singer, _, raga = fileName.rsplit('-', 1)[0].rsplit('_')
                group = 'Alap' if 'Alap' in root else 'Pakad'
                ids, start_times, end_times = get_subseq_times(pitch_filename=os.path.join(root, fileName))
                
                # update res
                res['filename'].extend(np.full(len(ids), os.path.join(root, fileName)))
                res['singer'].extend(np.full(len(ids), singer))
                res['raga'].extend(np.full(len(ids), raga))
                res['group'].extend(np.full(len(ids), group))
                res['unique_id'].extend(ids)
                res['start_times'].extend(start_times)
                res['end_times'].extend(end_times)

    df = pd.DataFrame(res)
    df = df.set_index('unique_id')
    df.to_csv(checkPath(dest))

def create_csv(file_extensions, keywords, summaryFile, dataFolder, destFolder, subseq_len=1200, class_index_file=None, replace=False, mask=True):
    '''
    Creates csv files for each feature based on summary csv created from create_summary()

    Parameters
        file_extensions (list): list of file extensions
        keywords (list): column name to extract values from; each value corresponds to a value in the file_extensions list
        summaryFile (str): path to summary file
        dataFolder (str): path to folder with data csvs
        destFolder (str): folder to store feature csvs at
        subseq_len (int): length of each subsequence
        class_index_file (str): path to csv file to map raga labels to integer values. If None, new map is created; if file path doesn't exist, new map is created and stored at the path
    '''

    # convert raga class to integers
    mapping = label_index_map(summaryFile=summaryFile, class_index_file=class_index_file)
    summary = pd.read_csv(summaryFile)
    summary['start_times'] = summary['start_times'].round(decimals=2)   # round off start_times to avoid errors
    summary['end_times'] = summary['end_times'].round(decimals=2)   # round off end_times to avoid errors
    cols = [f'x_{i}' for i in range(subseq_len)]
    cols.append('y')
    cols.append('unique_id')
    if mask:
        mask_vals = []
        if '-pitch' not in file_extensions:
            # add pitch to file extensions list
            file_extensions.append('-pitch')
            keywords.append('pitch')
            write_pitch = False     # indicates whether to write pitch values
        else:
            write_pitch = True

    for ind, file_extension in enumerate(file_extensions):
        print('Creating file for ' + file_extension + ' ' + keywords[ind])
        if os.path.exists(os.path.join(destFolder, file_extension[1:] + '-' + keywords[ind] + '.csv')) and not replace:
            print(f'{file_extension} {keywords[ind]} already exists')
            # continue
        data_vals = []  # values to populate dataframe
        for root, _, fileNames in os.walk(dataFolder):
            for fileName in fileNames:
                if fileName.endswith(file_extension + '.csv'):
                    if file_extension == '_voicing' or file_extension == '_others':
                        # if the file extension is just voicing.csv, don't consider the SS voicing file
                        if fileName.endswith(f'SS{file_extension}.csv'):
                            continue
                    fileVals = pd.read_csv(os.path.join(root, fileName))
                    fileVals['time'] = fileVals['time'].round(decimals=2)   # round time to 2 decimal places to avoid errors while searching for subsequence values based on time
                    fileVals[keywords[ind]].fillna(fileVals[keywords[ind]].min(skipna=True), inplace=True)  # replace nan values with the minimum non-nan value in the series
                    for _, row in summary.loc[summary['filename'] == os.path.join(root, fileName).replace(file_extension, '-pitch')].iterrows():
                        temp_vals = fileVals.loc[(fileVals['time'] >= row['start_times']) & (fileVals['time'] <= row['end_times']), keywords[ind]].values
                        if file_extension == '-pitch' and keywords[ind] == 'pitch':
                            # for pitch vals, replace unvoiced frame value from -3000 to -550
                            # pdb.set_trace()
                            temp_vals = replace_unvoiced(temp_vals, -3000, -550)
                            # add pitch bool also
                            if mask:
                                mask_val = (temp_vals.astype(int) != -550).astype(int)
                                mask_vals.append(np.append(mask_val, [mapping[row['raga']], row['unique_id']]))
                            if not write_pitch:
                                # if not supposed to write pitch, go to next iteration; it is just collecting mask values in this case
                                continue
                        temp_vals = np.append(temp_vals, [mapping[row['raga']], row['unique_id']])
                        data_vals.append(temp_vals)
        data_df = pd.DataFrame(data_vals, columns=cols)     # dataframe for values
        # data_df = data_df.set_index('unique_id')
        data_df.to_csv(checkPath(os.path.join(destFolder, file_extension[1:] + '-' + keywords[ind] + '.csv')), index=False)
        # pdb.set_trace()
        if mask:
            mask_df = pd.DataFrame(mask_vals, columns=cols)
            mask_df.to_csv(checkPath(os.path.join(destFolder, 'mask.csv')), index=False)


    
def train_test_split(splitFolder, csvFolder, destFolder, subseq_len=1200, normalize=True, mask=True):
    '''
    splits data into train and test npz files.

    Parameters
        splitFolder (str): folder path with splits (list of files to include in test set)
        csvFolder (str): folder path to csvs with feature values
        destFolder (str): folder to store train and test npz files at
        subseq_len (int): subsequence length
        normalize (bool): if True, will save a normalized version of the data
        mask (bool): if True, will add the mask array to the npz files
    '''
    csvFiles = [pd.read_csv(os.path.join(csvFolder, file_extensions[ind][1:] + '-' + keywords[ind] + '.csv')) for ind in range(len(file_extensions))]   # sort the csv files alphabetically so that the order of features remains the same
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

        # if mask is true, load the mask also
        if mask:
            # pdb.set_trace()
            mask_df = pd.read_csv(os.path.join(csvFolder, 'mask.csv'))
            mask_train = []
            mask_test = []
            for unique_id in csvFiles[0]['unique_id'].values:
                temp_mask = mask_df.loc[mask_df['unique_id'] == unique_id, xCols].values[0]
                if unique_id.rsplit('_', 1)[0] in test_files:
                    # test data
                    mask_test.append(temp_mask)
                else:
                    # train data
                   mask_train.append(temp_mask)
            # reshape mask to (x, 1200, 1) from (x, 1200)
            mask_test = np.reshape(mask_test, (-1, subseq_len, 1))
            mask_train = np.reshape(mask_train, (-1, subseq_len, 1))
        else:
            mask_train = None
            mask_test = None

        # pdb.set_trace()
        splitFileNames = fileName.rsplit("-", 1)
        if len(splitFileNames) > 1:
            destFilename = os.path.join(finalData, f'{fileName.rsplit("-", 1)[1].rsplit(".", 1)[0]}-orig.npz')
        else:
            destFilename = os.path.join(finalData, fileName.rsplit('.', 1)[0] + '-orig.npz')
        np.savez(checkPath(destFilename), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, train_ids=train_ids, test_ids=test_ids, mask_train=mask_train, mask_test=mask_test, channels=np.array([file_extensions[i][1:] + '-' + keywords[i] for i in range(len(keywords))]))

        if normalize:
            # set up empty X_train and X_test arrays
            X_train_norm = np.empty(X_train.shape)
            X_test_norm = np.empty(X_test.shape)
            # decide min and max value in the train data set
            min_val = np.min(X_train, axis=(0, 1))
            max_val = np.max(X_train, axis=(0, 1))
            
            # transform train, test data based on min and max value for each axis separately
            for ind in range(min_val.shape[0]):
                X_train_norm[:, :, ind] = (X_train[:, :, ind] - min_val[ind])/(max_val[ind]-min_val[ind])
                X_test_norm[:, :, ind] = (X_test[:, :, ind] - min_val[ind])/(max_val[ind]-min_val[ind])

            # save file
            np.savez(checkPath(destFilename.replace('-orig.npz', '-norm.npz')), X_train=X_train_norm, X_test=X_test_norm, y_train=y_train, y_test=y_test, train_ids=train_ids, test_ids=test_ids, mask_train=mask_train, mask_test=mask_test, channels=np.array([file_extensions[i][1:] + '-' + keywords[i] for i in range(len(keywords))]))



def main():
    # create_summary(dataFolder, os.path.join(seqsFolder, 'summary.csv'))
    # print('Creating csvs')
    create_csv(file_extensions, keywords, summaryFile, dataFolder, csvFolder, subseq_len=1200, class_index_file=None, replace=True, mask=True)
    # print('Splitting data')
    # train_test_split(splitFolder=splitFolder, csvFolder=csvFolder, destFolder=finalData, subseq_len=300, mask=False, normalize=False)

if __name__=="__main__":
    main()