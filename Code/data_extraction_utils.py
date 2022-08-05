import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from extract_seqs import get_uniform_subsequence_times
import pdb

def get_subseq_times(pitch_filename, subseq_len=1200, seq_hop=160, random_hop=(-80, 80), time_step=0.01):
    '''
    Given a pitch contour file extracts unique index, start time, end time which is used for further subsequence extraction.

    Parameters:
        pitch_filename (str): filename of tpe file
        subseq_len (int): max length of subseq expected
        unvoiced_frame_val (int): value used to depict unvoiced frames
        seq_hop (int): number of frames to hop before extracting the next sequence
        random_hop (None, int, (int, int)): if random_hop is None, no random value is added to the seq_hop; if random_hop is int, it indicates getting a random number of frames between (0, int) between each hop; if (int, int) indicates minimum and maximum limits of random frames to add between strides
    '''

    if isinstance(pitch_filename, str):
        pitch = pd.read_csv(pitch_filename)
    else:
        raise Exception('Pitch file name is incorrect')
    timestamps = get_uniform_subsequence_times(pitch_df=pitch, subseq_len=subseq_len, seq_hop=seq_hop, random_hop=random_hop)

    filename = pitch_filename.rsplit('/', 1)[1].rsplit('-', 1)[0]

    # arrays to store result values in 
    ids = []
    start_times = []
    end_times = []
    idx = 0     # unique index of subsequence
    for timestamp in timestamps:
        end_time = timestamp[1]
        start_time = timestamp[0]
        if np.round(((end_time + time_step) - start_time)/time_step) == subseq_len:
            # check if end_time - start_time = subseq_len
            ids.append(filename + '_' + str(idx))
            idx += 1
            start_times.append(start_time)
            end_times.append(end_time)
    
    return ids, start_times, end_times

def label_index_map(summaryFile, class_index_file=None):
    '''
    Creates a map from class values to index

    Parameters
        summaryFile (str): path to summary file
        class_index_file (str): path to csv file to map raga labels to integer values. If None, new map is created; if file path doesn't exist, new map is created and stored at the path

    Returns
        mapping (dict): dictionary object with mapping
    '''
    if class_index_file is not None and os.path.isfile(class_index_file):
        class_mapping = pd.read_csv(class_index_file)
        mapping = {}
        for _, row in class_mapping.iterrows():
            mapping[row['raga']] = row['index']
    else:
        summary = pd.read_csv(summaryFile)
        le = LabelEncoder()
        le.fit(summary['raga'].values)
        mapping = {x: le.transform([x])[0] for x in le.classes_}
        if class_index_file is not None:
            class_mapping = pd.DataFrame({'raga': list(mapping.keys()), 'index': list(mapping.values())})
            class_mapping.to_csv(class_index_file, index=False)
    return mapping

def replace_unvoiced(arr, orig, new_value):
    '''
    Function replaces original value with new_value in the array

    Parameters
        arr (np.array): array to replace values in
        orig (float): value in arr to be replaced
        new_value (float): value to replace orig value with in arr

    Returns
        arr (np.array): array with replaced values
    '''
    arr[arr == orig] = new_value
    return arr