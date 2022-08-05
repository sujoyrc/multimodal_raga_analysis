import warnings
import pandas as pd
import math
import random
import numpy as np
import os
import sys
sys.path.append('../../CommonScripts/')
from common_utils import addBack, checkPath
import pdb

srcFolder = '../PitchInter-New/'
destFolder = '../Seqs_Jin-Normal/'

time_step=0.01  # time step each frame corresponds to
voice_thresh=0   # minimum voiced fraction for a subsequence to be considered

def change_unvoiced_frame_val(pitch_df, orig_unvoiced_val=-3000, new_unvoiced_val=-550):
    '''
    Given a tpe dataframe, this function replaced the value of an unvoiced frame

    Parameters
        pitch_df (pd.Dataframe): tpe dataframe
        orig_unvoiced_val (int): value used to represent unvoiced frames in pitch_df
        new_unvoiced_val (int): new value to use to represent unvoiced frames

    Return
        pitch_df (pd.Dataframe): tpe with replaced values for unvoiced frames
    '''
    pitch_df.loc[:, 'pitch'].replace(orig_unvoiced_val, new_unvoiced_val, inplace=True)

    return pitch_df

def get_uniform_subsequence_times(pitch_df, subseq_len, unvoiced_frame_val=-3000, seq_hop=160, random_hop=(-80, 80)):
    '''
    Given a pitch contour, this returns a list of (start time, end time) of sequences that have a len = subseq_len. The sequences are uniformly sampled from song. Sequences can be overlapping, overlap is controlled by the seq_hop value

    Parameters
        pitch_df (pd.DataFrame): tpe dataframe of song
        subseq_len (int): max length of subseq expected
        unvoiced_frame_val (int): value used to depict unvoiced frames
        seq_hop (int): number of frames to hop before extracting the next sequence
        random_hop (None, int, (int, int)): if random_hop is None, no random value is added to the seq_hop; if random_hop is int, it indicates getting a random number of frames between (0, int) between each hop; if (int, int) indicates minimum and maximum limits of random frames to add between strides

    Returns
        seqs (list): List of tuples-> (start time, end time) for each subsequence to be considered
    '''
    
    seqs_times = []     # array to add (start time, end time)
    if random_hop is None:
        # in the case that no random hop is added
        for ind in range(0, pitch_df.shape[0]-subseq_len, seq_hop):
            seqs_times.append([pitch_df.iloc[ind, 0], pitch_df.iloc[ind + subseq_len-1, 0]])
    else: 
        # random hop is given
        if isinstance(random_hop, int):
            # in the case that random hop is just an integer and not a range
            random_hop = (0, random_hop) 
        ind = np.abs(random.randint(random_hop[0], random_hop[1]))  # generate a random index to start from and make sure the ind is positive
        while ind < pitch_df.shape[0] - subseq_len - 1:
            seqs_times.append([pitch_df.iloc[ind, 0], pitch_df.iloc[ind + subseq_len - 1, 0]])
            ind += seq_hop + random.randint(random_hop[0], random_hop[1])   # update the random index value
    if len(seqs_times) == 0:
        # if there are no values in seqs_times, then keep the whole song. This is kept for the edge case where the song ends at 11.99, and wouldn't be considered otherwise. To maintain, same sequences across all input representations
        seqs_times.append([pitch_df.iloc[0, 0], min(pitch_df.iloc[-1, 0], (subseq_len - 1)*time_step)])
    return seqs_times

def get_bp_subsequence_times(pitch_df, subseq_len, unvoiced_frame_val=-3000, bp_thresh=50, bp_hop=0):
    '''
    Given a pitch contour, this returns a list of (start time, end time) of sequences of breath phrases that have a len < subseq_len. 

    Parameters
        pitch_df (pd.DataFrame): tpe dataframe of song
        subseq_len (int): max length of subseq expected
        unvoiced_frame_val (int): value used to depict unvoiced frames
        bp_thresh (int): minimum number of unvoiced frames to consider a breath pause
        bp_hop (int): number of breath phrases to hop to generate new sequences; has to be a positive integer

    Returns
        seqs (list): List of tuples-> (start time, end time) for each subsequence to be considered
    '''
    
    # create a dataframe with time, pitch, duration and end time columns with groups for repeated occurence of the same pitch value
    group_pitches = pitch_df.iloc[(np.diff(pitch_df['pitch'].values, prepend=np.nan) != 0).nonzero()][['time', 'pitch']].copy()
    group_pitches['duration'] = np.diff(group_pitches['time'], append=(pitch_df.iloc[-1, 0]+0.1))
    group_pitches['end time'] = group_pitches['time'] + group_pitches['duration']

    # dataframe of breath pauses 
    bps = group_pitches.loc[(group_pitches['pitch'] == unvoiced_frame_val) & (group_pitches['duration'] >= bp_thresh * time_step)].reset_index(drop=True)
    
    # create a dataframe with both breath pauses and breath phrases. Columns are - start time, end time, duration, type 
    phrases_df = {
        'start time': [],
        'end time': [],
        'duration': [],
        'type': []
    }
    for i, row in list(bps.iterrows())[:-1]:
        # breath phrase
        phrases_df['start time'].append(row['time'])
        phrases_df['end time'].append(row['end time'])
        phrases_df['duration'].append(row['duration'])
        phrases_df['type'].append('BPause')
        # singing phrase
        phrases_df['start time'].append(row['end time'])
        phrases_df['end time'].append(bps.iloc[i+1, 0])
        phrases_df['duration'].append(phrases_df['end time'][-1] - phrases_df['start time'][-1])
        phrases_df['type'].append('BPhrase')
    phrases_df = pd.DataFrame(phrases_df)

    # extract start and end time stamps
    bp_hop_count = np.random.randint(bp_hop+1)   # number of breath phrases skipped already; used to check that bp_hop phrases are skipped; initialised randomly
    seqs_times = []
    for i, row in phrases_df.iterrows():
        if row['type'] == 'BPhrase':
            if bp_hop_count == bp_hop:
                # bp_hop breath phrases have been skipped
                bp_hop_count = 0    # reset bp_hop_count
                if row['duration'] >= subseq_len*time_step:
                    # only one breath phrase
                    seqs_times.append([row['start time'], ((subseq_len-1)*time_step)+row['start time']])
                else:
                    subset = phrases_df.loc[(phrases_df['start time'] >= row['start time']) & (phrases_df['end time'] < np.around((row['start time']+(subseq_len*time_step))/time_step)*time_step)]
                    seqs_times.append([row['start time'], subset.iloc[-1, 1]])
            else:
                bp_hop_count += 1

    return seqs_times

def get_voiced_frame_len(pitch_df, start_time, end_time, unvoiced_frame_val=-3000):
    '''
    Returns the number of voiced frames in the pitch dataframe between start and end time

    Parameters
        pitch_df (pd.DataFrame): dataframe of tpe
        start_time (float): time to start reading pitch values at
        end_time (float): time to stop reading pitch values at

    Returns
        voiced_frames (int): number of voiced frames within start and stop time

    '''
    if unvoiced_frame_val is None:
        unvoiced_frame_val = min(pitch_df['pitch'].values)
    return pitch_df.loc[(pitch_df['time'] >= start_time) & (pitch_df['time'] <= end_time) & (pitch_df['pitch'] != unvoiced_frame_val)].shape[0]

def subsequence_select(src, type, subseq_len=2000, unvoiced_frame_val=-3000, bp_hop=0, seq_hop=160):
        '''
        First standardises values in the pitch contour, this involves replacing the unvoiced frame value as well. Then selects phrase subsequences from contour

        Parameters
            src (str): dataframe of pitch contour or source file path with pitch contour
            type (str): has to be either 'uniform' or 'bp'; to choose the type of subsequence selection
            subseq_len (int): length of subsequence to extract
            unvoiced_frame_val (int): value used to depict unvoiced frames
            bp_hop (int): number of breath phrases to hop to generate new subsequences; has to be positive integer; valid only if get_bp_subsequence_times() is called
        
        Returns
            subsequences (np.array): a list of subsequences
            info_arr (list): list of metadata for each subsequence including - [index, start time, end time, percentage of voiced frames]
        
        '''
        
        if isinstance(src, str):
            df = pd.read_csv(src)
        else:
            df = src
        info_arr = []  # maintains index, start point, percentage of voiced frames in each subsequence
        subsequences = []   # maintains a list of subsequences selected
        if type == 'uniform':
            seqs_time = get_uniform_subsequence_times(df, subseq_len, seq_hop=seq_hop)
        elif type == 'bp':
            seqs_time = get_bp_subsequence_times(df, subseq_len, bp_hop=bp_hop)
        else:
            raise(f'Type {type} is invalid')
        ind=0   # subsequence index
        for _, (start_time, end_time) in enumerate(seqs_time):
            # pdb.set_trace()
            voiced_frac = np.around(get_voiced_frame_len(df, start_time, end_time, unvoiced_frame_val=unvoiced_frame_val)/subseq_len, 2)
            if voiced_frac < voice_thresh:
                # skip the sequence if voiced_frac is less than threshold value
                continue
            unpadded_seq = df.loc[(df['time'] >= start_time) & (df['time'] <= end_time), 'pitch'].values
            if len(unpadded_seq) == subseq_len:
                # consider only subsequences that are as long as subsequence length
                subsequences.append(unpadded_seq)
            # subsequences.append(np.pad(unpadded_seq, (0, subseq_len-len(unpadded_seq)), mode='constant', constant_values=unvoiced_frame_val))
                info_arr.append([ind, start_time, end_time, voiced_frac])
                ind += 1
        return np.array(subsequences), info_arr

def parse_files(srcFolder, destFolder, type, subseq_len, seq_hop, orig_unvoiced_val=-3000, new_unvoiced_val=-550):
        '''Generate subsequences to use for the model
        
        Parameters
            srcFolder (str): File path to the source folder with the pitch contours
            destFolder (str): Folder to store subsequences in
            type (str): has to be either 'uniform' or 'bp'; to choose the type of subsequence selection
            subseq_len (int): number of frames in each sequence
            seq_hop (int): number of frames to skip between sequences extracted; used for uniform extraction
            orig_unvoiced_val (int): value used to represent unvoiced frames; if None will be considered as the minimum value in the pitch contour
            new_unvoiced_val (int): if not None, will be used to replace orig_unvoiced_val in the subsequences extracted
        '''
        random.seed(42)
        info_arrs = []  # maintains a list of arrays with metadata for each subsequence
        
        for root, _, fileNames in os.walk(os.path.join(srcFolder)):
            for fileName in fileNames:
                if fileName.endswith('-pitch.csv'):
                    # consider only pitch files
                    dataDest = checkPath(os.path.join(destFolder, root.replace(addBack(srcFolder), ''), fileName.rsplit('-', 1)[0] + '.npz'))

                    print(f'Processing {os.path.join(root, fileName)}')
                    if 'SCh' in fileName and 'Alap' in root:
                        # add a breath phrase hop only for SCh's Alaps
                        bp_hop=1
                    else:
                        bp_hop=0
                    
                    # load tpe
                    tpe = pd.read_csv(os.path.join(root, fileName))
                    if new_unvoiced_val is not None:
                        tpe = change_unvoiced_frame_val(tpe, orig_unvoiced_val=orig_unvoiced_val, new_unvoiced_val=new_unvoiced_val)
                    seqs, info_arr = subsequence_select(tpe, type=type, subseq_len=subseq_len, bp_hop=bp_hop, seq_hop=seq_hop, unvoiced_frame_val=new_unvoiced_val if new_unvoiced_val is not None else orig_unvoiced_val)
                    
                    # if seqs is empty print a warning
                    if len(seqs) == 0:
                        warnings.warn(f'No sequences extracted from {fileName}')
                    # save the data to files
                    np.savez_compressed(dataDest, seqs)
                    # save metadata info
                    info_arrs.extend([[dataDest, fileName.split('_', 1)[0], fileName.rsplit('_', 1)[1].rsplit('-', 1)[0], info_arr_vals[0], info_arr_vals[1], info_arr_vals[2], info_arr_vals[3]] for info_arr_vals in info_arr])
                    del seqs
        metadata = pd.DataFrame(info_arrs, columns=['Filename', 'Singer', 'Raag', 'Subsequence Index', 'Start Time', 'End Time', 'Fraction of voiced frames'])
        metadata.to_csv(os.path.join(destFolder, 'summary.csv'), index=False)
# parse_files(srcFolder, destFolder, type='uniform', subseq_len=1200, seq_hop=160, orig_unvoiced_val=-3000, new_unvoiced_val=-550)