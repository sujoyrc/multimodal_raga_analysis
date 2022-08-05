'''
This script is used to resample the pitch data from 1200 samples to 600 samples in 12 s. This is done by dropping alternate samples.

It also resamples video data from 300 samples to 600 samples using scipy.signal.resample.
'''

import numpy as np
import pandas as pd
from scipy.signal import resample 
import os
import sys
sys.path.append('../../CommonScripts/')
from common_utils import checkPath

import pdb

audioFolder = '../Data/'
videoFolder = '../Final Video Data/Normalized/'
resampledFolder = '../Final Both Data/'

def resample_audio(src_file, dest_file):
    '''
    Resamples the audio by dropping alternate samples

    Parameters
    ----------
    src_file    : str
        Location of source pitch file
    dest_file   : str
        Location of dest pitch file
    '''
    pitch_df = pd.read_csv(src_file)
    pitch_df['mask'] = (pitch_df.loc[:, 'pitch'].values != -3000).astype(int)
    new_pitch_df = pitch_df.iloc[np.arange(0, pitch_df.shape[0], 2), :]
    new_pitch_df.to_csv(dest_file, columns=pitch_df.columns, index=False)

def resample_video(src_file, dest_file):
    '''
    Resamples video data with fourier method

    Parameters
    ----------
    src_file    : str
        Location of source pitch file
    dest_file   : str
        Location of dest pitch file
    '''

    video_df = pd.read_csv(src_file, header=[0, 1])
    new_video_dict = {}

    for cols in video_df.columns:
        if cols[1] == 'c':
            continue
        elif cols[1] == 'Variable':
            new_video_dict[cols] = np.arange(0, int(video_df.shape[0]*2))
        else:
            new_video_dict[cols] = np.around(resample(video_df[cols], num=int(video_df.shape[0]*2)), 2)
    
    pd.DataFrame(new_video_dict).to_csv(dest_file, index=False)

def main():
    for root, _, fileNames in os.walk(audioFolder):
        for fileName in fileNames:
            if fileName.endswith('-pitch.csv'):
                srcFile = os.path.join(root, fileName)
                destFile = checkPath(os.path.join(root.rsplit('/', 1)[0].replace(audioFolder, os.path.join(resampledFolder, 'Audio/')), fileName.rsplit('-', 1)[0] + '.csv'))
                resample_audio(srcFile, destFile)
    
    for root, _, fileNames in os.walk(videoFolder):
        for fileName in fileNames:
            srcFile = os.path.join(root, fileName)
            destFile = checkPath(os.path.join(root.replace(videoFolder, os.path.join(resampledFolder, 'Video/')), fileName))
            resample_video(srcFile, destFile)

main()