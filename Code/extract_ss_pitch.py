import sys
sys.path.append('../../CommonScripts/')
from Archived.extract_pitch_contours import process
from utils import checkPath, addBack
import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import warnings
import pdb
'''
This script generates pitch contours for each audio file. 
'''

def interpolate_gaps(pitch_df, thresh=0.5, kind='cubic', unvoiced_frame_val=-3000):
    '''
    This function interpolates gaps in the pitch contour that are less than thresh s long.
    Parameter
        pitch_df (pd.DataFrame): TPE dataframe for song
        thresh (float): duration (in s) below which the contour will be interpolated
        kind (str): type of interpolation performed, passed as a parameter to scipy.interpolate.interp1d()
        unvoiced_frame_val (int): value used to represent unvoiced frames
    
    Returns
        pitch_df (pd.DataFrame): TPE dataframe with short gaps interpolated
    '''
    group_pitches = pitch_df.iloc[(np.diff(pitch_df['pitch'].values, prepend=np.nan) != 0).nonzero()][['time', 'pitch']].copy()
    group_pitches['duration'] = np.diff(group_pitches['time'], append=(pitch_df.iloc[-1, 0]+0.1))
    group_pitches['end time'] = group_pitches['time'] + group_pitches['duration']
    pitch_vals = pitch_df['pitch'].values
    
    for ind, row in group_pitches.loc[(group_pitches['pitch'] == unvoiced_frame_val) & (group_pitches['duration'] < thresh)].iterrows():
        # pdb.set_trace()
        pitch_subset = pitch_df.loc[(pitch_df['time'] >= row['time']-0.1) & (pitch_df['time'] <= row['end time']+0.1) & (pitch_df['pitch'] != unvoiced_frame_val)]
        # values given to the interpolate function
        x_old = pitch_subset['time']
        y_old = pitch_subset['pitch']
        # interpolate function
        try:
            f = interp1d(x_old, y_old, kind=kind, fill_value="extrapolate", assume_sorted=True)
        except:
            warnings.warn(str(f'Skipping interpolating values between {row["time"]} and {row["end time"]}'))
            continue
        # use function to find pitch values for short gaps
        #pdb.set_trace()
        y_new = f(pitch_df.loc[(pitch_df['time'] >= row['time']) & (pitch_df['time'] <= row['end time']), 'time'].values)
        y_new[y_new <= -550] = -3000    # all values interpolated to values below -550 are set to unvoiced
        y_new[y_new > 1950] = -3000    # all values interpolated to values above 1950 are set to unvoiced
        pitch_vals[pitch_df.loc[(pitch_df['time'] >= row['time']) & (pitch_df['time'] <= row['end time'])].index.values] = y_new
    pitch_df.loc[:, 'pitch'] = pitch_vals
    return pitch_df

def fileParseToInterpolate(srcFolder, destFolder):
    for root, _, fileNames in os.walk(srcFolder):
        for fileName in fileNames:
            if fileName.endswith('-pitch.csv'):
                print(f'Processing {fileName}')
                pitch_df = pd.read_csv(os.path.join(root, fileName))
                new_pitch_df = interpolate_gaps(pitch_df)
                new_pitch_df.to_csv(checkPath(os.path.join(root, fileName).replace(addBack(srcFolder), addBack(destFolder))), index=False)


srcFolder = '../Data/'
ssFolder = None
pitchFolder = None
interpolateFolder = '../PitchInter-New/'
process(srcFolder, ssDestFolder=ssFolder, pitchDestFolder=pitchFolder, normalize=True, k=5)
if pitchFolder is None:
    pitchFolder = srcFolder
fileParseToInterpolate(pitchFolder, interpolateFolder)
