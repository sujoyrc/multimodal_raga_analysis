import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import sys
sys.path.append('../../CommonScripts/')
from utils import checkPath, addBack
import pdb

srcFolder = '../PitchInter/'
destFolder = '../Pitch-Normalized/'

def standardize(srcFile, destFile, orig_unvoiced_val=-3000, new_unvoiced_val=-550, scalerType='standard'):
    '''
    Function that replaces the unvoiced frame with a relevant value and standardizes the pitch contour to have mean 0 and std. dev. 1

    Parameters
        srcFile (str): file path to src of pitch contour
        destFile (str): file path to dest of standardized pitch contour
        orig_unvoiced_val (int): original value used to represent unvoiced frames
        new_unvoiced_val (int): new value used to represent unvoiced frames
        scalerType (str): type of scaler to use; can be `standard` or `minmax`
    
    Returns
        None
    '''
    tpe = pd.read_csv(srcFile)
    tpe.loc[:, 'pitch'].replace(orig_unvoiced_val, new_unvoiced_val, inplace=True)    # replaced unvoiced frame

    # normalize values
    if scalerType == 'standard':
        scaler = StandardScaler()
    elif scalerType == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise Exception('Invalid scalerType')
    tpe.loc[:, 'pitch'] = scaler.fit_transform(tpe.loc[:, 'pitch'].values.reshape(-1, 1))
    tpe.loc[:, 'energy'] = scaler.fit_transform(tpe.loc[:, 'energy'].values.reshape(-1, 1))

    # save the file
    tpe.to_csv(destFile, index=False)

def file_parse(srcFolder, destFolder, scalerType='standard'):
    '''
    Parses through files in srcFolder, and standardizes the values

    Parameters
        srcFolder (str): file path to src folder with csv files of pitch contours
        destFolder (str): file path to store new tpes in
        scalerType (str): type of scaling performed; passed as a parameter to standardize function
    
    Returns 
        None
    '''
    for root, _, fileNames in os.walk(srcFolder):
        for fileName in fileNames:
            if fileName.endswith('.csv'):
                # check if the file is a pitch contour file
                srcFile = os.path.join(root, fileName)
                destFile = checkPath(os.path.join(root.replace(addBack(srcFolder), addBack(destFolder)), fileName))
                standardize(srcFile, destFile, scalerType=scalerType)

file_parse(srcFolder, destFolder, 'minmax')