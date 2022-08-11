import pandas as pd 
import numpy as np
import os
from common_utils import checkPath

'''This script converts normalised pitch files to tab separated tsvs which are then fed into the pitch resynthesis program'''

# Variables
srcFolder = None
tonicFolder= None
normalised=False
includeFileList=None

def create_tsv(srcFile=None, destFile=None, normalised=False, tonic=None, k=100, unvoiced_val=-3000):
    '''
    This function converts normalised frequency values into Hz values and stores it as a tab-separarated tpe

    Parameters
        srcFile (str): File path to a tpe with normalised pitch contour present as a csv file
        destFile (str): File path to store tab-separated tpe; folder will be created if it doesn't already exist; if None, file won't be stored
        normalised (bool): if True, then the pitch contour is normalised
        tonic (float): Tonic with which the pitch contour was normalised
        k (int): Number of divisions per semitone
        unvoiced_val (int): Number used to depict unvoiced regions
    
    Returns
        None
    '''

    if srcFile is None:
        raise Exception('srcFile has to be provided.')
    if tonic is None and normalised:
        raise Exception('tonic has to be provided if normalised is True')
    if tonic is not None:
        tonic = np.around(tonic, 2) # round tonic to 2 decimal placed, for simplicity of calculation

    # calculate true contour values
    tpe = pd.read_csv(srcFile)
    voiced_idx = tpe.loc[tpe.iloc[:, 1] != unvoiced_val].index    # index of unvoiced samples
    if normalised is True:
        pitches = np.zeros(tpe.shape[0])
        pitches[voiced_idx] = (2**((tpe.iloc[voiced_idx, 1]).astype('float')/(int(k)*12)))*float(tonic)
        tpe.iloc[:, 1] = pitches    # replace pitched with Hz values

    if destFile is not None:
        # store dest file if specified
        checkPath(destFile)
        tpe.to_csv(destFile, sep='\t', header=False, index=False)
    
    
def wine(srcFile=None, destFile=None, wineDir = 'PitchResynthesis_Kaustuv/'):
    '''This script generates resynthesis of pitch contour
    
    Parameters
        srcFile (str): file path of tab separated tpe file
        destFile (str): file path for resynthesised audio
        wineDir (str): folder containing the pitch resysnthesis code
    
    Returns
        None
    '''
    if srcFile is None or destFile is None:
        raise Exception('Both srcFile and destFile have to be provided')
    checkPath(destFile) # check that destFile path exists

    # change directory to Pitch resynthesis directory
    srcFile = os.path.abspath(srcFile)  # get absolute path of srcFile
    destFile = os.path.abspath(destFile)
    cwd = os.getcwd()   # save current working directory
    os.chdir(wineDir)

    # call function
    os.system(f'wine tpe2syn.exe "{srcFile}" "{destFile}" 0 Amp.txt 0')

    # restore working directory
    os.chdir(cwd)

def fileParse(srcFolder, tonicFolder=None, normalised=False, includeFileList=None):
    '''Parses through the folder and generates wav resynthesis files for each tpe
    
    Parameters
        srcFolder (str): file path to csv with tpe
        tonicFolder (str): file path to folder with tonics of the audio clips; if None, does not exist
        normalise
        normalised (bool): if True, assumes pitch contour is normalised (in cents) else assumes pitch contour is in Hz
        includeFileList (list): list of file names to process (in case you don't want to process the whole folder); if None all files in the folder are processed
    
    Returns
        None
    '''
    for root, _, fileNames in os.walk(srcFolder):
        for fileName in fileNames:
            if fileName.endswith('.csv') and (includeFileList is None or os.path.abspath(os.path.join(root, fileName)) in includeFileList) :
                # select only csv tpe files
                print(f'Processing {os.path.join(root, fileName)}')

                # set file locations
                tpeDest = os.path.join(root, fileName.rsplit('.', 1)[0] + '.tpe')
                wineDest = os.path.join(root, fileName.rsplit('.', 1)[0] + '.wav')
                # crete tab-separated tpe
                if not os.path.isfile(wineDest):
                    # check if the wav file already exists
                    if not os.path.isfile(tpeDest):
                        # if tab-separated file doesn't exist
                        if tonicFolder is not None:
                            tonicFile = os.path.join(root.replace(srcFolder, tonicFolder), fileName.rsplit('-', 1)[0] + '.tonic')
                            with open(tonicFile, 'r') as f:
                                tonic = float(f.read())
                        else:
                            tonic = None
                        create_tsv(srcFile=os.path.join(root, fileName), destFile=tpeDest, tonic=tonic, normalised=normalised)
                    wine(srcFile=tpeDest, destFile=wineDest)    

fileParse(srcFolder, tonicFolder=None, normalised=False, includeFileList=None)