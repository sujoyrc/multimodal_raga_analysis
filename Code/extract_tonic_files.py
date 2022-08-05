import sys
sys.path.append('../../CommonScripts/')
from extract_tonic import process
import os

'''
This script first creates a folder for each file, then extracts the tonic for each file
'''

srcFolder = '../Data/'

def folderise(srcFolder):
    '''Creates a folder for each song

    Parameters
        srcFolder: file path to the source folder
    Returns
        None
    
    '''
    for root, _, fileNames in os.walk(srcFolder):
        for fileName in fileNames:
            # create a folder for each file
            os.makedirs(os.path.join(root, fileName.rsplit('.', 1)[0]))
            os.rename(os.path.join(root, fileName), os.path.join(root, fileName.rsplit('.', 1)[0], fileName))   # move file to new folder

# folderise(srcFolder)
process(srcFolder)