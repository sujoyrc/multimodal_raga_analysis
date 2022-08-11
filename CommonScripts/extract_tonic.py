import essentia.standard
import os
from common_utils import addBack, checkPath
import sys
import pdb

srcFolder = '/home/nithya/Projects/OSF Visualisations/onsetExample/New-data-for-pr/OrigAudio/'
tonicFolder = '/home/nithya/Projects/OSF Visualisations/onsetExample/New-data-for-pr/tonics/'

def extract_tonic(src, dest):
    '''
    Extracts tonic using essentia library

    Parameters
        src: file path to audio file
        dest: file path to store tonic in
    '''
    loader = essentia.standard.MonoLoader(filename=src)
    audio = loader()
    tonic = essentia.standard.TonicIndianArtMusic(maxTonicFrequency=530)(audio)
    with open(dest, 'w') as f:
        f.write(str(tonic))
    del audio

def process(srcFolder, tonicFolder=None):
    '''Parse through files a folder. All processed files will be placed in the same folder
    
    Parameters
        srcFolder (str): folder with audio file
        tonicFolder (str): folder with tonic files; if None, assumed to be in the srcFolder itself
       
    Returns
        None
    '''        
    for root, _, fileNames in os.walk(os.path.join(srcFolder)):
        for fileName in fileNames:
            if fileName.endswith('mp3') or fileName.endswith('wav') or fileName.endswith('mp4'):
                srcFile = os.path.join(root, fileName)
                if tonicFolder is None:
                    tonicFile = checkPath(os.path.join(root, fileName).rsplit('.', 1) [0] + '.tonic')
                else:
                    tonicFile = checkPath(os.path.join(root, fileName).replace(addBack(srcFolder), addBack(tonicFolder)).rsplit('.', 1)[0] + '.tonic')
                if os.path.isfile(tonicFile):
                    print('Skipping ' + os.path.join(fileName) + '. Already exists.')
                    continue
                else:
                    print('Processing ' + os.path.join(fileName))
                    extract_tonic(srcFile, tonicFile)

process(srcFolder, tonicFolder)