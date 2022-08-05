import sys
sys.path.append('Kamini_Code/')
import numpy as np
import os
import yaml
import pdb
from voicingContoursfunc import voicingContours
from energyContoursfunc import energyContours

inputFolder = "../Data/"
configFolder = 'Kamini_Code/config.yaml'

def convert_to_wav(inputFile, outputFile):
    '''
    Function to convert mp4 file to wav

    Paramters
        inputFile (str): file path to mp4 file
        outputFile (str): file path to output file
    '''
    os.system(f"ffmpeg -y -i {inputFile} -ar 16000 -ac 1 {outputFile}")

# # convert all original mp4 files to wav
# for root, _, fileNames in os.walk(inputFolder):
#     for fileName in fileNames:
#         if fileName.endswith('.mp4'):
#             mp4File = os.path.join(root, fileName)
#             wavFile = os.path.join(root, fileName.replace('.mp4', '.wav'))
#             convert_to_wav(mp4File, wavFile)

with open(configFolder, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
config['segflag']= '1'
# extract intensity from orig and SS files
for root, _, fileNames in os.walk(inputFolder):
    for fileName in fileNames:
        if fileName.endswith('.wav'):
            if fileName.endswith('-pitch.wav'):
                # skip in the case of pitch resynthesised file
                continue
            audioPath = os.path.join(root, fileName)
            print(audioPath)
            energyContours(audioPath, os.path.join(root, fileName).rsplit('/', 1)[0] + '/')
            # voicingContours(audioPath, config['modelfolder'], os.path.join(root, fileName).rsplit('/', 1)[0] + '/')