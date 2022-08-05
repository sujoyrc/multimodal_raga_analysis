import os
import sys
from shutil import copy2
sys.path.append('../../CommonScripts/')
from utils import checkPath

data_folder = '../Data/'
pitch_folder = '../Pitch/PitchInter-New/'
new_pitch_folder = '../Pitch/OrigPitch/'

# for root, _, fileNames in os.walk(data_folder):
#     for fileName in fileNames:
#         if fileName.endswith('-pitch.csv'):
#             os.rename(os.path.join(root, fileName), checkPath(os.path.join(root, fileName).replace(data_folder, new_pitch_folder)))

# copy interpolated files to data folder
# for root, _, fileNames in os.walk(pitch_folder):
#     for fileName in fileNames:
#         if fileName.endswith('-pitch.csv'):
#             os.rename(os.path.join(root, fileName), checkPath(os.path.join(root, fileName).replace(pitch_folder, data_folder)))

# copy pitch inter files into pitch_folder
for root, _, fileNames in os.walk(data_folder):
    for fileName in fileNames:
        if fileName.endswith('-pitch.csv'):
            copy2(os.path.join(root, fileName), checkPath(os.path.join(root, fileName).replace(data_folder, pitch_folder)))