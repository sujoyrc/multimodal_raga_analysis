import os
import sys
import shutil
sys.path.append('../../CommonScripts/')
from common_utils import checkPath
import pdb

orig='/home/nithya/Projects/Gesture Analysis/Data/' 
new='/home/nithya/Projects/Gesture Analysis/Data-ForHemantha/' 
for root, _, fileNames in os.walk(orig):
     for fileName in fileNames:
             if fileName.endswith('.wav') or fileName.endswith('.mp4') or fileName.endswith('.tonic') or fileName.endswith('-pitch.csv'):
                 shutil.copy(os.path.join(root, fileName), checkPath(os.path.join(root, fileName).replace(orig, new)))