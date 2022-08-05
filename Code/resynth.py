import sys
sys.path.append('../../CommonScripts/')
from resynthesise_contours import fileParse

srcFolder = '../PitchInter-New/'
tonicFolder = '../Data/'
includeFileList = ['/home/nithya/Projects/Gesture Analysis/PitchInter-New/Alap/CC_3a_MM/CC_3a_MM-pitch.csv']
fileParse(srcFolder, tonicFolder=tonicFolder, normalised=True, includeFileList=includeFileList)