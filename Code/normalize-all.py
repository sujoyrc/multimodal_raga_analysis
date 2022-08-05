import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
sys.path.append('../../CommonScripts/')
from utils import checkPath

srcFolder = '../Seqs_Jin-Normal/easy-1/'
destFolder = checkPath('../Seqs_Jin-Normalized-all/easy-1/')
min_val = -550
max_val = 1950

for root, _, fileNames in os.walk(srcFolder):
    for fileName in fileNames:
        if fileName.startswith('train') or fileName.startswith('test'):
            df = pd.read_csv(os.path.join(root, fileName))
            df.iloc[:, :-1] = 0 + ((df.iloc[:, :-1] - min_val)/(max_val - min_val))
            df.to_csv(os.path.join(destFolder, fileName), index=False)