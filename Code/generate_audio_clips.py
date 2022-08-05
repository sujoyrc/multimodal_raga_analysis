import librosa
import soundfile as sf
import pandas as pd
import sys
sys.path.append('../../CommonScripts/')
from common_utils import checkPath

summaryFile = '../Seqs/summary.csv'
OrigAudioOutput = '../Audio/OrigAudioSeqs/'
SSAudioOutput = '../Audio/SSAudioSeqs/'

for _, row in pd.read_csv(summaryFile).iterrows():

    print(row['filename'])
    
    origAudioPath = row["filename"].rsplit('-', 1)[0] + ".wav"
    y, sr = librosa.load(origAudioPath, sr=None, offset=row['start_times'], duration=12)
    sf.write(checkPath(OrigAudioOutput + row['unique_id'] + '.wav'), y, sr)

    ssAudioPath = row["filename"].rsplit('-', 1)[0] + "-SS.wav"
    y, sr = librosa.load(ssAudioPath, sr=None, offset=row['start_times'], duration=12)
    sf.write(checkPath(SSAudioOutput + row['unique_id'] + '.wav'), y, sr)
