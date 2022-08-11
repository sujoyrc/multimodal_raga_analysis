import numpy as np
import parselmouth
import math
import pandas as pd
import os
from common_utils import addBack, checkPath
import sys

def pitch_contour(src=None, dest=None, tonic=None, k=100, sample_rate=16000, normalize=False, time_step=0.01):
        '''
        Returns a normalized pitch contour at 10 ms intervals

        Parameters
            src (str): loaded audio or source audio file
            dest (str): destination file for contour
            tonic (float): tonic of audio (in Hz)
            k (int): number of divisions per semitone; default to 5
            sample_rate (int): sample rate to load audio in
            normalize (bool): if True, then the pitch contour will be normalised w.r.t. the tonic
            time_step (float): time interval (in s) at which the pitch is going to be extracted

        Returns
            pitchDf (pd.DataFrame): dataframe with time, normalised pitch, energy values

        '''
        if type(src) is np.ndarray:
            # loaded audio
            snd = parselmouth.Sound(src, sample_rate, 0)
        else:
            # audio file name
            snd = parselmouth.Sound(src)
        min_pitch = tonic*(2**(-5/12)) #lower Pa
        max_pitch = tonic*(2**(19/12))  #higher Pa

        pitch = snd.to_pitch_ac(time_step, min_pitch, 15, True, 0.03, 0.45, 0.01, 0.9, 0.14, max_pitch)
        inten = snd.to_intensity(50, time_step, False)

        timestamps = np.arange(0, snd.duration, time_step)
        pitchvals = []
        intensityvals = []
        for t in timestamps:
            pitchvals.append(pitch.get_value_at_time(t) if not math.isnan(pitch.get_value_at_time(t)) else 0)
            intensityvals.append(inten.get_value(t) if not math.isnan(inten.get_value(t)) else 0)

        df = pd.DataFrame(columns=['time', 'pitch', 'energy'])
        for i, f in enumerate(pitchvals):
            df = df.append({'time': timestamps[i],
                            'pitch': f,
                            'energy': intensityvals[i]
                            }, ignore_index=True)
        # silence first and last second, to coverup distortion due to ss
        df.iloc[:int(1/time_step), 1] = 0
        df.iloc[-int(1/time_step):, 1] = 0
        # pdb.set_trace()
        if normalize:
            df = normalize_pitch(df, tonic, k)  # normalize pitch values
        if dest is not None:
            df.to_csv(dest, header=True, index=False)
        return df

def normalize_pitch(df, tonic, k):
        '''
        Replaces the pitch values from dataframe df to normalised pitch values

        Parameters
            df (pd.DataFrame): dataframe with t, p (in Hz) and e values
            tonic (float): tonic in Hz
            k (int): number of divisions per semitone
        
        Returns
            df (pd.DataFrame): dataframe with t, p (normalised) and e values
        '''
        frequency_normalized = [np.round_(1200*math.log2(f/tonic)*(k/100)) if f>0 else -3000 for f in df['pitch']]
        df['pitch'] = frequency_normalized
        return df

def process(tonicFolder=None, ssDestFolder=None, pitchDestFolder=None, normalize=False, k=100):
        '''Parse through files a folder. All processed files will be placed in the same folder
        
        Parameters
            tonicFolder (str): folder with tonic files; if None, assumed to be in the srcFolder itself
            ssDestFolder (str): folder to store SS in
            pitchDestFolder (str): folder to stor pitch contour in
            normalize (bool): if true will normalize pitch contour
            k (int): number of divisions per semitone; used only if normalize is true
        Returns
            None
        '''
        normalize = bool(int(normalize))
        for root, _, fileNames in os.walk(os.path.join(ssDestFolder)):
            for fileName in fileNames:
                if fileName.endswith('mp3') or fileName.endswith('-SS.wav') or fileName.endswith('mp4'):
                    print('Processing ' + os.path.join(fileName))
                    
                    try:
                        tonicFile = checkPath(os.path.join(root, fileName).replace(addBack(ssDestFolder), addBack(tonicFolder)).rsplit('-', 1)[0] + '.tonic')   # <song name>-SS.wav is changed to <song name>.tonic and set as tonicFile variable
                        with open(tonicFile, 'r') as t:
                            tonicVal = float(t.read())
                    except:
                        raise Exception('tonicFolder has to be provided')
                    
                    # pitch extraction
                    pitchDest = checkPath(os.path.join(root, fileName).replace(addBack(ssDestFolder), addBack(pitchDestFolder)).rsplit('.', 1)[0] + '-pitch.csv')
                    print(pitchDest)
                    if not os.path.isfile(pitchDest):
                        pitch_contour(src=os.path.join(root, fileName), dest=pitchDest, tonic=tonicVal, normalize=normalize, k=k)

        return None

if __name__ == "__main__":
    args = sys.argv[1:]
    process(*args)
