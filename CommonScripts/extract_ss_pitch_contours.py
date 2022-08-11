from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.separator import Separator
import numpy as np
from scipy.signal import resample
import soundfile as sf
import parselmouth
import math
import pandas as pd
import random
import os
import librosa
from common_utils import addBack, checkPath
import warnings
import sys
import pdb

def source_separate(src, dest, sample_rate_init=44100, sample_rate_new=16000, offset=0, duration=None):
        '''
        Source Separate voice from the audio recording using Spleeter:4stems

        Parameters
            src (str): source audio file
            dest (str): destination audio file
            sample_rate_init (int): sample rate to load the original audio at
            sample_rate_new (int): sample rate to store the source separated audio file in
        
        Returns
            waveform: source separated audio
        '''
        audio_loader = get_default_audio_adapter()
        waveform, _ = audio_loader.load(src, sample_rate=sample_rate_init, offset=offset, duration=duration)
        separator = Separator('spleeter:4stems', multiprocess=True)
        waveform = separator.separate(waveform)
        
        #convert vocal audio to mono
        waveform = np.mean(waveform['vocals'], axis=1)
        
        #downsample audio to 16kHz
        ratio = float(sample_rate_new)/sample_rate_init
        n_samples = int(np.ceil(waveform.shape[-1]*ratio))
        waveform = resample(waveform, n_samples, axis=-1)
        # set the first and last 0.5 seconds to silence
        waveform[:math.floor(sample_rate_new/4)] = 0
        waveform[-math.floor(sample_rate_new/4):] = 0
        if dest is not None:
            sf.write(dest, waveform, sample_rate_new, subtype='PCM_16')
        separator.__del__()

        return waveform

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

def process(srcFolder, tonicFolder=None, ssDestFolder=None, pitchDestFolder=None, normalize=False, k=100):
        '''Parse through files a folder. All processed files will be placed in the same folder
        
        Parameters
            srcFolder (str): folder with audio file
            tonicFolder (str): folder with tonic files; if None, assumed to be in the srcFolder itself
            ssDestFolder (str): folder to store SS in
            pitchDestFolder (str): folder to stor pitch contour in
            normalize (bool): if true will normalize pitch contour
            k (int): number of divisions per semitone; used only if normalize is true
        Returns
            None
        '''
        
        for root, _, fileNames in os.walk(os.path.join(srcFolder)):
            for fileName in fileNames:
                if fileName.endswith('mp3') or fileName.endswith('.wav') or fileName.endswith('mp4'):
                    if fileName.endswith('-SS.wav'):
                        # source separated audio
                        continue
                    print('Processing ' + os.path.join(fileName))
                    # pdb.set_trace()
                    # check if audio file is being used
                    if tonicFolder is None:
                        tonicFile = checkPath(os.path.join(root, fileName).rsplit('.', 1) [0] + '.tonic')
                    else:
                        tonicFile = checkPath(os.path.join(root, fileName).replace(addBack(srcFolder), addBack(tonicFolder)).rsplit('.', 1)[0] + '.tonic')
                    # check if tonic file exists
                    # if not os.path.isfile(tonicFile):
                    #     warnings.warn(f'Tonic not found at {tonicFile}. Skipping File: {os.path.join(root, fileName)}')
                    #     continue
                    # else:
                    #     with open(tonicFile, 'r') as f:
                    #         tonic = float(f.read())
                    offsetFile = os.path.join(root, fileName).rsplit('.', 1)[0] + '.offset'     # contains timesteps to read audio at
                    if not os.path.isfile(offsetFile):
                        # read the full song; i.e. feed default params for offset and duration into file
                        if ssDestFolder is None:
                            ssDestF = srcFolder
                        else:
                            ssDestF = ssDestFolder
    
                        if pitchDestFolder is None:
                            if ssDestFolder is not None:
                                pDestF = ssDestFolder
                            else:
                                pDestF = srcFolder
                        else:
                            pDestF = pitchDestFolder
                        
                        # source separate
                        ssDest = checkPath(os.path.join(root, fileName).replace(addBack(srcFolder), addBack(ssDestF)).rsplit('.', 1)[0] + '-SS.wav')
                        if not os.path.isfile(ssDest):
                            source_separate(os.path.join(root, fileName), ssDest, 48000)

                        # pitch extraction
                        # pitchDest = checkPath(os.path.join(root, fileName).replace(addBack(srcFolder), addBack(pDestF)).rsplit('.', 1)[0] + '-pitch.csv')
                        # if not os.path.isfile(pitchDest):
                        #     pitch_contour(src=ssDest, dest=pitchDest, tonic=tonic, normalize=normalize, k=k)
                        
                    else:
                        lines = open(offsetFile, 'r').readlines()
                        for i, line in enumerate(lines):
                            start, end = line.strip().split('\t')

                            # source separate
                            ssDest = os.path.join(root, fileName).replace(addBack(srcFolder), addBack(ssDestFolder)).rsplit('.', 1)[0] + f'-SS{i}.wav'
                            if not os.path.isfile(ssDest):
                                source_separate(os.path.join(ssDestF, fileName), ssDest, offset=float(start), duration=float(end)-float(start))

                            # pitch extraction
                            # pitchDest = ssDest = os.path.join(root, fileName).replace(addBack(srcFolder), addBack(pitchDestFolder)).rsplit('.', 1)[0] + f'-pitch{i}.csv'
                            # if not os.path.isfile(pitchDest):
                            #     pitch_contour(src=ssDest, dest=pitchDest, tonic=tonic, normalize=normalize, k=k)

        return None

if __name__ == "__main__":
    args = sys.argv[1:]
    process(*args)
