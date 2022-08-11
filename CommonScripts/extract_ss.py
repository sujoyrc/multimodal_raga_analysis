from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.separator import Separator
import numpy as np
from scipy.signal import resample
import soundfile as sf
import math
import os
from common_utils import addBack, checkPath
import sys

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

def process(srcFolder, ssDestFolder=None, sr=44100):
        '''Parse through files a folder. All processed files will be placed in the same folder
        
        Parameters
            srcFolder (str): folder with audio file
            ssDestFolder (str): folder to store SS in
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
            
                    offsetFile = os.path.join(root, fileName).rsplit('.', 1)[0] + '.offset'     # contains timesteps to read audio at
                    if not os.path.isfile(offsetFile):
                        # read the full song; i.e. feed default params for offset and duration into file
                        if ssDestFolder is None:
                            ssDestF = srcFolder
                        else:
                            ssDestF = ssDestFolder
                        
                        # source separate
                        ssDest = checkPath(os.path.join(root, fileName).replace(addBack(srcFolder), addBack(ssDestF)).rsplit('.', 1)[0] + '-SS.wav')
                        if not os.path.isfile(ssDest):
                            source_separate(os.path.join(root, fileName), ssDest, sr)
                    else:
                        lines = open(offsetFile, 'r').readlines()
                        for i, line in enumerate(lines):
                            start, end = line.strip().split('\t')

                            # source separate
                            ssDest = os.path.join(root, fileName).replace(addBack(srcFolder), addBack(ssDestFolder)).rsplit('.', 1)[0] + f'-SS{i}.wav'
                            if not os.path.isfile(ssDest):
                                source_separate(os.path.join(ssDestF, fileName), ssDest, offset=float(start), duration=float(end)-float(start))

        return None

if __name__ == "__main__":
    args = sys.argv[1:]
    process(*args)
