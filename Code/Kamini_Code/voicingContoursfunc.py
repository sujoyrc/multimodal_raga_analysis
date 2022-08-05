import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import interpolate
from scipy import signal
import sys
# import tryFunc
import temporalsmoothing
import json
import pickle
import functionsforVoicingFeaturesNew as featfile

# audiolist=[
# '/home/kamini/Children_Speech_Prosody/ASERDataExp/pitchAudios/pcy_27122016_1_cs_e2_2.wav',\
# '/home/kamini/Children_Speech_Prosody/ASERDataExp/pitchAudios/areeb-ansari_07-b-010_dfs_thane_25112019-105101-1_m005_2.wav',\
# '/home/kamini/Children_Speech_Prosody/ASERDataExp/pitchAudios/anushkargaikwad05d019_11122018_1_vjhs_h011_2.wav',\
# '/home/kamini/Children_Speech_Prosody/ASERDataExp/pitchAudios/avnivijay05b018_15102018_1_vjhs_m002_2.wav',\
# '/home/kamini/Children_Speech_Prosody/ASERDataExp/pitchAudios/avnivijay05b018_15102018_1_vjhs_m002_3.wav',\
# '/home/kamini/Children_Speech_Prosody/ASERDataExp/pitchAudios/motul-suresh-hanamantha_07-b-020_cs_powai_14102019-095629-1_s017_2.wav'\
# ]
# featurefolder='../featuresNew/voicing/'
# for audio in audiolist:
# audio=sys.argv[1]
# modelFolder=sys.argv[2]
# outfolder=sys.argv[3]
def voicingContours(audio,modelFolder,outfolder):
	name=audio.split('/')[-1][:-4] 
	print(name)
	data=featfile.voicingfeaturesCompute(audio,winSize=0.02,hopSize=0.01)
	data.to_csv(outfolder+name+'_voicing.csv',float_format="%.5f",columns=['time','zeroCross','intensity','autoCorr','LPC1','error','bandRatio','HNR'],index_label='centerFrame')
	# data.to_csv(featurefolder+name+'_voicingfeaturesNew.csv',float_format="%.5f",columns=['time','zeroCross','intensity','autoCorr','LPC1','error'],index_label='centerFrame')
	featurelist=['zeroCross','intensity','autoCorr','LPC1','error','bandRatio','HNR']
	data.fillna(0,inplace=True)
	X=np.array(data[featurelist])
	# modelName='voicingNewRFmodelFull3class.pkl'
	modelName=modelFolder+'voicingRFmodel0.pkl'
	RFmodel=pickle.load(open( modelName, "rb" ))
	voicingCon=np.array(RFmodel.predict(X).tolist())
	voicingCon=voicingCon.astype('<U3')
	voicingCon[data['intensity']<-30]='SIL'
	voicingCon[data['HNR']<-150]='uv'
	data['voicingpred']=voicingCon
	voicingCon[voicingCon=='uv']=0
	voicingCon[voicingCon=='SIL']=0
	voicingCon[voicingCon=='v']=1
	voicingCon=voicingCon.astype(int)
	voicingsmooth = temporalsmoothing.post_proc(voicingCon,10,30,50) # 10ms is hop size, 30ms is unvoiced_threshold, 50ms is voiced_threshold
	data['voicingpredbinary']=voicingCon
	data['voicingpredsmooth']=voicingsmooth
	data.to_csv(outfolder+name+'_voicingPred.csv',float_format="%.5f",columns=['time','voicingpred','voicingpredbinary','voicingpredsmooth'],index_label='centerFrame')
	# break

